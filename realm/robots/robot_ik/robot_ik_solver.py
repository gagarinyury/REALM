"""Cartesian-velocity-to-joint-velocity IK solver for the DROID/Franka arm.

Original implementation (upstream REALM) delegated this to
`dm_robotics.moma.effectors.cartesian_6d_velocity_effector`, which wraps a
compiled C++ solver: `dm_robotics/controllers/lsqp/cartesian_6d_to_joint_velocity_mapper`
(a two-level hierarchical least-squares QP -- see
https://github.com/google-deepmind/dm_robotics/tree/main/cpp/controllers/lsqp).

`dm-robotics-controllers` (the compiled package) ships manylinux wheels only --
no Windows wheel and no sdist -- which transitively blocks `dm-robotics-moma`
and `dm-robotics-manipulation` from installing on Windows at all (pip refuses
the whole dependency chain). Since our REALM baseline is being reproduced on
native Windows (see PLAN.md), this file reimplements the *same* mathematical
formulation directly against `dm_control.mjcf` (already Windows-compatible)
using the `osqp` QP solver (has Windows wheels), rather than reimplementing
DeepMind's ADMM solver internals from scratch. The formulation is transcribed
directly from the upstream C++ sources
(cartesian_6d_velocity_task.cc, joint_position_limit_constraint.cc,
cartesian_6d_to_joint_velocity_mapper.cc), not guessed:

  Primary QP (single combined weighted-least-squares level):
      minimize_qdot  || W (J qdot - v_target) ||^2  +  regularization_weight * ||qdot||^2
      s.t.           lower_bound <= qdot <= upper_bound

  where W is the 6x6 Cartesian task weighting matrix (identity here -- REALM
  does not override it), J is the 6xN site Jacobian, and the box bounds are
  the tighter of (a) the joint velocity magnitude limits and (b) a joint
  position limit converted to a velocity bound via:
      gain = joint_position_limit_velocity_scale / control_timestep_seconds
      upper_bound_i = gain * (hi_lim - qpos_i)   if that distance > 0, else 0
      lower_bound_i = -gain * (qpos_i - lo_lim)  if that distance > 0, else 0
  (hi_lim/lo_lim are the joint range shrunk by `minimum_distance_from_joint_position_limit`).

  Secondary (nullspace) objective -- upstream solves this as a second,
  lower-priority QP level; here it is applied via the standard robotics
  "task-priority" nullspace projection (Siciliano et al.), which is the
  textbook equivalent construction for a soft secondary objective:
      qdot_null_bias = nullspace_gain * (q_ref - q_current) / control_timestep_seconds
      N = I - J_w^+ J_w                      (nullspace projector of the primary task)
      qdot_final = clip(qdot_primary + N @ qdot_null_bias, lower_bound, upper_bound)

This is documented explicitly as a platform-compatibility substitution in the
thesis methodology -- see chapters/methodology.tex -- not silently swapped.
"""

import numpy as np
import osqp
import scipy.sparse as sp
import mujoco
from dm_control import mjcf

from realm.robots.robot_ik.simple_arm import SimpleFrankaArm


class RobotIKSolver:
    def __init__(self):
        self.relative_max_joint_delta = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.max_joint_delta = self.relative_max_joint_delta.max()
        self.max_gripper_delta = 0.25
        self.max_lin_delta = 0.075
        self.max_rot_delta = 0.15
        self.control_hz = 15

        self._arm = SimpleFrankaArm()
        self._physics = mjcf.Physics.from_mjcf_model(self._arm.mjcf_model)

        self._n_dof = len(self._arm.joints)
        self._joint_qpos_adr = np.array(
            [self._physics.bind(j).qposadr for j in self._arm.joints]
        ).reshape(-1)
        self._wrist_site_id = self._physics.bind(self._arm.wrist_site).element_id

        # Joint position limits (mujoco range), shrunk by the same safety margin
        # dm_robotics uses.
        jnt_range = self._physics.bind(self._arm.joints).range
        self._jnt_lo = np.array(jnt_range[:, 0], dtype=float)
        self._jnt_hi = np.array(jnt_range[:, 1], dtype=float)

        # --- control params (transcribed 1:1 from the original ControlParams call) ---
        self.control_timestep_seconds = 1.0 / self.control_hz
        self.nullspace_joint_position_reference = np.zeros(self._n_dof)
        self.nullspace_gain = 0.025
        self.regularization_weight = 1e-2
        self.minimum_distance_from_joint_position_limit = 0.3
        self.joint_position_limit_velocity_scale = 0.95
        self._pos_limit_gain = (
            self.joint_position_limit_velocity_scale / self.control_timestep_seconds
        )

        self._qp = osqp.OSQP()
        self._qp_initialized = False

    # ------------------------------------------------------------------
    # Core IK: Cartesian 6D velocity -> joint velocity
    # ------------------------------------------------------------------
    def _compute_jacobian(self):
        """6xN site Jacobian (3 translational rows, 3 rotational rows) w.r.t. this arm's DOFs."""
        model = self._physics.model.ptr
        data = self._physics.data.ptr
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, self._wrist_site_id)
        dof_idx = self._physics.bind(self._arm.joints).dofadr
        J = np.zeros((6, self._n_dof))
        J[:3, :] = jacp[:, dof_idx]
        J[3:, :] = jacr[:, dof_idx]
        return J

    def _joint_velocity_box_constraints(self, qpos):
        """Per-joint [lower, upper] qdot bounds = tighter of velocity limit and
        position-limit-derived velocity bound (see module docstring)."""
        vel_lo = -np.abs(self.relative_max_joint_delta)
        vel_hi = np.abs(self.relative_max_joint_delta)

        hi_lim = self._jnt_hi - self.minimum_distance_from_joint_position_limit
        lo_lim = self._jnt_lo + self.minimum_distance_from_joint_position_limit
        hi_lim_dist = hi_lim - qpos
        lo_lim_dist = qpos - lo_lim
        valid = hi_lim > lo_lim

        pos_hi = np.where((hi_lim_dist > 0) & valid, self._pos_limit_gain * hi_lim_dist, 0.0)
        pos_lo = np.where((lo_lim_dist > 0) & valid, -self._pos_limit_gain * lo_lim_dist, 0.0)

        lower = np.maximum(vel_lo, pos_lo)
        upper = np.minimum(vel_hi, pos_hi)
        # Guard against numerically inverted bounds (e.g. right at a limit).
        lower = np.minimum(lower, upper)
        return lower, upper

    def _solve_primary_qp(self, J, v_target, lower, upper):
        # minimize ||J qdot - v||^2 + reg * ||qdot||^2
        #   = qdot^T (J^T J + reg I) qdot - 2 v^T J qdot + const
        n = self._n_dof
        P = 2.0 * (J.T @ J + self.regularization_weight * np.eye(n))
        q = -2.0 * (J.T @ v_target)
        P_sparse = sp.csc_matrix((P + P.T) / 2.0)  # ensure exact symmetry
        A = sp.eye(n, format="csc")

        self._qp = osqp.OSQP()
        self._qp.setup(
            P=P_sparse,
            q=q,
            A=A,
            l=lower,
            u=upper,
            verbose=False,
            polish=True,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=4000,
        )
        result = self._qp.solve()
        if result.x is None or np.any(np.isnan(result.x)):
            # Infeasible/failed solve -- fall back to a clipped damped-least-squares
            # solution rather than propagating NaNs into the simulator.
            qdot = np.linalg.lstsq(
                J.T @ J + self.regularization_weight * np.eye(n), J.T @ v_target, rcond=None
            )[0]
            return np.clip(qdot, lower, upper)
        return result.x

    def cartesian_velocity_to_joint_velocity(self, cartesian_velocity, robot_state):
        cartesian_delta = self.cartesian_velocity_to_delta(cartesian_velocity)
        qpos = np.array(robot_state["joint_positions"])
        qvel = np.array(robot_state["joint_velocities"])

        self._arm.update_state(self._physics, qpos, qvel)
        mujoco.mj_forward(self._physics.model.ptr, self._physics.data.ptr)

        J = self._compute_jacobian()
        lower, upper = self._joint_velocity_box_constraints(qpos)

        qdot_primary = self._solve_primary_qp(J, cartesian_delta, lower, upper)

        # Nullspace secondary objective: pull joints toward the reference posture,
        # projected into the nullspace of the primary (Cartesian tracking) task.
        J_pinv = np.linalg.pinv(J, rcond=1e-4)
        N = np.eye(self._n_dof) - J_pinv @ J
        qdot_null_bias = (
            self.nullspace_gain
            * (self.nullspace_joint_position_reference - qpos)
            / self.control_timestep_seconds
        )
        qdot = qdot_primary + N @ qdot_null_bias
        qdot = np.clip(qdot, lower, upper)

        joint_delta = qdot.copy()
        np.any(joint_delta)

        joint_velocity = self.joint_delta_to_velocity(joint_delta)
        return joint_velocity

    # ------------------------------------------------------------------
    # Velocity <-> delta conversions (unchanged from upstream)
    # ------------------------------------------------------------------
    def gripper_velocity_to_delta(self, gripper_velocity):
        gripper_vel_norm = np.linalg.norm(gripper_velocity)
        if gripper_vel_norm > 1:
            gripper_velocity = gripper_velocity / gripper_vel_norm
        return gripper_velocity * self.max_gripper_delta

    def cartesian_velocity_to_delta(self, cartesian_velocity):
        if isinstance(cartesian_velocity, list):
            cartesian_velocity = np.array(cartesian_velocity)
        lin_vel, rot_vel = cartesian_velocity[:3], cartesian_velocity[3:6]
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        if lin_vel_norm > 1:
            lin_vel = lin_vel / lin_vel_norm
        if rot_vel_norm > 1:
            rot_vel = rot_vel / rot_vel_norm
        lin_delta = lin_vel * self.max_lin_delta
        rot_delta = rot_vel * self.max_rot_delta
        return np.concatenate([lin_delta, rot_delta])

    def joint_velocity_to_delta(self, joint_velocity):
        if isinstance(joint_velocity, list):
            joint_velocity = np.array(joint_velocity)
        relative_max_joint_vel = self.joint_delta_to_velocity(self.relative_max_joint_delta)
        max_joint_vel_norm = (np.abs(joint_velocity) / relative_max_joint_vel).max()
        if max_joint_vel_norm > 1:
            joint_velocity = joint_velocity / max_joint_vel_norm
        return joint_velocity * self.max_joint_delta

    def gripper_delta_to_velocity(self, gripper_delta):
        return gripper_delta / self.max_gripper_delta

    def cartesian_delta_to_velocity(self, cartesian_delta):
        if isinstance(cartesian_delta, list):
            cartesian_delta = np.array(cartesian_delta)
        cartesian_velocity = np.zeros_like(cartesian_delta)
        cartesian_velocity[:3] = cartesian_delta[:3] / self.max_lin_delta
        cartesian_velocity[3:6] = cartesian_delta[3:6] / self.max_rot_delta
        return cartesian_velocity

    def joint_delta_to_velocity(self, joint_delta):
        if isinstance(joint_delta, list):
            joint_delta = np.array(joint_delta)
        return joint_delta / self.max_joint_delta
