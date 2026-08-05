import torch as th
import math
from omnigibson.controllers.controller_base import (
    BaseController,
    ControlType,
    GripperController,
    IsGraspingState,
    LocomotionController,
    ManipulationController,
)
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import ControllableObjectViewAPI
import omnigibson as og  # For og.sim.device
from omnigibson.macros import gm
#import pinocchio as pin
import numpy as np
import os


# Create module logger
log = create_module_logger(module_name=__name__)

# NOTE: patched for native-Windows execution against a current OmniGibson (see PLAN.md /
# thesis Methodology for the full rationale). Verified against franka's own model definition
# file (omnigibson-robot-assets/models/franka/franka.yaml, end_effectors.gripper.eef_link_names)
# -- not guessed. Only correct for the default "gripper" end-effector variant used here.
EEF_LINK_NAME = "eef_link"


class IndividualJointPDController(LocomotionController, ManipulationController, GripperController):
    def __init__(
            self,
            control_freq,
            motor_type,  # This will be forced to 'effort' for hybrid control
            control_limits,
            dof_idx,
            command_input_limits="default",
            command_output_limits="default",
            Kq=None,  # Kq: Can be scalar, list, or torch.Tensor
            Kqd=None,  # For Kqd: Can be scalar, list, or torch.Tensor
            Kx=None,  # Kx: Cartesian P gain (scalar, list (for diagonal), or 6x6 tensor)
            Kxd=None,  # Kxd: Cartesian D gain (scalar, list (for diagonal), or 6x6 tensor)
            use_impedances=False,
            use_gravity_compensation=False,
            use_cc_compensation=True,
            use_delta_commands=False,  # Delta commands are less common for torque control
            compute_delta_in_quat_space=None,  # Delta commands are less common for torque control
            max_effort=None,
            min_effort=None
    ):
        motor_type = "effort"
        self._motor_type = motor_type.lower()
        self._use_impedances = True

        self.max_effort = None if max_effort is None else th.tensor(max_effort).to(og.sim.device)
        self.min_effort = None if min_effort is None else th.tensor(min_effort).to(og.sim.device)

        self._use_gravity_compensation = use_gravity_compensation
        self._use_cc_compensation = use_cc_compensation

        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )

        Kq = self._diagonalize_gain(self._to_tensor(Kq))
        Kqd = self._diagonalize_gain(self._to_tensor(Kqd))
        assert Kq.shape == Kqd.shape
        Kx = self._diagonalize_gain(self._to_tensor(Kx))
        Kxd = self._diagonalize_gain(self._to_tensor(Kxd))
        assert Kx.shape == th.Size([6, 6])
        assert Kxd.shape == th.Size([6, 6])

        self.Kq = th.nn.Parameter(Kq).to(og.sim.device)
        self.Kqd = th.nn.Parameter(Kqd).to(og.sim.device)
        self.Kx = th.nn.Parameter(Kx).to(og.sim.device)
        self.Kxd = th.nn.Parameter(Kxd).to(og.sim.device)

        urdf_path = f"/app/realm/robots/panda_robotiq/panda_arm.urdf"
        self.time_tracker = -1 # we update at the very beginning of compute_control, so this is 0 when controller is queried for the very first time
        self.cached_torque = None

    def _update_goal(self, controller_idx, command):
        # NOTE: patched signature (controller_idx, command) instead of (command, control_dict).
        # `current_joint_pos` was fetched in the original but never actually used below (dead
        # code in the upstream REALM implementation) -- kept for fidelity to the original
        # structure, now sourced from ControllableObjectViewAPI instead of control_dict.
        target_joint_pos = command.to(og.sim.device)

        target_joint_pos = target_joint_pos.clip(
            self._control_limits[ControlType.get_type("position")][0][self.dof_idx],
            self._control_limits[ControlType.get_type("position")][1][self.dof_idx],
        )

        prim_path = self._articulation_root_paths[controller_idx]
        # NOTE: ControllableObjectViewAPI may return numpy arrays depending on the active compute
        # backend; wrap with th.as_tensor() before any torch-only ops (.to(), indexing on a th
        # tensor still works on numpy arrays via __getitem__, but .to() does not).
        current_joint_pos = th.as_tensor(ControllableObjectViewAPI.get_joint_positions(prim_path))[
            self.dof_idx
        ].to(og.sim.device)
        target_joint_vel = th.zeros_like(target_joint_pos)

        return dict(target_joint_pos=target_joint_pos, target_joint_vel=target_joint_vel)

    def compute_control(self, goals):
        """
        NOTE: patched -- `compute_control(self, goal_dict, control_dict)` -> `compute_control(self,
        goals)`. The controller framework now always calls this batched over this group's N active
        members (here N=1, but the array shapes below are still (N, ...) since that's what the
        framework passes/expects back); current robot state comes from `ControllableObjectViewAPI`
        via `self.routing_path`/`self.view_row_indices` instead of a `control_dict` parameter.
        The actual impedance-control math (Kp = J^T Kx J + Kq, etc.) is unchanged from REALM's
        original implementation, just looped per batch row instead of operating on a single
        (non-batched) set of tensors.
        """
        self.time_tracker += 1

        rows = self.view_row_indices
        routing_path = self.routing_path

        joint_pos_desired = goals["target_joint_pos"].to(og.sim.device)  # (N, 7)
        joint_vel_desired = goals["target_joint_vel"].to(og.sim.device)  # (N, 7)

        all_joint_positions = th.as_tensor(ControllableObjectViewAPI.get_all_joint_positions(routing_path))[
            rows, :
        ]
        all_joint_velocities = th.as_tensor(
            ControllableObjectViewAPI.get_all_joint_velocities(routing_path, estimate=True)
        )[rows, :]
        current_joint_pos = all_joint_positions[:, self.dof_idx].to(og.sim.device)  # (N, 7)
        current_joint_vel = all_joint_velocities[:, self.dof_idx].to(og.sim.device)  # (N, 7)

        # Batched relative Jacobians for ALL links of ALL group members: (N_view, n_links, 6, n_dof_total).
        all_jac = th.as_tensor(ControllableObjectViewAPI.get_all_relative_jacobians(routing_path))
        eef_link_idx = ControllableObjectViewAPI.get_link_index(routing_path, EEF_LINK_NAME)
        jac_row = eef_link_idx - 1  # Jacobian excludes root body (index 0), per ik_controller.py/osc_controller.py.
        # Select this group's rows, the eef link's row, and this controller's dof columns -> (N, 6, 7).
        jacobian_batch = all_jac[rows, jac_row][:, :, self.dof_idx].to(og.sim.device)

        u = th.zeros_like(current_joint_pos)
        for n in range(jacobian_batch.shape[0]):
            jacobian = jacobian_batch[n]
            assert jacobian.shape == (6, 7)

            Kp = jacobian.T @ self.Kx @ jacobian + self.Kq
            Kd = jacobian.T @ self.Kxd @ jacobian + self.Kqd

            u_feedback = Kp @ (joint_pos_desired[n] - current_joint_pos[n]) + Kd @ (
                joint_vel_desired[n] - current_joint_vel[n]
            )
            u[n] = u_feedback

        # Add Coriolis / centrifugal compensation
        if self._use_cc_compensation:
            all_cc = th.as_tensor(
                ControllableObjectViewAPI.get_all_coriolis_and_centrifugal_compensation_forces(routing_path)
            )[rows, :]
            u = u + all_cc[:, self.dof_idx].to(og.sim.device)

        if self.min_effort is not None and self.max_effort is not None:
            u = u.clip(self.min_effort, self.max_effort)

        return u

    # NOTE: patched -- removed the custom clip_control() override entirely. REALM's original
    # implementation was written for the pre-refactor single-instance (non-batched) control
    # convention and crashed under the current batched (N, control_dim) shape. It was also
    # functionally a no-op beyond clipping: `idx = [True] * self.control_dim` selects every
    # element unconditionally, so `control_copy[idx] = clipped_control[idx]` always reduces to
    # `return clipped_control`. BaseController.clip_control (controller_base.py) already
    # implements exactly this -- batched, using precomputed self._clip_lo/self._clip_hi -- and
    # its POSITION-only "undo clip for unlimited joints" branch never triggers here since this
    # controller's control_type is EFFORT. So inheriting it is behaviorally equivalent, just
    # correctly batched.

    def compute_no_op_goal(self, controller_idx):
        # NOTE: patched -- was compute_no_op_goal(self, control_dict).
        prim_path = self._articulation_root_paths[controller_idx]
        target_joint_pos = th.as_tensor(ControllableObjectViewAPI.get_joint_positions(prim_path))[
            self.dof_idx
        ].to(og.sim.device)
        target_joint_vel = th.zeros_like(target_joint_pos)

        return dict(target_joint_pos=target_joint_pos, target_joint_vel=target_joint_vel)

    def _compute_no_op_action(self, controller_idx):
        # NOTE: patched signature -- was _compute_no_op_action(self, control_dict); control_dict
        # was never used here.
        return th.zeros(self.command_dim, device=og.sim.device)

    def _get_goal_shapes(self):
        return dict(
            target_joint_pos=(self.control_dim,),
            target_joint_vel=(self.control_dim,)
        )

    def _to_tensor(self, input):
        if th.is_tensor(input):
            return input.to(th.Tensor())
        else:
            return th.tensor(input).to(th.Tensor())

    def _diagonalize_gain(self, gain: th.Tensor) -> th.Tensor:
        if gain.dim() == 1:
            return th.diag(gain)
        elif gain.dim() == 2:
            return gain
        else:
            raise ValueError(f"Gain tensor must be 1D or 2D, but got {gain.dim()}D.")

    def is_grasping(self, controller_idx=0):
        # NOTE: patched signature -- added controller_idx=0 (base class now always calls with an
        # index); behavior unchanged (this controller never reports a grasp state).
        return IsGraspingState.UNKNOWN

    @property
    def motor_type(self):
        return self._motor_type

    @property
    def control_type(self):
        return ControlType.EFFORT

    @property
    def command_dim(self):
        return len(self.dof_idx)
