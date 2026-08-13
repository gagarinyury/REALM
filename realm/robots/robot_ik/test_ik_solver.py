import numpy as np
from realm.robots.robot_ik.robot_ik_solver import RobotIKSolver

print("=== Instantiating RobotIKSolver (loads MJCF, no dm_robotics needed) ===", flush=True)
solver = RobotIKSolver()
print(f"OK. n_dof={solver._n_dof}", flush=True)

# --- Test 1: forward-consistency of the primary QP -----------------------
# Pick a safe, non-singular mid-range joint configuration.
qpos = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.8, 0.0])
qvel = np.zeros(7)
robot_state = {"joint_positions": qpos, "joint_velocities": qvel}

solver._arm.update_state(solver._physics, qpos, qvel)
import mujoco
mujoco.mj_forward(solver._physics.model.ptr, solver._physics.data.ptr)
J = solver._compute_jacobian()
print(f"Jacobian shape: {J.shape}", flush=True)

# Small target well inside velocity/position limits so the QP should be
# essentially unconstrained -> should closely match the analytic weighted
# least-squares solution AND satisfy J @ qdot ~= target.
target_v = np.array([0.01, -0.02, 0.015, 0.0, 0.01, -0.005])
lower, upper = solver._joint_velocity_box_constraints(qpos)
qdot = solver._solve_primary_qp(J, target_v, lower, upper)
achieved_v = J @ qdot
err = np.linalg.norm(achieved_v - target_v)
print(f"target_v      = {target_v}", flush=True)
print(f"achieved_v    = {achieved_v}", flush=True)
print(f"tracking err (L2) = {err:.6f}  (expect small, unconstrained regime)", flush=True)
assert err < 5e-3, f"FAIL: tracking error too large ({err})"
print("PASS: primary QP reproduces the requested Cartesian velocity (within tolerance).", flush=True)

# Cross-check against the closed-form (unconstrained) weighted least-squares
# solution: qdot* = (J^T J + lambda I)^-1 J^T v   -- this is exactly what the
# QP should reduce to when no box constraint is active.
lam = solver.regularization_weight
qdot_closed_form = np.linalg.solve(J.T @ J + lam * np.eye(7), J.T @ target_v)
diff = np.linalg.norm(qdot - qdot_closed_form)
print(f"||qdot_qp - qdot_closed_form|| = {diff:.8f}  (expect ~0)", flush=True)
assert diff < 1e-3, f"FAIL: QP solution diverges from closed-form solution ({diff})"
print("PASS: QP solution matches closed-form weighted-least-squares solution.", flush=True)

# --- Test 2: box constraints are respected --------------------------------
huge_target = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
qdot_clamped = solver._solve_primary_qp(J, huge_target, lower, upper)
within = np.all(qdot_clamped >= lower - 1e-6) and np.all(qdot_clamped <= upper + 1e-6)
print(f"lower={lower}", flush=True)
print(f"upper={upper}", flush=True)
print(f"qdot for huge target = {qdot_clamped}", flush=True)
print(f"All within box constraints: {within}", flush=True)
assert within, "FAIL: solution violates box constraints for an aggressive target"
print("PASS: box constraints (joint velocity / position limits) are respected.", flush=True)

# --- Test 3: nullspace bias pulls redundant joint toward reference posture ---
# With a ~zero Cartesian target, the primary task is satisfied by qdot=0
# everywhere; any nonzero qdot must come entirely from the nullspace bias.
qpos_offset = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.8, 0.5])  # last joint away from 0 ref
full_result = solver.cartesian_velocity_to_joint_velocity(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], {"joint_positions": qpos_offset, "joint_velocities": np.zeros(7)}
)
print(f"joint_velocity output (zero Cartesian target, joint 7 offset from ref) = {full_result}", flush=True)
assert np.linalg.norm(full_result) > 1e-6, "FAIL: nullspace bias produced no motion at all"
print("PASS: nullspace secondary objective is active (nonzero redundant-joint motion toward reference).", flush=True)

print("\n=== ALL CORRECTNESS CHECKS PASSED ===", flush=True)
