# Running REALM natively on Windows (no Docker / no WSL2 for the simulator)

This is a documented, working port of REALM to run natively on a single
Windows machine with an NVIDIA GPU — no Docker, and critically, **the
simulator itself does not run under WSL2** (Isaac Sim is architecturally
incompatible with WSL2 rendering; see NVIDIA's own confirmation of this
limitation). This was produced while evaluating π0-FAST inside REALM as
part of a bachelor's thesis at CTU FEL, on a machine without access to a
Linux GPU cluster — a common situation for individual researchers and
students.

REALM was originally built and is officially distributed against
`stanfordvl/omnigibson:1.1.1` (Linux/Docker only). The current OmniGibson
release (3.9.1 at the time of this port) has diverged substantially from
that pinned version — the robot registry, the controller API, contact
handling, and USD-scene editing all changed. This document and the patches
in this branch/fork make REALM run against a **current** OmniGibson
install, natively on Windows.

If you're in the same situation — one Windows workstation with an NVIDIA
GPU, no cluster, need to run REALM — this should get you there. If you
have access to a real Linux GPU cluster, just use REALM's official Docker
setup instead; that remains the more faithful, better-supported path (it
also lets you keep the original DROID robot without the substitution
described below).

## What's different from upstream

### 1. Vulkan render crash on native Windows startup

`OmniGibson/omnigibson_5_1_0.kit` force-enables `vulkan = true`
unconditionally. Isaac Sim's Windows build defaults to Vulkan *off*
(comment in the same file: `"...on by default on Linux, off by default on
Windows"`), and forcing it on causes a startup crash. Fix: set
`vulkan = false` in your local copy of that `.kit` file. This is an
Isaac Sim/OmniGibson packaging issue, not something in this repo's own
code — you'll need to patch your own Isaac Sim install, not just this
fork.

### 2. OmniGibson 1.1.1 → current: API migration

REALM's robot/controller code was written against OmniGibson 1.1.1, before
the merge with BEHAVIOR-1K. Between that version and current OmniGibson,
the robot system moved from a Python-class-per-robot-type registry to a
purely data-driven YAML/USD model lookup, and the controller system was
rearchitected to a batched, multi-robot-capable design
(`ControllerView` + `ControllableObjectViewAPI`, replacing the old
`control_dict`-based single-instance API). This fork's patches cover:

- `realm/robots/droid_joint_controller.py`, `droid_gripper_controller.py`
  — rewritten against the batched controller API, verified against
  OmniGibson's own native `joint_controller.py` / `multi_finger_gripper_controller.py`.
- `realm/robots/droid_arm.py`, `ur.py`, `widowx.py`, `droid_arm_mounted.py`
  — `ManipulationRobot` → `Robot` (classes merged upstream).
- `realm/environments/env_base.py` — `ContactBodies` (removed upstream) →
  `RigidContactAPI.is_in_contact` / `get_contact_pairs`. Note: the newer
  contact API is a boolean contact matrix with no per-contact impulse
  magnitude, so REALM's original impulse-threshold filtering in
  `check_collisions()` cannot be reproduced exactly as written.
- `realm/environments/env_dynamic.py`, `perturbations/v_light.py` — wrapped
  direct USD prim edits in `og.sim.editing_usd()`, a guard added upstream
  for USD↔Fabric sync that didn't exist in 1.1.1.
- `realm/config/robots/DROID.yaml` — `type: DROID` → `type: franka`. The
  DROID robot class no longer exists in the current registry (robots are
  now purely data-driven by YAML model name). **This is a real
  methodological substitution, not a transparent shim**: the published
  `pi0_fast_droid` checkpoint was trained on DROID's specific
  gripper/arm geometry, and swapping in a stock Franka Panda will affect
  measured success rates. If you have access to a Linux cluster and can
  run REALM's official Docker image against the pinned 1.1.1 version,
  you keep the real DROID robot and don't need this substitution.
- `realm/robots/robot_ik/robot_ik_solver.py`, new `simple_arm.py` — REALM's
  IK solver depends on `dm_robotics.moma`/`dm_robotics.controllers`, which
  **have no Windows wheel at all** (Linux-only manylinux distribution,
  Bazel C++ build, no sdist). Reimplemented from scratch: the same
  hierarchical QP formulation (primary weighted Cartesian-velocity
  tracking + Tikhonov regularization + joint box constraints, secondary
  nullspace posture regularization), transcribed from DeepMind's own open
  C++ source (`cartesian_6d_to_joint_velocity_mapper.cc`,
  `cartesian_6d_velocity_task.cc`, `joint_position_limit_constraint.cc`),
  solved with `osqp` (has Windows wheels) instead of the proprietary ADMM
  solver. Correctness verified against the closed-form analytic
  weighted-least-squares solution in
  `realm/robots/robot_ik/test_ik_solver.py`.
- Compute backend: current OmniGibson supports a runtime-selectable
  numpy/torch backend (`gm.USE_NUMPY_CONTROLLER_BACKEND`, defaults to
  numpy). This port's controller rewrites assume torch throughout (matching
  the single-backend design REALM was originally written for) —
  `gm.USE_NUMPY_CONTROLLER_BACKEND = False` is set in `realm/eval.py`
  before simulator launch.
- Various path-separator fixes (`.split("/")` → normalized) for Windows
  paths in `env_dynamic.py` and `perturbations/_helpers.py`.

### 3. Sharing one GPU between the simulator and the policy server

If you only have one machine with one GPU (not a cluster), you'll likely
run the simulator natively on Windows and the model policy server (e.g.
openpi) inside WSL2, sharing one physical GPU. Two things worth knowing:

- **JAX memory fraction matters non-obviously.** With
  `XLA_PYTHON_CLIENT_MEM_FRACTION` too low, model loading itself fails
  (checkpoint doesn't fit). With it "reasonable" but still tight, loading
  succeeds but the *first real inference call* fails with
  `RESOURCE_EXHAUSTED` at a fixed, deterministic allocation size —
  independent of what the simulator is doing (we confirmed this: cutting
  the simulator's render resolution had zero effect on the failure). The
  fix was to *raise* the fraction (`0.4` → `0.6`), not lower it, and to
  **not** set `XLA_PYTHON_CLIENT_PREALLOCATE=false` — dynamic/incremental
  GPU allocation performed worse than one large static preallocation in
  this WSL2 GPU-passthrough setup, plausibly due to allocator fragmentation
  over the virtualized GPU-PV path.
- **Background WSL2 processes don't survive SSH session teardown the way
  you'd expect.** `nohup`, `setsid`, and even a detached
  `Start-Process -WindowStyle Hidden` on the Windows side all failed to
  keep a long-running WSL2 server alive once the originating SSH
  connection closed — this looks like Windows OpenSSH tying spawned
  processes to a Job Object that gets torn down with the session,
  regardless of POSIX-level detachment. Setting `vmIdleTimeout=-1` in
  `.wslconfig` keeps the WSL2 VM itself alive, but the *process* still
  needed a launch mechanism outside any SSH session entirely — the only
  approach that reliably worked was launching it via the Windows Task
  Scheduler (`schtasks /Create ... /SC ONCE` + `schtasks /Run`).

## Status

Full pipeline verified working end-to-end natively on Windows (RTX 5080,
16GB VRAM): scene/robot loading, physics stepping, real π0-FAST inference
calls over the network, action execution, and metrics/video/trajectory
logging to disk — a complete rollout with no crashes.

## Attribution

Produced jointly with Yahor Pachkouski (pachkyah@fel.cvut.cz), as part of
his bachelor's thesis "Simulated VLA Model Evaluation for Robotic
Manipulation" at CTU FEL (supervisor: Ing. Vladimír Petrík, Ph.D., CIIRC).
