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
- `realm/config/robots/DROID.yaml` — `type: DROID` → `model: franka` with
  `end_effector: robotiq`, i.e. the stock `franka_robotiq` shipped with
  BEHAVIOR-1K. The DROID robot class no longer exists in the current
  registry (robots are now purely data-driven by YAML model name). See
  section 3 below for why REALM's own `droid.usd` cannot be loaded at all
  on a current engine, and section 4 for the failure mode that picking the
  wrong stock model produces. **This is a real methodological
  substitution, not a transparent shim**: the published `pi0_fast_droid`
  checkpoint was trained on DROID's specific gripper/arm geometry, and
  while `franka_robotiq` is the same physical platform (Franka Panda +
  Robotiq 2F-85), its wrist camera sits on `panda_link7` as `camera_link`
  rather than on `panda_link8` as `gripper_link_camera`, and its gripper
  exposes 2 controlled joints instead of 4. Absolute success rates are
  therefore not directly comparable to the paper's table. If you have
  access to a Linux cluster and can run REALM's official Docker image
  against the pinned 1.1.1 version, you keep the real DROID robot and
  don't need this substitution.
- `realm/environments/env_dynamic.py` — the robot base is raised by
  `DROID_BASE_HEIGHT` (0.86244 m). REALM's `REALM_DROID10` tasks load
  `droid_mounted.usd`, a DROID that ships its own pedestal lifting the arm
  to worktop height, so the poses in `scenes.yaml` are given with `z = 0`
  (pedestal foot on the floor). `franka_robotiq` is the arm alone, so
  without this offset it stands on the floor while the external cameras
  (which already add the same offset) and the world↔robot transforms keep
  pointing at worktop height.
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

### 3. Why REALM's own `droid.usd` cannot be loaded on a current engine

This one is not Windows-specific and will affect anyone moving REALM to a
newer OmniGibson.

`realm/robots/panda_robotiq/droid.usd` ships in this repository and does
contain the real thing: the full Robotiq 2F-85 and the wrist camera at
`/panda/gripper_link_camera/Camera`. It is unreachable for two separate
reasons. The path in `droid_arm.py` is hardcoded for the Docker image
(`os.path.join(gm.ASSET_PATH, "/app/realm/robots/...")`, where the leading
slash discards `gm.ASSET_PATH` entirely), and the class registry it was
loaded through no longer exists.

Registration itself turns out to be easy on a current engine — robots are
discovered by globbing `{gm.DATA_PATH}/*/models/<name>/<name>.yaml`, so no
Python class is needed at all, just a YAML whose values are all already
present in `droid_arm.py`. The blocker is the asset:
`prims/entity_prim.py::_compute_articulation_tree` requires the
articulation graph to be a strict tree (in-degree 1 for every non-root
link), and the Robotiq 2F-85 here is modelled as a genuine parallel
linkage with closed loops — `inner_finger` is driven both by the base
prismatic joint and from `outer_finger`. OmniGibson 1.1.1 had no such
check.

Two further defects in the same asset look like export slips, and
`realm/robots/panda_robotiq/fix_droid_articulation.py` in this fork
repairs both: `left_outer_finger_knuckle_joint` has `body0`/`body1`
reversed relative to its mirrored right-hand counterpart, and
`right_outer_finger_knuckle_joint` targets mesh prims
(`Defeatured_2F_85_..._finger2step`) instead of links. Because of them
neither `outer_finger` is any joint's child, so
`_preapply_articulation_root` finds three root candidates rather than one
and fails even earlier. Separately, the asset references six Robotiq part
files over **http** although all six sit next to it on disk;
`relocalize_droid_refs.py` rewrites those references to the local copies.

Repairing the topology gets the asset down to a single root but still not
to a tree, because the closed loop is deliberate geometry, not a mistake.
The remaining route is the one REALM's own authors took:
`realm/misc/modified_entity_prim.py` is a verbatim copy of OmniGibson's
`entity_prim.py` with those assertions stripped ("modified to remove
specific assertions on robot kinematic trees that made our USD file
incompatible"). It was taken against 1.1.1 and is not wired into anything
in the repo. Carrying that patch forward to a current `entity_prim.py` is
plausible — the file has only drifted ~80 lines across two major versions
— but it means vendoring a 1.6k-line internal engine file and re-applying
it on every upgrade. We chose the stock `franka_robotiq` instead. Either
way, "just point at `droid.usd`" is not among the options.

### 4. The silent black wrist camera

Worth guarding against regardless of platform, because it cost us eight
rollouts and looked like a genuine model result.

`realm/inference/utils.py` hardcoded the wrist camera as
`DROID:gripper_link_camera:Camera:0` and, when that key was absent, fell
back to a black 128×128 image with no warning. That is correct for
`no_render`, where the robot carries no cameras at all. It is dangerous
for a *misconfigured* robot: an earlier revision of this port used the
stock `franka` model, which has no wrist camera, so π0-FAST was fed black
for all 500 steps of every rollout while the benchmark reported a clean
`binary_SR = 0.00` — with smoothness, collision and path-length metrics
all still looking plausible. Nothing in the logs said why. We only caught
it by unpacking the recorded video parquet and noticing that the wrist
half of every frame had a maximum pixel value of 0.

This fork resolves the wrist camera from whatever the robot actually
exposes (`gripper_link_camera` on REALM's DROID asset, `camera_link` on
`franka_robotiq`), warns when it has to guess, and keeps the black
fallback only for the case where the robot exposes no cameras whatsoever.

The same function also normalised the gripper state as `proprio[7] / 0.05`
— the finger travel of REALM's own DROID gripper, in metres. On
`franka_robotiq`, `proprio[7]` is a knuckle *angle* in radians, so the
policy was told the gripper was nearly closed while it was wide open.
Index and range are now resolved from the loaded robot in `realm/eval.py`
and passed in, with the defaults preserving upstream behaviour for
REALM's own DROID.

### 5. Sharing one GPU between the simulator and the policy server

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

Both cameras now return real images (the wrist view measured at
`max = 239`, `mean = 97.4` after the fixes in section 4; it was uniformly
0 before). Robot base pose relative to the scene is still being aligned,
so no success-rate numbers from this fork should be quoted yet — any
rollouts recorded before the section-4 fixes are invalid by construction,
since the policy never saw its wrist view.

## Attribution

Produced jointly with Yahor Pachkouski (pachkyah@fel.cvut.cz), as part of
his bachelor's thesis "Simulated VLA Model Evaluation for Robotic
Manipulation" at CTU FEL (supervisor: Ing. Vladimír Petrík, Ph.D., CIIRC).
