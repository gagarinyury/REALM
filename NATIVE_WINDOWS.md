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
- `realm/config/robots/DROID.yaml` — the DROID robot class no longer exists
  in the current registry (robots are now purely data-driven by YAML model
  name). Two configurations are supported. `model: droid` uses REALM's own
  asset once it has been repaired as described in section 3, and is the
  closer reproduction. `model: franka` with `end_effector: robotiq` uses the
  stock `franka_robotiq` shipped with BEHAVIOR-1K, which needs no asset work;
  section 4 describes the failure mode that picking the wrong stock model
  produces. Picking the stock model **is a real methodological substitution,
  not a transparent shim**: the published `pi0_fast_droid` checkpoint was
  trained on DROID's specific gripper and arm geometry, and while
  `franka_robotiq` is the same physical platform (Franka Panda + Robotiq
  2F-85), its wrist camera sits on `panda_link7` as `camera_link` rather than
  on `panda_link8` as `gripper_link_camera`, and it exposes a different
  gripper linkage. Repairing the original asset removes that caveat, which is
  why section 3 exists.
- `realm/environments/env_dynamic.py` — the robot base is raised by
  `DROID_BASE_HEIGHT` (0.86244 m), but only for a robot that does not carry
  its own pedestal. REALM's `REALM_DROID10` tasks assume a DROID whose root
  link is the floor beneath its pedestal, so the poses in `scenes.yaml` are
  given with `z = 0`
  (pedestal foot on the floor). The repaired `droid.usd` satisfies that by
  construction — its `base_link` sits 0.8645 m below `panda_link0`, matching
  the hardcoded constant to within 2 mm. `franka_robotiq` is the arm alone,
  so without the explicit offset it stands on the floor while the external
  cameras (which already add the same offset) and the world↔robot transforms
  keep pointing at worktop height.
- `realm/robots/droid_joint_controller.py` — the end-effector link name is
  resolved from the links the loaded robot actually exposes (`panda_link8` on
  `droid.usd`, `eef_link` on stock `franka_robotiq`) rather than hardcoded.
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

### 3. Restoring REALM's own `droid.usd` on a current engine

Not Windows-specific: this will affect anyone moving REALM to a newer
OmniGibson.

`realm/robots/panda_robotiq/droid.usd` ships in this repository and contains
the real thing — the full Robotiq 2F-85 and the wrist camera at
`/panda/gripper_link_camera/Camera` on `panda_link8`. Two things make it
unreachable out of the box. The path in `droid_arm.py` is hardcoded for the
Docker image (`os.path.join(gm.ASSET_PATH, "/app/realm/robots/...")`, where
the leading slash discards `gm.ASSET_PATH` entirely), and the class registry
it was loaded through no longer exists.

Registration is the easy half: robots are now discovered by globbing
`{gm.DATA_PATH}/*/models/<name>/<name>.yaml`, so no Python class is needed,
just a YAML whose values all already exist in `droid_arm.py`. See
`realm/robots/panda_robotiq/droid_robot_definition.yaml`.

The real blocker is the asset. `prims/entity_prim.py` requires the
articulation graph to be a strict tree, and the Robotiq 2F-85 here is a
genuine parallel linkage: `inner_finger` is driven both by a prismatic joint
from the gripper base and from `outer_finger`. Five links end up with an
in-degree of two. OmniGibson 1.1.1 had no such check.

Upstream's own workaround is `realm/misc/modified_entity_prim.py` — a
verbatim copy of the engine's `entity_prim.py` with the offending assertions
commented out ("modified to remove specific assertions on robot kinematic
trees that made our USD file incompatible"). Diffing it against the original
from `v1.1.1` shows the patch to be exactly three suppressed assertions, and
all three are present unchanged in 3.9.1 — so the approach still works, at
the price of vendoring a 1.6k-line internal engine file forever.

**This fork repairs the asset instead, and modifies no engine file.** The
precedent is BEHAVIOR-1K's own robots: stock `ur5e` carries the same Robotiq
2F-85 and loads without complaint, because Stanford never model the closing
link of the parallelogram. Their gripper is a plain tree and the coordinated
jaw motion is reproduced by PhysX mimic joints, each slaved to the driven
`outer_knuckle` with a gearing of -1. Three idempotent scripts bring
`droid.usd` to the same form; run them in this order, each asserts its own
postcondition:

```
python realm/robots/panda_robotiq/relocalize_droid_refs.py  <droid.usd>
python realm/robots/panda_robotiq/untangle_droid_gripper.py <droid.usd>
python realm/robots/panda_robotiq/align_droid_root.py       <droid.usd>
```

1. **`relocalize_droid_refs.py`** — the asset references six Robotiq part
   files over http (`omniverse-content-production` S3) although all six sit
   next to it on disk under the same names; upstream downloaded them but
   never rewrote the references, which resolved through the Omniverse cache
   inside their Docker image. Rewritten to local relative paths.

2. **`untangle_droid_gripper.py`** — brings the gripper to the stock
   topology. Reverses three joints recorded `finger -> knuckle`; re-binds
   `right_outer_finger_knuckle_joint` from mesh prims
   (`Defeatured_2F_85_..._finger2step`) onto links, a mirrored export slip,
   recomputing its relative pose from the asset's world transforms; removes
   the four joints forming the second path to each finger; clears
   `excludeFromArticulation`, which upstream set on the entire gripper
   precisely because PhysX cannot hold closed loops inside an articulation;
   and gives the four now-undriven joints mimic relationships to their driven
   knuckle, with gearing and limits copied from stock `ur5e`, removing their
   `PhysicsDriveAPI`. Result: 24 joints to 20, five loop closures to none,
   three root candidates to one, nine actuated joints exactly as in the stock
   model.

3. **`align_droid_root.py`** — the asset's origin sits at worktop height
   while its root link `base_link` represents the floor under the robot's
   pedestal, 0.85 m below, and the engine asserts that entity prim and root
   link coincide. Moves the frame of reference onto `base_link`. This is a
   reparameterisation, not a displacement, and the script verifies it: every
   link's world pose is unchanged to numerical zero.

Two consequences worth knowing. `base_link` sits 0.8645 m below
`panda_link0` — against `DROID_BASE_HEIGHT = 0.86244` hardcoded in
`env_dynamic.py`, a 2 mm match, confirming REALM's scene poses and camera
offsets were written for exactly this asset. And the end-effector link is
`panda_link8` here but `eef_link` on stock `franka_robotiq`, which
`droid_joint_controller.py` now resolves from the loaded robot rather than
assuming.

The cost: the jaws are no longer coupled by a mechanism but by mimic
constraints, so their parallelism is enforced by the solver rather than by
geometry. That is the same assumption every stock Robotiq in BEHAVIOR-1K
operates under, but it is an assumption. What it buys is the wrist camera in
its original place on `panda_link8` — the viewpoint the `pi0_fast_droid`
checkpoint was trained against.

Verified on OmniGibson 3.9.1: environment created, 13 DOF, eef `panda_link8`,
wrist camera present as `DROID:gripper_link_camera:Camera:0` with a maximum
pixel value of 239, gripper range `[0, 45 deg]`.

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
- **The same teardown kills a server you did not start.** A Task
  Scheduler-launched policy server is destroyed by *any* subsequent
  `ssh <host> "wsl -- <command>"`, including a read-only one. Checking on it
  with `wsl -- bash -c 'tail /root/serve_policy.log'` is enough to kill it.
  The symptom is thoroughly misleading: the log simply stops mid-line, with
  no traceback, exactly as if the process had hung on that step. The only
  place the truth shows up is the task's exit code,
  `Last Result: -1073741510` (`STATUS_CONTROL_C_EXIT`), i.e. terminated from
  outside. Cost us four consecutive "crashes" that were nothing of the sort.
  While a server is running or loading, observe it **only from the Windows
  side** — `Test-NetConnection -Port 8000` for readiness, `nvidia-smi` for
  GPU memory, and `Get-Process vmmemWSL` for the VM's RAM footprint (which
  rises as the checkpoint is read). Enter WSL2 again only once the port
  answers, or after the server is deliberately stopped.

- **WSL2's default memory limit does not fit the larger checkpoints.** WSL2
  takes half the host's RAM by default — 15.5 GB on a 31 GB machine — and the
  `pi05_droid_jointpos` checkpoint is 12 GB on disk. Loading it exceeds the
  limit once Python and JAX are accounted for, and the OOM killer takes the
  process silently: the log simply stops after `Restoring checkpoint from
  ...`, with no traceback and no exit message, which looks exactly like the
  SSH teardown described above. `pi0_fast_droid_jointpos` (11 GB) fits, so
  the problem only appears when switching models. Fix in `%USERPROFILE%\.wslconfig`:

  ```ini
  [wsl2]
  vmIdleTimeout=-1
  memory=20GB
  swap=8GB
  ```

  followed by `wsl --shutdown`. The reservation is virtual — WSL2 commits
  pages as needed — so leaving 11 GB for Windows still accommodates Isaac Sim
  running natively alongside. The swap entry is insurance: paging is
  preferable to a process that dies without saying so.

### 6. A partially downloaded checkpoint reports success

Worth knowing because the failure surfaces far from its cause. `openpi`
caches checkpoints under `~/.cache/openpi/openpi-assets/checkpoints/`, and
`openpi.shared.download.maybe_download` treats an existing directory as a
completed download. Interrupt a first download — a timeout, a killed SSH
session, anything — and every later call returns the truncated cache as
`OK`. The model then fails much later, while reading weights:

```
ValueError: OUT_OF_RANGE: Error reading "params.PaliGemma.llm.embedder.input_embedding/0.0"
  ... Requested byte range [0, 1957696997) is not valid for value of size 1330626560
```

Recovery is to delete the checkpoint directory (plus its `.partial` and
`.lock` siblings) and download again in one uninterrupted run, then verify
by size rather than by the tool's own report — comparing the largest files
against a known-good checkpoint works well, since the DROID checkpoints are
structurally identical and each is roughly 11 GB.

Note also that the download runs through `gcsfs` unless `gsutil` is present;
the `gsutil not found, falling back to gcsfs` warning is harmless and the
transfer is fast (~75 MB/s in our setup), so a stalled download is more
likely to be a killed process than a network problem.

## Status

Full pipeline verified working end-to-end natively on Windows (RTX 5080,
16GB VRAM): scene/robot loading, physics stepping, real π0-FAST inference
calls over the network, action execution, and metrics/video/trajectory
logging to disk — a complete rollout with no crashes.

REALM's own `droid.usd` loads and runs after the asset repair of section 3:
13 DOF, end-effector `panda_link8`, wrist camera present as
`DROID:gripper_link_camera:Camera:0`, gripper range `[0, 45 deg]`. Both
cameras return real images (wrist `max = 239`, `mean = 98.6`; it was
uniformly 0 before the section-4 fixes).

Robot base pose relative to the scene is still being aligned — the wrist view
currently points away from the work area — so no success-rate numbers from
this fork should be quoted yet. Any rollouts recorded before the section-4
fixes are invalid by construction, since the policy never saw its wrist view.

## Attribution

Produced jointly with Yahor Pachkouski (pachkyah@fel.cvut.cz), as part of
his bachelor's thesis "Simulated VLA Model Evaluation for Robotic
Manipulation" at CTU FEL (supervisor: Ing. Vladimír Petrík, Ph.D., CIIRC).
