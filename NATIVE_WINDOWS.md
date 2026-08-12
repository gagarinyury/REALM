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

- **Scheduled tasks must not go through a `.cmd` wrapper.** A task whose action is a
  batch file runs `cmd.exe` in the interactive session and draws a console window on
  the logged-in user's desktop — unacceptable if someone is working at the machine.
  Nested quoting makes `schtasks /Create` awkward with a direct `wsl.exe -- bash -c
  "..."` action, but the fix is a VBScript wrapper rather than a batch file:

  ```vbs
  CreateObject("Wscript.Shell").Run "wsl.exe -- bash -c ""bash /mnt/c/.../serve_policy.sh > /mnt/c/.../serve_policy.log 2>&1""", 0, False
  ```

  launched as `wscript.exe //B //Nologo <script>.vbs`. The `0` is the window style —
  hidden at launch, rather than suppressed after the fact. Verify with
  `Get-Process | Where-Object { $_.MainWindowTitle -ne '' }`, which must come back
  empty. Also remember that `schtasks /Create /SC ONCE /ST <time>` arms a real trigger:
  the task will fire on its own at that time unless you `schtasks /Change /DISABLE` it.

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

### 7. The `*_droid_jointpos` checkpoints predict deltas, not positions

The names suggest otherwise, and getting this wrong costs a day. Both
`pi0_fast_droid_jointpos` and `pi05_droid_jointpos` are trained to predict
**delta** joint positions. The conversion back to absolute targets is not
part of the checkpoint — it lives in the training config, and only in the
one that declares `action_space=JOINT_POSITION`:

```python
# openpi/src/openpi/training/config.py
if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
    # Data loader returns absolute joint position actions -- convert to delta actions for training.
    delta_action_mask = _transforms.make_bool_mask(7, -1)
    data_transforms = data_transforms.push(
        inputs=[_transforms.DeltaActions(delta_action_mask)],
        outputs=[_transforms.AbsoluteActions(delta_action_mask)],
    )
```

The mask is seven joints as deltas, gripper absolute. Serve such a
checkpoint under any other config — `pi0_fast_droid` is the tempting one,
since its `action_horizon` of 10 looks friendlier than the finetune
config's 16 — and `AbsoluteActions` is absent. The server then returns raw
increments of roughly 0.01 to 0.06 rad, REALM's joint controller applies
them as absolute targets (`droid_joint_controller.py`: `target_joint_pos =
command`), and the arm walks to the zero configuration, straight up, where
it stays for the rest of the episode.

Nothing reports an error: the vectors have the right shape, the robot
moves, metrics and video are written. The tell is in the logs —
`realm/eval.py` records `robot_state` into `logs/qpos`, so comparing it
against `logs/actions` is decisive. Correct pairing gives a mean
`|action - qpos|` around 0.01 rad; the broken one gives commands hovering
near zero regardless of where the arm currently is.

The obvious repair — reach for the config that does declare
`JOINT_POSITION`, which for π0-FAST is `pi0_fast_full_droid_finetune` —
substitutes one silent failure for a worse one. That config also declares
`action_horizon=16` and `max_token_len=180`, and under those the checkpoint
decodes nothing at all: `FASTTokenizer.extract_actions` finds no `Action: `
marker in the generated text and returns `np.zeros`, silently
(`openpi/models/tokenizer.py`). `AbsoluteActions` then adds the current
state to that zero, so the reply looks like "almost where you already are"
and the arm drifts along plausibly while the policy is, in fact, mute.

The pairing that works is `pi0_fast_droid_jointpos_polaris` — horizon 10,
`max_token_len` 180, `action_space=JOINT_POSITION`:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_droid_jointpos_polaris \
    --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid_jointpos
```

The config lives in `openpi/training/misc/polaris_config.py` and is
registered in the main config list. It names a different checkpoint of its
own, but that does not matter here: with `--policy.dir` given, norm stats
are deliberately loaded from the checkpoint rather than from the config's
assets directory (`policies/policy_config.py`, with a comment to that
effect), so the denormalisation is the one belonging to our weights.

**Check for the mute-policy failure before trusting any rollout.** It takes
a minute and needs no simulator: send the server one observation and
compare the reply against the denormalisation of a normalised zero,
computed from the checkpoint's own `norm_stats.json`. Two things give the
failure away — every channel matches that zero, and every row of the action
chunk is identical. A working policy answers differently to different
inputs; the mute one returns the same vector for a real observation and for
uniform noise.

### 8. HQ rendering and the DROID rate are mutually exclusive when headless

Three constraints meet on OmniGibson 3.9.1 and cannot all hold:

1. headless requires `rendering_dt == sim_step_dt` (`simulator.py`,
   `_validate_dts`);
2. `gm.ENABLE_HQ_RENDERING` asserts a rendering frequency of at least
   60 FPS, inside the isosurface block of `_set_renderer_settings`;
3. the DROID checkpoints run at 15 Hz, which `realm/eval.py::set_sim_config`
   sets as `DEFAULT_SIM_STEP_FREQ` and `DEFAULT_RENDERING_FREQ`.

REALM enables HQ rendering for every mode except `r`, so its own default
`rt` configuration does not start on this engine version. It did on 1.1.1:
compare `simulator.py` at tag `v1.1.1` of the (then separate)
`StanfordVL/OmniGibson` repository, where the flag toggles RTX settings
only and no frame-rate assert exists. The headless assert was already
there, and was satisfied, because both frequencies were 15.

Raising the rendering frequency to 60 is the obvious escape and the wrong
one — headless drags the action and control rates up with it, and the
policy then runs four times faster than it was trained for. What the flag
still guards on 3.9.1 is narrow: DLSS "Realism" versus "Performance",
`updateVelocitiesToUsd`, and the isosurface path itself. Reflections,
indirect diffuse and ambient occlusion are now enabled unconditionally. So
turn the flag off and restore the one setting that matters for a scene
without particle systems — which is what `set_rendering_mode` now does for
`rt`. No REALM_DROID10 task contains fluids or particles, so the isosurface
path is dead weight here regardless.

### 9. The repaired gripper does not close -- a 0.2 velocity clamp (solved)

Section 3 restores REALM's own `droid.usd` by breaking the parallelogram loop
and re-coupling the gripper with `PhysxMimicJointAPI`, the way the stock
Robotiq assets in BEHAVIOR-1K do it. The asset loads, the arm works, the
wrist camera works — and the fingers never close. Measured by driving the
gripper from a script, with no policy in the loop:

```
command CLOSE:  outer knuckles 0 -> 0.7854 rad   inner knuckles 0.005 -> 0.019
finger pads:    7.94 cm apart open  ->  7.98 cm apart "closed"
a cube suspended exactly between the pads: no finger contact for 40 steps
```

The driven joints travel their whole range, the mimicked ones do not follow,
and so nothing ever grips. Every rollout therefore stalls at the GRASP stage
regardless of the policy, which is exactly what both π0-FAST and π0.5 do.

Mimic joints themselves are fine in this build: the stock `ur5e`, carrying
the same 2F-85 and the same scheme, closes correctly under an identical probe
(followers reach 0.78 rad). Compared line by line against it — applied API
schemas, joint axis, limits, body pairs, drive placement — our asset matches,
with one real discrepancy since fixed: the sides are mirrored, and the stock
`right_inner_knuckle_joint` carries gearing `+1` with limits `[-45, 0]` where
we had copied the left side's `-1` and `[0, 45]`. Correcting it alone did not
change the outcome.

**Root cause, found 10.08.2026.** `droid.usd` sets `physxJoint:maxJointVelocity = 0.2` on the
trailing gripper joints, against `120` on the leading ones and `inf` on the same joints of the
stock Robotiq. The joints were never stuck and never unpowered -- they moved at 0.2 deg/s.
Forty steps at 15 Hz is 2.7 s, so 0.2 * 2.7 = 0.0094 rad against the 0.0112 rad measured; a full
45 deg stroke would take four minutes. The clamp is harmless as shipped, where those joints are
carried by the parallelogram's loop constraints rather than driven; breaking the loop (section 3)
made them driven and turned it into a handbrake.

Lift it -- `realm/robots/panda_robotiq/lift_gripper_velocity_clamp.py` -- and the coupling from
section 3 works at once: leading knuckles reach 0.7854 rad, trailing ones follow to 0.7158 and
0.7854. That repair was correct all along.

Worth knowing when hunting something similar: a clamped joint and an uncoupled joint look
identical from outside, so six coupling hypotheses were tried and discarded before this surfaced.
It showed up only by exporting the binary asset to text (`Usd.Stage.Open(...).Export(...)` under
the Isaac interpreter) and diffing a working joint against a stuck one, where the whole
difference is one line.

Two earlier observations, kept because they cost time and may bite again:

- Driving the inner knuckles directly (restore `DriveAPI`, list them in
  `finger_joint_names` so `gripper_control_idx` picks them up) does move them
  — but only once the mimic API is removed from those joints. A driven joint
  that also carries a mimic constraint stays put: the constraint wins. With
  mimic dropped the joints travel about a third of their range, in the
  direction opposite to their own limits, and scene loading then hangs.
- The outer branch differs structurally: `outer_knuckle -> outer_finger` is a
  `RevoluteJoint` with gearing `0.01` in the stock asset and a `FixedJoint` in
  ours. A welded outer branch may be blocking the pads mechanically.

Useful when investigating this: the asset is binary, so every question costs a
simulator start, while the reference (`models/ur5e/usd/ur5e.usda`) is text.
`Usd.Stage.Open(...).Export("droid_flat.usda")` under the Isaac interpreter
turns ours into text once and makes the comparison a grep.

### 10. Reproducing this fork on a clean machine: four things that are not in it

Found on 10.08.2026 by deploying the fork onto a rented machine (RunPod, RTX A6000,
image `stanfordvl/behavior:3.9.1`, Isaac Sim 5.1) — i.e. by doing what a reader of this
repository would do. None of the four failures below reproduce on the development machine,
because its state accumulated by hand over a week.

**a. A patched engine file that lived only on the dev machine.**
`Robot._generate_controller_config` looks the controller name up in
`self._default_controller_config[group]`, and that dict is assembled *exclusively* from the
engine's own built-in sets (`_default_arm_joint_controller_configs`, `gripper_pj_configs`, ...)
keyed by their own `["name"]`. It has nothing to do with `REGISTERED_CONTROLLERS`: registering
a class there lets the engine *construct* it, but does not make it *selectable by name*. So
REALM's `CustomJointController` / `CustomGripperController` raise `KeyError`. The exception is
thrown inside an Isaac Sim C++ callback, so the process dies as
`Segmentation fault (core dumped)` with no traceback at all.
Now shipped as `realm/misc/robot_controller_name_fallback_og391.patch`.

**b. A config that contradicted the repair scripts.**
`untangle_droid_gripper.py` removes the drive from the four follower joints and slaves them
with `PhysxMimicJointAPI`, exactly as every stock Robotiq in BEHAVIOR-1K does. But
`finger_joint_names` still listed those four joints as controlled — a leftover from a
hypothesis abandoned on 09.08. The engine refuses:
`AssertionError: Controllers should only control driveable joints!` (again surfacing as a
segfault). Fixed: the list is back to the two driven `outer_knuckle` joints.

**c. The gripper closing direction.**
See section 11 below — this one changes results rather than preventing startup.

**d. Environment facts that only a failed run reveals.**
The dependency set from `.docker/realm_og391.Dockerfile` with the pins from
`og391-constraints.txt` is mandatory (`numpy==1.26.0`, `torch==2.7.0+cu128`) — without it,
`ModuleNotFoundError: mujoco`. Stock robot assets must carry a `VERSION` file >= 3.8.2, and
placing our `models/droid` there *before* the download makes the downloader silently skip.
The dataset archive unpacks *without* a top-level directory while the engine expects
`{DATA_PATH}/behavior-1k-assets/`. Finally: 141485 small files on a network filesystem is a
trap — unpacking ran at ~31 files/s and then hung in `request_wait_answer` (FUSE); on a local
disk of the same machine, 150-320 files/s.

### 11. The gripper command was inverted (found 10.08.2026)

Symptom, spotted by watching a rollout video: the jaws never squeeze, and the robot nudges the
cube sideways with the gripper body instead of grasping it.

Measured, not inferred — from the rollout logs of this fork against those of the stock stack:

| | stock 1.1.1 (task solved, 1.0) | this fork on 3.9.1 |
|---|---|---|
| distinct gripper commands over 800 steps | 8, range 0.000-0.653 | **1**, constant -0.006 |
| driven joint travel | moves | **0.0000, std 0.0000 — never moves** |

The chain: `robot.py:3450` builds `"inverted": self._grasping_direction == "upper"`; the
parameter is unset, so the default `"lower"` applies; `MultiFingerGripperController` in
`binary` mode therefore sends the joints to their **lower limit** on a close command. The
driven `outer_knuckle` joints are limited to **0..45 deg** (read straight out of `droid.usd`),
and zero is the *open* pose — as this repository's own
`droid_robot_definition.yaml` states: "the gripper opens at zero". So "close" drives the jaws
fully open, the joint parks at zero and stops moving, the `gripper_position` fed back to the
policy stops changing, and the policy repeats one command for the rest of the episode.

This was correct in the original: `droid_arm.py:146` sets `grasping_direction="lower"` with the
comment "gripper grasps in the opposite direction", and for *prismatic* joints
(`_gripper_control_idx = th.arange(7, 11)`, travel 0..0.05 m) zero really is the closed pose.
`untangle_droid_gripper.py` deletes those prismatic joints — they are what closes the
parallelogram — and moves control onto revolute knuckles, where the meaning of "lower limit"
is the opposite. The parameter stayed; the mechanism changed.

The engine's own documentation states the default plainly
(`docs/omnigibson/controllers.md`): *"By default, <closed, open> is assumed to correspond to
<q_lower_limit, q_upper_limit> for each joint"*. `grasping_direction` itself appears nowhere in
the documentation — only at `robot.py:200`.

Fix, one line in `realm/config/robots/DROID.yaml`: `grasping_direction: "upper"`. Note it does
**not** belong in the robot definition YAML — `definition_schema.py` has no such field; it
works because extra keys in the robot config are passed straight to the constructor
(`docs/omnigibson/robots.md`). An equivalent alternative is `closed_qpos`/`open_qpos` in
`controller_config.gripper_0`.

`tests/gripper_bench.py` checks this in ~3 minutes without a policy server: it drives the
gripper open, then closed, and prints every joint's travel plus the gap between the finger
links. Any question about the gripper should go through it rather than through a 20-minute
rollout — that cost is why six wrong hypotheses were tried before this one.

### 12. The grasp clause was dead code, and upstream chose to keep it that way

Found here on 10.08.2026 (`3257bab`, 13:35 UTC) and, independently, upstream on 11.08.2026
(`dce5ae7` on `port-to-og391`, 16:03 UTC). Same defect, **different fixes** — anyone comparing
numbers across the two repositories needs this section.

`is_grasping` requires three conditions at once, and the third one read `proprio[7:9]` and
tested `0.45 - q > 1e-3`. On REALM's own `droid.usd` those two entries are the *prismatic*
finger joints with a 0..0.05 m stroke, so the test evaluates `0.45 - 0.05 > 0`: true for every
pose the gripper can physically reach. It has never rejected anything, on any run, in any
published REALM result — a grasp there is decided purely by the two contact conditions.
Upstream states the same conclusion in `dce5ae7` and adds a number from the other side: on a
*revolute* 2F-85 the identical constant rejected 78 of 78 steps in which both pads were on the
block and the block was lifted.

The constant only becomes visible once the parallelogram is untangled (section 3) and control
moves onto the revolute knuckles, stroke 0..0.7854 rad. Measured here with
`tests/gripper_bench.py` on 10.08.2026:

| | `proprio[7:9]` | upstream clause |
|---|---|---|
| jaws open | `[0.0000, 0.0000]` | True |
| jaws closed | `[0.7774, 0.5414]` | **False** |

So GRASP could never be awarded and `task_progression` froze at 0.2 (REACH only) no matter what
the gripper did — four rollouts with progressively better gripper physics all returned exactly
0.2, which is what led here.

**This fork restores the clause's meaning.** It asks "are the fingers driving towards closure",
and the honest answer for an arbitrary gripper compares against that gripper's own joint
limits, taken from the robot rather than from hard-coded indices — `eval.py:172` already
resolves them this way:

```python
closed_frac = (finger_joints - lower) / max(upper - lower, 1e-9)
is_either_finger_closing = bool((closed_frac > 0.1).any())
```

**Upstream deliberately preserves the tautology.** `dce5ae7` scales the threshold as
`open + 9.0 * (closed - open)`, which reproduces `0.45` exactly for `droid.usd` — so every
historical number stays bit-identical — and stays vacuous on the new robolab asset, "exactly as
it has always been on the stock asset". The commit notes that `0.45` is likely a typo for
`0.045`, which would give the guard real meaning, and leaves adopting that to a separate
decision because it would change SR.

Both choices are defensible: upstream protects the continuity of a published benchmark, this
fork wants the stage to mean what its name says. The consequence for anyone reading numbers
from both repositories is that **this fork's grasp criterion is strictly stricter**, so its
progression and success rates are not directly comparable to REALM's published figures, and
where they are compared the difference has to be stated. Concretely: the 0.61 π0-FAST baseline
was obtained with a grasp check that never rejected a single step.

`tests/gripper_bench.py` prints both the fork's live formula and the resulting flag, so the
bench cannot silently drift away from `env_base.py` again.

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

The robot base pose was measured against the scene and matches to the
millimetre, and the wrist field of view was corrected (section 3). The
exterior view matches the reference frames published in the REALM paper:
the arm is outside that camera's frustum in the Default setting, which is
how the benchmark is meant to look, not a misalignment on our side.

Rollouts recorded before the section-4 fixes are invalid by construction,
since the policy never saw its wrist view; those recorded before sections 7
and 8 are invalid too, since the policy either ran at four times its
training rate, or had its output interpreted in the wrong space, or was
returning zeros.

With all of them addressed, π0-FAST drives the arm for the first time on
this fork: on `put_green_block_into_bowl` (Default, 800 steps) it clears the
REACH stage at step 290 and brings the gripper onto the block, commanding a
close for 184 of the 800 steps. Progression stops at 0.2.

**It stops there for a reason on our side, not the model's — see section 9.
The gripper never actually closes.** π0.5 reaches exactly the same ceiling on
the same task, holding the close command for up to nine seconds at a time.
No number from this fork is a statement about either model yet, and none is
comparable to the 0.61 in REALM's README (which in any case averages tiered
progression over all ten tasks).

## Attribution

Produced jointly with Yahor Pachkouski (pachkyah@fel.cvut.cz), as part of
his bachelor's thesis "Simulated VLA Model Evaluation for Robotic
Manipulation" at CTU FEL (supervisor: Ing. Vladimír Petrík, Ph.D., CIIRC).
