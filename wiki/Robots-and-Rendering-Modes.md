# Robots and Rendering Modes

REALM defaults to the DROID arm-mounted Panda configuration used in the paper, but the
benchmark code and configs support a small zoo of arms. Rendering quality is a separate
axis controlled at runtime via `--rendering_mode`.

## Robots (`--robot`)

Robot YAMLs live under `realm/config/robots/`. The active config is selected by the
`--robot` flag and loaded into the scene by `RealmEnvironmentDynamic`.

| Value | YAML | DOF | Step / render Hz | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `DROID` (default) | `DROID.yaml` | 7 + gripper | 15 | Franka Panda with the DROID base + wrist camera mount; 720×1280 sensors. |
| `DROID_default_pd_control` | `DROID_default_pd_control.yaml` | 7 | 15 | PD control variant used in some checkpoints. |
| `DROID_ee_control` | `DROID_ee_control.yaml` | 6 (EE) | 15 | End-effector pose control. Sets `ee_control=True` in the env, which routes the policy output through an IK solver. |
| `DROID_ee_delta_control` | `DROID_ee_delta_control.yaml` | 6 (EE Δ) | 15 | EE delta variant. |
| `DROID_no_wrist_cam` | `DROID_no_wrist_cam.yaml` | 7 | 15 | Same arm, no wrist camera. Useful for ablations of policies that consume wrist video. |
| `DROID_polaris_control` | `DROID_polaris_control.yaml` | 7 | 15 | Project-specific control variant. |
| `WidowX` | `WidowX.yaml` | 6 | **5** | Trossen WidowX. Note the lower step rate (`gm.DEFAULT_SIM_STEP_FREQ = 5`). |
| `UR5` | `UR5.yaml` | 6 | **30** | Universal Robots UR5. Step rate doubled vs. DROID. |
| `UR5_default_pd_control` | `UR5_default_pd_control.yaml` | 6 | 30 | UR5 PD control variant. |
| `UR5_aligned_pd_control` | `UR5_aligned_pd_control.yaml` | 6 | 30 | UR5 with the wrist aligned to match DROID conventions. |

Frequencies are set in `realm/eval.py::set_sim_config`. Physics always runs at 120 Hz.
`gm.ENABLE_HQ_RENDERING` is `False` for `--rendering_mode r` and `True` otherwise; this
is an OmniGibson-wide global so it affects all sensors.

## Picking a robot for evaluation

- The DROID checkpoints (Pi0, Pi0-FAST, GR00T) in the paper all assume `--robot DROID`
  with joint-position control. Don't change it unless you've also retrained the policy
  on the target embodiment.
- For evaluating in-house EE-control models, use `--robot DROID_ee_control`. The env
  branches in `RealmEnvironmentDynamic` switch from joint to EE control automatically
  based on the YAML's `ee_control` field.
- The `arm_mounted` vs. `arm` import branch is keyed off the task config path —
  `REALM_DROID10/...` always uses the arm-mounted DROID base; everything else uses the
  table-mounted DROID arm.

## Rendering modes (`--rendering_mode`)

Set at construction time in `RealmEnvironmentDynamic.set_rendering_mode`. The flag also
toggles `gm.ENABLE_HQ_RENDERING` in `set_sim_config`.

| Mode | Renderer | When to use |
| :--- | :--- | :--- |
| `pt` | RTX path tracing, 8 SPP, Optix denoiser | Final qualitative figures; not for sweeps. ~3–4× slower than `rt`. |
| `rt` (default) | RTX ray-traced lighting, HQ on | Standard benchmark setting. Matches the paper. |
| `r` | Rasterized "performance" path (no shadows, AO, indirect diffuse) | Ablations / debugging, headless smoke tests. ~2× faster than `rt`, expect visible artifacts (especially on glossy surfaces). |

Modes are set on `lazy.carb.settings`; switching mid-run is not supported — you have to
recreate the env.

### `pt` specifics

`pt` enables path tracing with these toggles:

- `/rtx/rendermode = PathTracing`
- `/rtx/pathtracing/spp = 8` (samples per pixel)
- `/rtx/pathtracing/totalSpp = 8`
- `/rtx/pathtracing/useDirectLightingCache = False`
- Optix denoiser on

Bumping SPP improves quality but quickly turns each frame into a full second.

### `r` specifics

`r` aggressively disables effects to push frame rate:

- `RaytracedLighting` mode but with shadows / reflections / indirect diffuse OFF
- DLSS off, AO off, DLSSG off
- Texture-streaming budget at 60 % of GPU memory
- `gm.ENABLE_HQ_RENDERING = False`

Side-effects to watch for: glass / transparent objects look wrong (wineglass, water
glass distractors), spec highlights are flat. Don't compare success rates across modes.

## Multi-view (`--multi-view`)

Adds a second external camera (`external_sensor1`). `cam2` extrinsics are read from
`config/env/external_sensors/camera_extrinsics.yaml` keyed by the per-task `camera_extrinsics.cam2`.
Required for:

- `--model_type dreamzero` (asserted in the inference client)
- The drawer tasks when paired with policies that condition on a second view

Multi-view increases per-step rendering cost ~80 %. Skip it unless your model uses it.

## `--no_render`

Disables both external camera rendering and the wrist camera. `extract_from_obs`
returns zero-filled 128×128 placeholders. Useful pairings:

- `--model_type debug --no_render --rendering_mode r` — pure physics throughput.
- `--no_render --no_record` — measures policy server latency without sim render
  contention.

Never combine `--no_render` with `--multi-view` (asserted in the env constructor — it's
treated as a configuration error).
