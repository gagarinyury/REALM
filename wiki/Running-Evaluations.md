# Running Evaluations

The benchmark entry point is `examples/02_evaluate.py`. It is a thin argparse wrapper
around `realm.eval.evaluate`, which builds a `RealmEnvironmentDynamic`, connects to a
running policy server, and rolls out the requested task × perturbation combination
`--repeats` times.

## Minimal example

```bash
python /app/examples/02_evaluate.py \
    --task_id 0 \
    --perturbation_id 0 \
    --repeats 25 \
    --max_steps 800 \
    --model_name pi05 \
    --model_type openpi \
    --port 8000 \
    --experiment_name first_full_eval
```

`--model_name` is a free-form label used in the log path. `--model_type` selects the
inference adapter and **must** be one of the supported values
(`openpi`, `GR00T`, `GR00T_N16`, `molmoact`, `hamster`, `dreamzero`, `debug`); see
[Inference Servers](Inference-Servers).

## Every flag, in one table

| Flag | Required | Default | Notes |
| :--- | :--- | :--- | :--- |
| `--task_id` | no | `0` | Index into `SUPPORTED_TASKS` (0–9). See [Tasks and Perturbations](Tasks-and-Perturbations). |
| `--perturbation_id` | no | `0` | Index into `SUPPORTED_PERTURBATIONS` (0–15). |
| `--task_cfg_path` | no | derived from `--task_id` | Override path under `realm/config/tasks/`, e.g. `REALM_DROID10/rotate_marker/default.yaml`. Useful for custom variants. |
| `--repeats` | no | `5` | Number of rollouts to perform. The paper uses 25. |
| `--max_steps` | no | `500` | Hard cap on rollout length. Paper uses 800. Once the task progresses to 1.0, an extra 15 "settling" steps run regardless. |
| `--horizon` | no | `8` | Number of action-chunk steps consumed per server query. |
| `--model_name` | **yes** | — | Free-form label used as a directory name. |
| `--model_type` | **yes** | — | Inference adapter (see above). Determines how `actions[-1]` (gripper) is binarized. |
| `--port` | **yes** | — | Inference server port. |
| `--host` | no | `127.0.0.1` | Inference server host. Use the LAN address of the policy node when running across machines. |
| `--experiment_name` | **yes** | — | Top-level grouping in `logs/`. Re-use it across many `02_evaluate.py` invocations belonging to the same sweep. |
| `--run_id` | no | timestamp | Override the run timestamp. **Required when using `--resume`.** |
| `--log_dir` | no | `/app/logs/<experiment>/<model_name>` | Override where logs land. |
| `--rendering_mode` | no | `rt` | `pt` (path-traced, slow, highest fidelity), `rt` (ray-traced, default), `r` (rasterized, fastest). |
| `--multi-view` | no | off | Enable a second external camera. **Required for `dreamzero` and the drawer tasks' second camera.** |
| `--no_record` | no | off | Skip writing video frames into the per-task parquet. |
| `--no_render` | no | off | Disable rendering completely. Cameras return zeroed images. Use only with `--model_type debug` or for headless smoke tests. |
| `--resume` | no | off | Resume a partially-completed run; see below. |
| `--robot` | no | `DROID` | One of `DROID`, `WidowX`, `UR5`, `UR5_aligned`. Switches the robot YAML and adjusts physics/control frequency. |

## Rendering modes (`--rendering_mode`)

The mode flips Isaac Sim Carb settings before the env is built:

- `rt` — RTX ray tracing, the default. Best speed/fidelity tradeoff.
- `pt` — Interactive path tracing at 8 SPP plus the Optix denoiser. Highest visual
  realism, much slower. Use this only for visualizations or final qualitative results.
- `r` — Rasterized "performance" mode. Disables shadows, AO, indirect diffuse, and the
  off-screen presenter. Pairs with `gm.ENABLE_HQ_RENDERING = False`. ~2× faster than
  `rt` on most GPUs but expect visible artifacts.

## Robots and frequencies

`set_sim_config` in `realm/eval.py` sets simulation frequencies based on `--robot`:

- `DROID` (default), Panda variants → `15 Hz` step / render, `120 Hz` physics
- `WidowX` → `5 Hz` step / render
- `UR5*` → `30 Hz` step / render

Action chunks are consumed at the step frequency, so a `--max_steps 800` rollout is
~53 seconds of simulated time on DROID and ~27 seconds on UR5.

## Resume functionality

If a sweep is interrupted (cluster preemption, OOM, you hit Ctrl-C) you can pick it up
without re-running completed repeats. Resume keys on the **task name + perturbation
name** report file under `<log_dir>/reports/<task>_<perturbation>.csv`:

```bash
python /app/examples/02_evaluate.py \
    ...same args as the original... \
    --run_id 20240101_120000 --resume
```

`--run_id` must match the original run's timestamp folder. The script reads the
existing CSV, sets `start_repeat = len(rows)`, and rolls only the missing repeats. All
other arguments **must match** the original; mixing perturbations or task IDs across
resumes will produce inconsistent reports.

The cluster orchestrator (`scripts/cluster_evals/run_evals_for_ckpt.sh`) takes
`--resume` too, but its check is video-count-based — see
[Cluster and Parallel Runs](Cluster-and-Parallel-Runs).

## What "success" means

Each repeat produces a row in `<task>_<perturbation>.csv` containing:

- `task_progression` — graded score in `[0, 1]`. Stage rubrics live in
  `realm/config/tasks/task_progressions.yaml`.
- `binary_SR` — `1.0` iff `task_progression == 1.0`.
- `stage` — last completed rubric stage, or `SUCCESS` if all stages cleared.
- Smoothness metrics: joint velocity/accel variance, joint jerk, joint and Cartesian
  path length, end-effector jerk.
- Safety metrics: self-collision count, environment-collision count, drop count
  (release of an object outside the target).

## Intricacies and gotchas

- **`task_id` strings vs. paper.** The README lists `put_green_block_in_bowl` (with one
  `o` in *into*); the actual config dir is `put_green_block_into_bowl`. Always use the
  ID rather than typing the string.
- **`assert len(to_cfgs) <= 1`.** Tasks support exactly one main object and at most one
  target. Authoring a custom task with two targets will trip the assertion in
  `RealmEnvironmentDynamic.__init__`.
- **`SB-NOUN + push` is unimplemented.** The combo raises `NotImplementedError`. The
  cluster sweep script does **not** automatically skip it — the job will fail. Exclude
  perturbation 11 when running `task_id 7` (`push_switch`) or filter at the report stage.
- **`V-AUG` is applied to images, not the env.** The perturbation is realised at the
  observation step (random blur σ, contrast α). This means the env config is identical
  to `Default` and `repeats` may need `+1` to see the variation matter on cached scenes.
- **Gripper binarization is model-dependent.** `realm/eval.py` looks up `model_type`:
  `openpi`/`GR00T`/`GR00T_N16`/`dreamzero`/`debug` use `1` if `action[-1] > 0.5` else
  `-1`; `molmoact` is inverted (`< 0.5`). New model types **must** be added there or
  the gripper will misfire.
- **`--no_render` zeros the cameras.** Useful for measuring server throughput in
  `--model_type debug`, but will tank any real model's success rate.
- **`OMNIGIBSON_HEADLESS=1`.** Required when running inside the container without an
  X server. The Docker/Apptainer launcher scripts already export it on `--headless`.
- **Determinism.** `set_sim_config` seeds `random`, `numpy`, and `torch` with `1234` and
  forces `cudnn.deterministic = True`. Per-repeat reseeding is currently commented out
  in `realm/eval.py` — repeats vary because of stochastic perturbations and IK, not
  because the master seed changes.

## Programmatic usage

If you'd rather call from Python directly:

```python
from realm.eval import evaluate
evaluate(
    task_id=0,
    perturbation_id=2,        # V-VIEW
    repeats=10,
    max_steps=800,
    model_type="openpi",
    port=8000,
    log_dir="/app/logs/my_sweep/pi05/run42",
    rendering_mode="rt",
    multi_view=False,
    robot="DROID",
)
```

`evaluate` does not return anything; outputs are written to `log_dir/reports/`,
`log_dir/qpos/`, `log_dir/actions/`, and `log_dir/videos/`.
