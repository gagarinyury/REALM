# Logs, Outputs and Viewer

`02_evaluate.py` writes everything under `<log_dir>`, which by default resolves to:

```
/app/logs/<experiment_name>/<model_name>/[<run_id>/]
```

When `--run_id` is omitted, every cell of the sweep gets its own timestamp. Cluster
sweeps pin the same `--run_id` for all cells of a single sweep so the directory stays
flat.

## Directory layout

```
logs/<experiment>/<model>/<run_id>/
├── reports/                                  # CSV per (task, perturbation)
│   ├── put_green_block_into_bowl_Default.csv
│   ├── put_green_block_into_bowl_V-LIGHT.csv
│   └── ...
├── qpos/                                     # joint trajectories per task
│   ├── put_green_block_into_bowl.parquet
│   └── ...
├── actions/                                  # commanded actions per task
│   └── put_green_block_into_bowl.parquet
├── videos/                                   # H.264 mp4 bytes per task
│   └── put_green_block_into_bowl.parquet
└── (logs/<experiment>/metadata.json)         # written by run_evals_for_ckpt.sh
```

Each parquet (`qpos/`, `actions/`, `videos/`) is **per task, all perturbations & repeats
appended**. Columns:

- `task` — string, redundant copy of the file's task
- `perturbation` — string (e.g. `V-LIGHT`)
- `repeat` — int, the run_id within that cell
- `data` (qpos/actions) or `video` (videos) — nested-list / bytes payload

This means a parquet for `put_green_block_into_bowl` can contain rows for all 16
perturbations × 25 repeats = 400 rows once a sweep finishes. The viewer fans them back
out.

## CSV report fields

`reports/<task>_<perturbation>.csv` has one row per repeat. The full schema (from
`realm/eval.py`):

| Column | Meaning |
| :--- | :--- |
| `run_id` | Repeat index within the cell (0-based). |
| `task`, `perturbation`, `instruction` | Identifiers and the actual instruction sent. `S-LANG`/`S-AFF`/etc. mutate this. |
| `model`, `real2sim`, `env` | `Simulated`, `REALM`, etc. — for joining with real-world data. |
| `task_progression` | Graded score `[0,1]`. |
| `task_progression_timestamps` | Step indices at which each rubric stage cleared. |
| `stage` | First incomplete rubric stage; `SUCCESS` if all clear; `N/A` for tasks without rubrics. |
| `binary_SR` | `1.0` iff `task_progression == 1.0`. |
| `joint_vel_var`, `joint_acc_var`, `joint_jerk` | Joint-space smoothness. |
| `joint_path_length` | Sum of step-wise joint-position deltas. |
| `cart_path_length`, `cart_jerk` | EE Cartesian smoothness. |
| `collisions_self`, `collisions_env` | Edge-counted collision events (rising-edge detected on `is_self_col` / `is_env_col`). |
| `object_drops` | Number of times a held object was released without being placed. The metric subtracts 1 if the task succeeded on a `put`/`stack` (the final placement looks like a "drop"). |

`qpos`/`actions` are stripped from the CSV (they live in the parquets). `video` bytes
are also stripped from the CSV.

## Where do logs end up *on the host*?

Inside the container the path is `/app/logs/...`. Outside, that's `$REALM_ROOT/logs/...`
because `run_docker.sh` and `run_apptainer.sh` bind-mount `$REALM_ROOT:/app`. If you
override `--log_dir`, point it somewhere also bound or you'll lose the logs when the
container exits.

For a custom mount, edit the `-v $(pwd):/app:rw` in `scripts/run_docker.sh` (or the
equivalent `--bind` in the apptainer scripts).

## REALM viewer (`REALM_toolkit`)

The viewer is a Streamlit app in a separate repo:

```bash
git clone https://github.com/martin-sedlacek/REALM_toolkit.git
cd REALM_toolkit
uv sync
export REALM_LOGS=$REALM_ROOT/logs
uv run streamlit run realm_viewer/dashboard.py
```

`REALM_LOGS` must point at the **host** logs directory — not the in-container path.
The dashboard:

- enumerates experiments / models / runs from the directory tree
- summarizes per-cell success rate and smoothness from `reports/*.csv`
- unpacks any `videos/<task>.parquet` row on demand into a playable mp4

If videos look black or missing, the most common cause is `--no_record` was passed at
eval time. There's no recovery — re-run the cell.

## Reading a parquet directly

```python
import pandas as pd
df = pd.read_parquet("logs/full_eval/pi05/20260101_120000/qpos/put_green_block_into_bowl.parquet")
print(df.columns)  # ['task', 'perturbation', 'repeat', 'data']
print(df[df.perturbation == "V-LIGHT"].iloc[0]["data"][:5])  # first 5 timesteps of qpos
```

For videos:

```python
import pandas as pd, io
df = pd.read_parquet(".../videos/put_green_block_into_bowl.parquet")
row = df[(df.perturbation == "V-LIGHT") & (df.repeat == 0)].iloc[0]
with open("rollout.mp4", "wb") as f:
    f.write(row["video"])
```

## Disk-budget heuristics

- One CSV row: ~1 KB.
- One qpos/actions row at 800 steps: ~50 KB before parquet compression.
- One H.264 video at 480p / 800 steps: ~2–4 MB.
- Full sweep (10 tasks × 16 perturbations × 25 repeats = 4000 rollouts):
  - ~60 MB CSVs
  - ~200 MB qpos+actions
  - ~10–16 GB videos

Pass `--no_record` to drop the video budget if you're tight on disk; you keep
trajectories and metrics.

## Cleaning up

The cluster scripts leave per-job tmp / pip / mamba caches in `$REALM_ROOT` on
**failure**. Periodically prune:

```bash
find $REALM_ROOT/tmp $REALM_ROOT/pip_cache $REALM_ROOT/mamba_cache \
     -mindepth 1 -maxdepth 1 -mtime +3 -exec rm -rf {} +
```

The Isaac Sim caches under `$REALM_DATA_PATH/isaac-sim/cache/` are **not** safe to
delete while jobs are running and not worth deleting otherwise — clearing them
multiplies first-run wall time by ~5×.
