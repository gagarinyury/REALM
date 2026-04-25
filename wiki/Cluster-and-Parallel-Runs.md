# Cluster and Parallel Runs

A full benchmark sweep is 10 tasks × 16 perturbations × 25 repeats × 800 steps. On a
single GPU that's ~3 days of wall time; this page covers how to shard it across a SLURM
cluster, how port allocation prevents two policy servers from colliding, and how to
recover from preemption.

The relevant scripts are:

```
scripts/cluster_evals/
├── run_evals_for_ckpt.sh   # orchestrator — fans out (task, perturbation) pairs to sbatch
└── run_single_eval.sh      # one SLURM job: starts a policy server + runs evaluate
```

Plus `scripts/eval.sh` which is the single-machine analogue (handles port locking on a
shared workstation).

## Architecture in one diagram

```
run_evals_for_ckpt.sh
  └── for task in TASK_IDS:
        for pert in PERT_IDS:
              sbatch run_single_eval.sh --task_id <t> --perturbation_id <p> ...
                ├── apptainer exec POLICY_SIF serve_policy.py --port=<p+100*t>      # policy GPU process
                └── apptainer exec REALM_SIF  examples/02_evaluate.py --port=<...>  # sim GPU process
```

Each SLURM job is **self-contained**: one GPU, one policy server, one sim, one
(task, perturbation) pair, `repeats` rollouts. Parallelism comes from launching `T × P`
jobs concurrently.

## Required environment variables

The cluster scripts assume these are set in your shell (they're written by `setup.sh`
plus a few extras you set yourself):

| Variable | Used for |
| :--- | :--- |
| `REALM_ROOT` | Repo root, used as the working dir of `run_evals_for_ckpt.sh` |
| `REALM_DATA_PATH` | Bound into the container at `/data` and the Isaac Sim cache mount points |
| `REALM_SIF` | Apptainer image for the simulator side |
| `OPENPI_ROOT` | Path to your openpi clone (for `--model_type openpi`) |
| `OPENPI_SIF` | Apptainer image for the openpi server (the cluster script hard-codes a path; override it for your environment) |
| `GR00T_ROOT` | Path to your GR00T clone (for `--model_type GR00T`) |
| `HF_HOME`, `HUGGINGFACE_HUB_CACHE` | Cluster scripts re-point these at `$REALM_ROOT/hf_cache` |
| `XDG_CACHE_HOME` | Re-pointed to `$REALM_ROOT/python_cache` |

> ⚠ **Hard-coded paths.** `run_single_eval.sh` references
> `/scratch/project/open-34-32/sedlam/projects/REALM_openpi/uv_cuda128.sif` and a
> sibling molmoact SIF. These are author-specific and will not exist on your cluster —
> edit them or pass the right paths through your own wrapper before running.

## SBATCH defaults

`run_single_eval.sh` ships with these resource requests:

```bash
#SBATCH --partition l40s
#SBATCH --gpus 1
#SBATCH --mem 120G
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-gpu 64
#SBATCH --time 00-04:30:00
```

For a sweep, this gives one L40s + 64 cores + 120 GB per (task, perturbation) cell.
Adjust the partition/`--time` for your scheduler. 4h30 is enough for 25 repeats × 800
steps on most VLAs at `rt`; bump it if you use `pt` or run with `--repeats 50`.

## Orchestrator flags (`run_evals_for_ckpt.sh`)

```bash
bash scripts/cluster_evals/run_evals_for_ckpt.sh \
    --policy_config pi05_full_droid_finetune \
    --checkpoint_path /scratch/$USER/checkpoints/pi05/some_step \
    --policy_run_dir $OPENPI_ROOT \
    --model_type openpi \
    --task_ids 0-9 \
    --perturbation_ids 0-15 \
    --repeats 25 \
    --max_steps 800 \
    --base_port 8000
```

| Flag | Meaning |
| :--- | :--- |
| `--policy_config` | Passed straight through to `serve_policy.py --policy.config=`. |
| `--checkpoint_path` | Bind-mounted into the policy SIF at `/checkpoint`. The path's last two segments become `MODEL_NAME` (e.g. `pi05_step_50000`). |
| `--policy_run_dir` | Working dir before the policy SIF is launched. Normally `$OPENPI_ROOT`. |
| `--model_type` | `openpi`, `GR00T`, `molmoact`. Picks which server-launch branch fires inside `run_single_eval.sh`. |
| `--task_ids` | Comma- or range-list, e.g. `0,2,5` or `0-9` or `0-3,7,9`. Default `0-9`. |
| `--perturbation_ids` | Same syntax. Default `0-15`. |
| `--task_cfg_path` | Forward a custom task YAML (see [Tasks and Perturbations](Tasks-and-Perturbations)). |
| `--repeats`, `--max_steps`, `--rendering_mode`, `--multi-view`, `--no_render`, `--robot` | Forwarded to every child job. |
| `--experiment_name` | Defaults to `t<TASKS>_p<PERTS>_s<MAX_STEPS>_r<REPEATS>`. Override to keep parallel sweeps separate. |
| `--run_id` | Override the timestamp folder. Useful for `--resume`. |
| `--base_port` | Starting port for policy servers, see below. |
| `--debug` | Skips the policy server and uses `model_type=debug` end-to-end. |
| `--resume` | Skip cells where `<VIDEO_DIR>/<task>_<pert>_*.mp4` already has ≥`REPEATS` files. Forwards `--resume` to the inner script too. |

The orchestrator writes `logs/<EXPERIMENT_NAME>/metadata.json` with the requested
ranges, then calls `sbatch` for every (task, perturbation) pair.

## Port allocation

Two patterns coexist:

**Cluster (deterministic).** `run_single_eval.sh` computes
```bash
port=$((BASE_PORT + PERTURBATION_ID + 100 * TASK_ID))
```
With `BASE_PORT=8000`, `task_id=3, perturbation_id=7` → port `8307`. This means:
- `task_id` shifts the port by 100 — supporting up to 100 perturbations per task without
  collision.
- Two jobs for the same `(task, perturbation)` *will* clash on the same port. Don't
  start a second sweep on the same checkpoint with the same `--base_port` while the
  first is running; bump `--base_port` by `1000` instead.
- `task_ids=0-9, perturbation_ids=0-15` claims ports `8000–8915` (with gaps).

**Single-host (random + filesystem lock).** `scripts/eval.sh` randomly picks a port in
`20000–25000`, checks that nothing's listening (`nc` or `ss`), then
`mkdir`s a lock under `/tmp/model_server_ports/<port>.lock`. The lock dir falls back to
`/tmp/model_server_ports_$USER` if the shared one isn't writable. The lock is released
in the `EXIT` trap.

If the cluster's port range collides with another tenant on the node, override
`--base_port`. Anything above `8000` and below `32768` is generally fine; coordinate
with cluster admins for restricted networks.

## Multi-GPU strategies

REALM does not currently support a vectorized environment ([roadmap]) — one `evaluate`
call uses one GPU. For multi-GPU throughput you have three options:

**1. SLURM fan-out (recommended, what the orchestrator does).**
Each (task, perturbation) cell is its own job on its own GPU. `T × P = 160` jobs run
fully in parallel up to your queue limit.

**2. Multi-instance on one node.**
Launch multiple containers manually, each binding a different GPU via
`CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 \
    python /app/examples/02_evaluate.py \
        --task_id 0 --perturbation_id 0 --port 8000 ... &

CUDA_VISIBLE_DEVICES=1 OMNIGIBSON_HEADLESS=1 \
    python /app/examples/02_evaluate.py \
        --task_id 1 --perturbation_id 0 --port 8001 ... &
```

Make sure each instance hits a **different** policy server (or run one server per GPU
and bump the policy `--port`). Isaac Sim is heavy on host RAM (~30–60 GB resident);
budget for that, not just VRAM.

**3. Per-task sharding via shell loop.**
Cheap parallelism without SLURM, suited to a 4–8 GPU workstation:

```bash
for t in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$t \
    python /app/examples/02_evaluate.py \
      --task_id $t --perturbation_id 0 \
      --port $((8000 + t)) \
      --model_name pi05 --model_type openpi \
      --experiment_name workstation_run \
      > logs/task_$t.log 2>&1 &
done
wait
```

You'll need one openpi server per port — either run 8 servers, or use a single multi-GPU
server if the policy supports it.

## Caches under SLURM

`run_single_eval.sh` redirects per-job caches to per-`SLURM_JOB_ID` directories:

```
$REALM_ROOT/tmp/$SLURM_JOB_ID            -> /tmp inside the SIF
$REALM_ROOT/mamba_cache/$SLURM_JOB_ID    -> $MAMBA_CACHE_DIR
$REALM_ROOT/pip_cache/$SLURM_JOB_ID      -> $PIP_CACHE_DIR
```

These are removed on a clean exit and **preserved** on failure for debugging. Sweep
the directory periodically — they accumulate fast on a busy queue.

The Isaac Sim render caches (`isaac-sim/cache/{kit,ov,glcache,computecache}`) are
shared across jobs and do **not** get the per-job suffix. This is intentional — it's
where most of the first-run pain disappears — but means concurrent jobs on a fresh
node can race on cache writes. Pre-warm by running one cell solo before launching the
full sweep.

## Resume on the cluster

```bash
bash scripts/cluster_evals/run_evals_for_ckpt.sh \
    ...same args... \
    --run_id 20260101_120000 \
    --resume
```

The orchestrator counts videos under `logs/<exp>/<model>/<run_id>/videos/` for each
`(task, perturbation)` pair. If `count >= repeats`, it skips that `sbatch`. Inside
each launched job, the `--resume` flag is also forwarded to `02_evaluate.py`, which
resumes mid-cell from the CSV row count.

> ⚠ Video count vs. CSV row count can drift if a job crashed *between* writing the
> video and the CSV. If a partially-resumed cell looks wrong, delete the
> `<task>_<perturbation>.csv` and the matching `<task>.parquet` rows for that
> perturbation, then re-run.

## Apptainer-only quirks

- `--writable-tmpfs` is required for both the policy and sim SIFs.
- `--userns` is set on the sim SIF to avoid root-mapped writes into the bind-mounts.
- `NVIDIA_DRIVER_CAPABILITIES=all` and `--nv` are both needed; either alone causes
  Isaac Sim to fail to find CUDA libraries on some hosts.
- The `OMNIVERSE_EULA_ACCEPTED=1` env from `setup.sh` is **not** propagated through
  `apptainer exec --env`; the cluster script doesn't need it because the sim image
  already has the EULA baked in. If you build your own image, propagate it explicitly.

## Sanity checks before launching a 160-job sweep

1. Run **one** cell end-to-end with `--debug` to verify the SLURM template:
   ```bash
   bash scripts/cluster_evals/run_evals_for_ckpt.sh \
       --task_ids 0 --perturbation_ids 0 \
       --debug --repeats 1 --max_steps 50 \
       --experiment_name sanity_check
   ```
2. Run **one** real cell with the policy you intend to evaluate. Confirm the video
   parquet is non-empty and the CSV row reports a reasonable `task_progression`.
3. Then unleash the full sweep with the same `--experiment_name` and `--resume` so the
   sanity-check cell isn't re-run.
