# Troubleshooting

A grab bag of failures you'll hit in practice and how to recover. If something here is
wrong or missing, please open an issue or PR against this wiki page.

## Setup / install

**`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`**
Install the NVIDIA Container Toolkit and restart Docker. This is unrelated to REALM.

**Apptainer build fails without a clear error.**
Re-run `./setup.sh --apptainer --force-build` and watch the output. The script auto-
retries with `sudo` if the unprivileged build fails — make sure you have sudo, or
build the image on another host and copy the SIF.

**`OMNIVERSE_EULA_ACCEPTED is not set` when running non-interactively.**
Either accept it interactively once (it persists in `~/.bashrc`) or export it before
launching: `export OMNIVERSE_EULA_ACCEPTED=1`. The cluster scripts assume this.

**`REALM_SIF is not set` / `REALM_DATA_PATH is not set`.**
`setup.sh --apptainer --dataset` writes both. If you ran setup as a different user, add
the exports to your own `~/.bashrc`.

## First-run sluggishness

The very first benchmark cell on a fresh install spends 5–15 minutes building Isaac
Sim shader caches under `$REALM_DATA_PATH/isaac-sim/cache/{kit,glcache,computecache}`.
Subsequent runs are dramatically faster. Don't kill the process during the first run —
you'll restart the cache build from scratch.

## Container & runtime

**`OMNIGIBSON_HEADLESS=1` reminders.**
Required when running inside the container without an X server. Both
`scripts/run_docker.sh` and `scripts/run_apptainer.sh` set this on `--headless`. Inside
the container, prepend it to every `python /app/examples/...` invocation.

**`RuntimeError: CUDA error: invalid device function`.**
Usually means the cached PTX doesn't match the host GPU. Set
`CUDA_FORCE_PTX_JIT=1` (Docker launcher already does this) and rerun. If it persists,
delete `$REALM_DATA_PATH/isaac-sim/cache/computecache/` and let it rebuild.

**`xformers` ImportError or version mismatch when starting an `openpi` server.**
Pin to the upstream-recommended versions from the openpi README; REALM doesn't manage
the openpi env. The model server runs in its own SIF / venv, isolated from the REALM
container.

**Apptainer crashes mid-run with "page fault" or similar.**
Known instability — see the Installation page. Options: switch to Docker if available,
or pre-warm the Isaac Sim caches with one short cell before the long sweep.

## Evaluation logic

**`assert task_cfg_path` mismatch / wrong task name in logs.**
The eval derives the task name from the *path*, not the ID. `default.yaml` collapses
to the bare task name; any other filename is appended. If your CSV reports under
`rotate_mug_heavy` instead of `rotate_mug`, you used a non-default config file.

**`SB-NOUN` job hangs / errors on `push_switch`.**
That combination raises `NotImplementedError` deliberately. Skip perturbation 11 for
task 7 in your sweep filters.

**`assert base_im_second is not None`** on `dreamzero`.
You forgot `--multi-view`. Add it to both the orchestrator and the inner script.

**`NotImplementedError` in the gripper-binarization block.**
You passed a `--model_type` that hasn't been wired into `realm/eval.py`. Add a branch
near the bottom of the rollout loop. See [Inference Servers](Inference-Servers).

**Server connection times out.**
`scripts/eval.sh` waits up to `MODEL_SERVER_TIMEOUT=180s` for the server's port to start
listening. Increase if your model needs longer to load: `export MODEL_SERVER_TIMEOUT=600`
before invoking. The cluster script sleeps a fixed `120` seconds; for slow checkpoints,
edit the `sleep 120` lines.

**Server replies but the rollout immediately terminates.**
Inspect the CSV's `stage` column — `"SUCCESS"` means the rubric thought the task
finished, which can happen if the perturbation moved the goal object onto/inside the
target at spawn time. `task_progression` will show `1.0` from step 0. Tighten the
`relative_bbox_position` fields in the task YAML or filter the perturbation.

## Resume

**`Resume requested but no report found. Starting fresh.`**
You set `--resume` but didn't pin `--run_id` or the timestamp folder doesn't exist.
Re-pass the *exact* `run_id` from the original sweep. Check the `logs/<exp>/<model>/`
listing.

**Resumed sweep skips cells that look incomplete.**
The cluster orchestrator's resume check is video-count based: ≥`REPEATS` mp4s in the
videos dir → cell skipped. If a job died after writing videos but before the CSV, you
get an inconsistent state. Delete the affected `<task>_<perturbation>.csv` and the
matching rows in the parquet, then re-run.

## Disk

**Logs eating the disk.**
A full sweep with videos is ~10–16 GB. Pass `--no_record` to skip videos when you only
need metrics, or relocate the log dir with `--log_dir /scratch/$USER/realm_logs`. Make
sure that path is bind-mounted into the container.

**`isaac-sim/cache` is huge (> 50 GB).**
Mostly normal — `ov` (asset cache) and `glcache` (shader cache) grow with every new
scene. Don't delete unless you're decommissioning the install; clearing it costs hours
on the next run.

## Cluster

**Two jobs collide on a port.**
`run_single_eval.sh` derives the port as `BASE_PORT + perturbation + 100*task`. Two
sweeps on the same `BASE_PORT` will collide. Bump the second sweep with
`--base_port 9000`.

**Per-job tmp / pip / mamba caches accumulate.**
Failed jobs preserve them by design. Periodically:
```bash
find $REALM_ROOT/tmp $REALM_ROOT/pip_cache $REALM_ROOT/mamba_cache \
     -mindepth 1 -maxdepth 1 -mtime +3 -exec rm -rf {} +
```

**SLURM kills the job at the time limit.**
Default is `04:30:00`. Bump `#SBATCH --time` in `run_single_eval.sh` for `--rendering_mode pt`
or `--repeats 50`. 25 repeats × 800 steps at `rt` finish in ~3h on an L40s with a
fast policy.

## Where to look first

For a new failure, in this order:
1. **Inside the container:** `tail -200 $REALM_ROOT/logs/<exp>/<model>/<run_id>/*.log`
2. **SLURM:** `slurm-<jobid>.out` next to the orchestrator's working dir
3. **Isaac Sim's own logs:** `$REALM_DATA_PATH/isaac-sim/logs/`
4. **Open an issue** with the failing command, the relevant tail, and your container
   type.
