# Quick Start

This walks through your first REALM evaluation end-to-end: serve a policy, run one task
under the default perturbation, and view the rollout.

## 1. Start a policy server

The reference example uses [openpi](https://github.com/Physical-Intelligence/openpi)
serving a Pi0.5 checkpoint. Run this on a host with a free GPU:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
uv sync
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_full_droid_finetune \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid_jointpos
```

The server listens on `127.0.0.1:8000` by default.

If you're using a different model family, jump to [Inference Servers](Inference-Servers)
for the equivalent commands for `GR00T`, `molmoact`, and `dreamzero`.

## 2. Open the REALM container

From the REALM repo root:

```bash
cd $REALM_ROOT          # set this to wherever you cloned REALM
source ./scripts/run_docker.sh
```

The script accepts `-h`/`--headless` to run with `OMNIGIBSON_HEADLESS=1` and no X11
forwarding. It will prompt for the Omniverse EULA on first launch.

## 3. Run a single rollout

Inside the container:

```bash
OMNIGIBSON_HEADLESS=1 python /app/examples/01_pi0_eval.py
```

This runs `task_id=1` (`put_banana_into_box`) under perturbation `0` (`Default`) for one
repeat at 500 max steps, talking to the policy on `127.0.0.1:8000`. Expect the first
launch to spend several minutes building Isaac Sim caches; subsequent runs are much
faster because of the cache mounts under `$REALM_DATA_PATH/isaac-sim/cache/`.

## 4. View the result

REALM logs land at `$REALM_ROOT/logs/<experiment>/<model>/<run_id>/...`. To browse them
through the viewer, install the toolkit on the host:

```bash
export REALM_LOGS=$REALM_ROOT/logs       # must point at the host-side logs dir

git clone https://github.com/martin-sedlacek/REALM_toolkit.git
cd REALM_toolkit
uv sync
uv run streamlit run realm_viewer/dashboard.py
```

Open the run, scroll to the bottom, click **unpack video parquet**, and the rollout
video plays.

## 5. Next steps

- Want to sweep all 16 perturbations × 10 tasks? See
  [Cluster and Parallel Runs](Cluster-and-Parallel-Runs).
- Want every flag and what it does? See [Running Evaluations](Running-Evaluations).
- Hit a `cuda out of memory` or `xformers` error? See [Troubleshooting](Troubleshooting).
