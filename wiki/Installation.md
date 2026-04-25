# Installation

REALM ships as a containerized environment around Isaac Sim / OmniGibson.
You install it via the `setup.sh` script at the repo root, which builds (or re-uses) a
Docker image **or** an Apptainer/Singularity image, optionally downloads the BEHAVIOR-1K
dataset, and writes the right environment variables to `~/.bashrc`.

## Prerequisites

- An NVIDIA GPU (RTX-class or newer recommended). The benchmark exercises Isaac Sim's
  ray-traced renderer, so consumer/server GPUs without RT cores will run very slowly.
- NVIDIA driver compatible with CUDA 12 / Isaac Sim.
- One of:
  - Docker with the NVIDIA Container Toolkit (recommended for workstations)
  - Apptainer 1.x **or** Singularity (recommended for HPC clusters)
- ~150 GB free disk for the container images, dataset, and Isaac Sim caches.
- You must accept the [NVIDIA Omniverse EULA](https://docs.omniverse.nvidia.com/app_isaacsim/common/NVIDIA_Omniverse_License_Agreement.html).
  `setup.sh` will prompt you and then export `OMNIVERSE_EULA_ACCEPTED=1` to your shell.

## `setup.sh` flags

```bash
./setup.sh [--docker|--apptainer] [--dataset] [--data-path PATH] [--sif-path PATH] [--force-build]
```

| Flag | Effect |
| :--- | :--- |
| `--docker` | Build the `realm:latest` Docker image from `.docker/realm.Dockerfile`. Default if neither `--docker` nor `--apptainer` is given. |
| `--apptainer` | Build the `realm.sif` Apptainer image from `.docker/realm.def`. Falls back to `singularity` if `apptainer` is not on PATH, and retries with `sudo` if the unprivileged build fails. |
| `--dataset` | Download the BEHAVIOR-1K assets into `<DATA_PATH>/datasets`. Skips the download if absent — but in that case you must download separately before running the benchmark. |
| `--data-path PATH` | Directory the dataset and Isaac Sim caches live under. Default: `<repo>/data`. Bound into the container at runtime. |
| `--sif-path PATH` | Output path for the SIF image (Apptainer only). Default: `<repo>/realm.sif`. |
| `--force-build` | Rebuild the image even if one already exists. |

`--docker` and `--apptainer` are mutually exclusive. `--sif-path` only makes sense with
`--apptainer`.

## Recommended invocations

```bash
# Workstation, first time
./setup.sh --docker --dataset

# Workstation, dataset already present somewhere else
./setup.sh --docker --dataset --data-path /scratch/realm_data

# HPC node where Docker is forbidden
./setup.sh --apptainer --dataset --sif-path /scratch/$USER/realm.sif
```

> ⚠ **Apptainer is currently less stable than Docker** — the upstream image
> occasionally crashes inexplicably on some host kernels. Prefer Docker on
> machines where you have the choice.

## Environment variables `setup.sh` writes

`setup.sh` appends managed blocks to `~/.bashrc`. Re-running it edits these blocks in
place rather than appending duplicates.

| Variable | Set when | Purpose |
| :--- | :--- | :--- |
| `OMNIVERSE_EULA_ACCEPTED` | You accept the EULA | Skips the EULA prompt on subsequent runs |
| `REALM_DATA_PATH` | `--dataset` | Where the OmniGibson dataset and Isaac Sim caches live; bound into the container |
| `REALM_SIF` | `--apptainer` | Absolute path of the built SIF image |

After installation you'll typically want to add **one more** variable, pointing at a
log directory the host can read (used by the viewer):

```bash
export REALM_LOGS=$REALM_ROOT/logs
```

## What gets created on disk

```
<DATA_PATH>/
├── datasets/                      # BEHAVIOR-1K assets bound to /data
└── isaac-sim/
    ├── cache/{kit,ov,pip,glcache,computecache}/
    ├── config/, data/, documents/, logs/
```

These directories are bind-mounted by every container launch (see
`scripts/run_docker.sh` and `scripts/run_apptainer.sh`). Don't move them after install —
move the whole tree and update `REALM_DATA_PATH`.

## Verifying the install

```bash
# Drop into the container shell
source ./scripts/run_docker.sh           # or: source ./scripts/run_apptainer.sh
# Inside the container:
OMNIGIBSON_HEADLESS=1 python /app/examples/01_pi0_eval.py --help  # arg parsing only
```

For an end-to-end check, see [Quick Start](Quick-Start).

## Custom Isaac Sim patch

The repo ships `scripts/patch_isaacsim.sh` which is invoked at image build time to apply
a few small fixes on top of the upstream Isaac Sim. You shouldn't need to run it
manually unless you're rebuilding the image yourself.
