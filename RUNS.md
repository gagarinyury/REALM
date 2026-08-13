# Test bench and run settings

Every number in this branch's commit messages was produced on the machine and with the
settings below. Recorded because a result without them is not reproducible — and because
half of what we spent days chasing turned out to be a property of the setup rather than of
the model.

## Machine

```
OS        Windows Server 2025 Standard, build 26100 (native -- the simulator does NOT run under WSL2)
GPU       NVIDIA GeForce RTX 5080, 16303 MiB, driver 610.47
CPU       AMD Ryzen 9 9900X, 12 cores
RAM       31 GB
```

The policy server runs in WSL2 (Ubuntu 22.04) on the same GPU. That split is forced, not
chosen: Isaac Sim cannot render under WSL2 (no Vulkan/RTX passthrough), and openpi has no
Windows build.

## Software

```
Python           3.11.15  (conda env "behavior")
omnigibson       3.9.1    (BEHAVIOR-1K, editable install)
isaacsim         5.1.0.0
bddl             3.7.0
torch            2.7.0+cu128
numpy            1.26.0   (pinned by .docker/og391-constraints.txt; the installer ships 1.26.4)
gymnasium        1.3.0
mujoco           3.2.7    | dm_control | osqp 0.6.7.post3
                          these three replace dm_robotics, which has no Windows build
dataset          behavior-1k-assets, downloaded 03.08.2026, 1829 object categories
```

Engine patches applied: `realm/misc/entity_prim_og391.patch`,
`realm/misc/usd_object_og391.patch`, plus a relaxation of the pose asserts in
`entity_prim.py` (applied by `scripts/install_windows.ps1`, stage `patches`).
`vulkan = true` is forced off in the Isaac Sim kit files — on Windows it crashes at startup.

## Policy

```
model      pi0_fast_droid_jointpos
config     pi0_fast_droid_jointpos_polaris
checkpoint gs://openpi-assets/checkpoints/pi0_fast_droid_jointpos
transport  websocket, 127.0.0.1:8000
memory     XLA_PYTHON_CLIENT_MEM_FRACTION=0.47
```

0.47 is measured, not chosen: at 0.45 the server dies with `RESOURCE_EXHAUSTED`, and above
0.47 the simulator no longer fits alongside it on a 16 GB card. The server takes about
9.5 GB, the simulator about 6.

## Run settings

```
rendering_mode   rt
max_steps        800
repeats          1
perturbation     0 (Default)
robot            DROID
sim step freq    15 Hz   | rendering 15 Hz | physics 120 Hz
```

`rt` rather than `r` for a concrete reason: `eval.py:230` binarises the gripper channel at
0.5, and in the reduced render mode the policy's output peaks at 0.487 — the gripper never
closes and every grasp task caps out at REACH.

**`repeats: 1` is a real limitation of these numbers.** The same task on an unchanged config
has given both 0.8 and 1.0 on separate runs. Single-rollout figures show direction, not
magnitude; treat any comparison of two single runs as indicative.

## Timing

Measured on `pick_spoon` in `Merom_1_int`, 191 s wall clock end to end:

```
 94 s   Isaac Sim startup and scene load (153 objects; the IMPACT bench loads in ~20 s)
 97 s   463 simulation steps
```

That is **4.8 steps/s against the configured 15 Hz**, i.e. 30.9 s of robot time in 97 s of
wall clock — **3.1x slower than real time**. The load cost is fixed per run regardless of
episode length, and it dominates on short tasks.

When comparing simulators, report these two numbers separately: merged into one
"time per run" they mostly reflect which scene was chosen.

## How to reproduce

```powershell
# from nothing
.\scripts\install_windows.ps1 -DatasetFrom <path-to-existing-datasets>   # or without it, to download

# one task
.\scripts\run_task_windows.ps1 -TaskId 6 -TaskCfgPath IMPACT/stack_cubes/default.yaml -Experiment smoke
```

The installer was verified by wiping this machine — conda env and both repositories removed,
only the dataset kept — and reinstalling from scratch. That exercise found seven defects in
the script that no amount of running it against the already-working machine had revealed.
