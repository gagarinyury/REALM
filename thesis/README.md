# Thesis material — not part of upstream REALM

Everything under `thesis/` belongs to the bachelor's thesis

> **Simulated VLA Model Evaluation for Robotic Manipulation**
> Yahor Pachkouski, CTU FEE, Department of Measurement
> supervisor: Ing. Vladimír Petrík, Ph.D. (CIIRC)

and is kept in this fork only so that the work behind the reported numbers is
inspectable. It is **not** a contribution to `martin-sedlacek/REALM`, is not
imported by any code under `realm/`, and is not meant to be merged upstream.
The thesis compares two simulators, so the material spans both: REALM runs and
MolmoSpaces runs share the same policy server and the same episode metrics.

## `runpod/` — the scripts the reported runs were actually launched with

The experiments did not run on a workstation. REALM needs an NVIDIA GPU with RT
cores for Isaac Sim, and MolmoSpaces on CPU spent 919.9 ms of every 952.9 ms step
on camera readout — at REALM's 800-step horizon that is ~12.7 min per episode.
Both therefore ran on rented RunPod GPU pods, and these scripts are the whole
mechanism: create the pod, install the stack, serve the policy, run the benchmark,
kill the pod.

The pods are ephemeral, so without these files the runs are unreproducible — the
machine they ran on no longer exists.

| file | what it does |
| --- | --- |
| `pod.py` | create / status / wait / kill a RunPod pod via the GraphQL API. `kill` matters: an idle pod keeps billing. |
| `setup_pod.sh` | the authors' REALM stack, reproduced with no edits of ours, on top of `stanfordvl/omnigibson:1.1.1` — the image and version the published 0.61 baseline was obtained on. |
| `setup_policy_only.sh` | π0-FAST server alone, no OmniGibson, for the runs where the simulator lives elsewhere. |
| `setup_molmospaces.sh` | MolmoSpaces with EGL GPU rendering, `uv pip install -e '.[mujoco]'`, ~13 GB of assets. Documents why `uv sync` cannot be used here. |
| `serve.sh` | π0-FAST policy server in the authors' configuration (`pi0_fast_full_droid_finetune`). |
| `run_all.sh` | all ten REALM tasks, Default variant, one rollout each. No repeats: the run is deterministic (seed 1234 + greedy decoding) and five repeats on the local machine gave bit-identical metrics. |
| `run_ms_eval.sh` | MolmoSpaces under π0-FAST, 31 episodes over four benchmarks, at REALM's horizon (800 steps) and rate (15 Hz) so the numbers are comparable. |
| `run_filament.sh` | the same under the filament renderer (PBR), which is what the authors' own launch commands use. |
| `scalability_bench.sh` | single-GPU scalability, `num_workers` = 1, 2, 4, 8 — the one assignment criterion no data existed for. |
| `planner_eval_config.py` | control experiment: the same benchmark solved by a scripted planner instead of the policy. Separates "weak model" from "broken rig" when the policy scores zero. |
| `make_object_pose_variations.py` | object-pose variations of a DroidMini episode — the assignment's "simple task variations", which no shipped benchmark provides. |
| `probe_policy.py`, `probe2.py` | mute-policy check. `FASTTokenizer.extract_actions` silently returns `np.zeros` when the generated text has no `Action:` marker; the robot then drifts plausibly and metrics and videos are still written, so a silent policy has to be ruled out explicitly. |

Comments inside the scripts are in Russian — they were written as working notes
during the runs and are kept verbatim rather than retyped, so that what is in the
repository is what was executed.

## Secrets

`pod.py` reads `RUNPOD_API_KEY` from the environment, or from the dotenv file
named by `RUNPOD_ENV_FILE` (default `./.env.local`). No key is stored here. The
pod's own address was a local scratch file and is deliberately not committed.

## Where the rest of the code is

* REALM-side changes: `main` of this fork (`gagarinyury/REALM`).
* MolmoSpaces-side staged progression metric: branch `realm-staged-progress` of
  `gagarinyury/molmospaces`, not `main` — `main` there is untouched upstream.
