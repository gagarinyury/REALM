# REALM Wiki

REALM is a real-to-sim validated benchmark for generalization in robotic manipulation.
This wiki collects everything you need to install REALM, run the benchmark, scale evaluations
to a multi-GPU cluster, and inspect the resulting logs.

If this is your first time, start with **Installation** and then **Quick Start**.
For everything else, jump straight to the topic you need.

## Pages

- [Installation](Installation) — `setup.sh`, container choice, dataset download, environment variables
- [Quick Start](Quick-Start) — first evaluation in under 10 minutes
- [Running Evaluations](Running-Evaluations) — `examples/02_evaluate.py`, every CLI flag, gotchas
- [Tasks and Perturbations](Tasks-and-Perturbations) — the 10 tasks, the 16 perturbations, their YAML configs
- [Inference Servers](Inference-Servers) — wiring up `openpi`, `GR00T`, `molmoact`, `dreamzero`, `debug`
- [Cluster and Parallel Runs](Cluster-and-Parallel-Runs) — SLURM, port allocation, multi-GPU sharding
- [Logs, Outputs and Viewer](Logs-Outputs-and-Viewer) — directory layout, parquet dumps, REALM_toolkit dashboard
- [Robots and Rendering Modes](Robots-and-Rendering-Modes) — DROID/UR5/WidowX, `pt`/`rt`/`r`, multi-view
- [Troubleshooting](Troubleshooting) — common errors and recovery

## Where to ask for help

- Issues: <https://github.com/martin-sedlacek/REALM/issues>
- Discussions: <https://github.com/martin-sedlacek/REALM/discussions>
- Project page: <https://martin-sedlacek.com/realm>
- Paper: <https://arxiv.org/abs/2512.19562>
