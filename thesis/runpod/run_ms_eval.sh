#!/usr/bin/env bash
# Прогон MolmoSpaces под π0-FAST: 31 эпизод на четырёх бенчмарках.
#
# Состав выбран так, чтобы каждая позиция отвечала на свой вопрос:
#   pick_and_place 10 — прямой аналог REALM put_green_block_into_bowl (там 1.00)
#   pick           10 — аналог pick_spoon (в REALM 0.00, дальше REACH не ушёл)
#   open / close   5+5 — обе задачи с ящиками, которые на REALM 3.9.1 не грузятся вовсе
#   + 1 пробный эпизод перед серией
#
# Горизонт 800 шагов и частота 15 Гц (policy_dt_ms=66) взяты из REALM, чтобы числа
# были сопоставимы; штатное значение MolmoSpaces — 500.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
export MLSPACES_CACHE_DIR=/workspace/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/workspace/mlspaces-assets
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
BENCH=/workspace/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-objaverse
OUT=/workspace/eval_out
CFG=molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig
cd /workspace/molmospaces

run() {  # run <имя> <каталог бенчмарка> <сколько эпизодов>
  local name=$1 dir=$2 n=$3
  echo "=== START $name ($n эп.) $(date +%H:%M:%S) ==="
  ./.venv/bin/python -m molmo_spaces.evaluation.eval_main \
      --benchmark_dir "$BENCH/$dir" \
      --output_dir "$OUT/$name" \
      --max_episodes "$n" \
      --task_horizon_steps 800 \
      --no_wandb \
      "$CFG"
  echo "=== DONE $name rc=$? $(date +%H:%M:%S) ==="
}

# smoke-эпизод уже прогнан отдельно 11.08 09:31 (2 мин, success=False, 101 вызов политики),
# результат в /workspace/eval_out/smoke — здесь не повторяем.
run pnp    FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260212_200ep_json_benchmark 10
run pick    FrankaPickHardBench/FrankaPickHardBench_20260212_200ep_json_benchmark 10
run open    FrankaOpenHardBench/FrankaOpenHardBench_20260212_200ep_json_benchmark 5
run close   FrankaCloseHardBench/FrankaCloseHardBench_20260212_200ep_json_benchmark 5
echo "=== ВСЁ ЗАВЕРШЕНО $(date +%H:%M:%S) ==="
