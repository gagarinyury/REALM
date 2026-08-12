#!/usr/bin/env bash
# Замер масштабируемости MolmoSpaces на одной GPU — пункт 3 задания
# («single-GPU scalability»), единственный из пяти критериев, по которому не было данных.
#
# Что меряем: один и тот же набор из 4 эпизодов при num_workers = 1, 2, 4, 8.
# Воркеры — отдельные процессы (data_generation/pipeline.py:1234), каждый со своей сценой.
# Батч сред (n_batch) здесь неприменим: task.py:438 требует n_batch == 1 при оценке,
# то есть векторизация у MolmoSpaces есть, но на пути оценки политики отключена.
#
# Сервер политики один на всех воркеров, поэтому замер показывает пропускную способность
# карты целиком — вместе с инференсом, как того и требует формулировка задания.
set -uo pipefail
export MLSPACES_CACHE_DIR=/workspace/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/workspace/mlspaces-assets
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/workspace
BENCH=/workspace/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark
CFG=molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig
EPISODES=4
cd /workspace/molmospaces

for W in 1 2 4 8; do
  echo "=== START workers=$W $(date +%H:%M:%S) ==="
  # снимаем нагрузку раз в 10 с в фоне
  ( while true; do
      echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) cpu=$(awk '{print $1}' /proc/loadavg)"
      sleep 10
    done ) > /workspace/load_w${W}.log 2>&1 &
  MON=$!

  START=$(date +%s)
  ./.venv/bin/python -m molmo_spaces.evaluation.eval_main \
      --benchmark_dir "$BENCH" \
      --output_dir /workspace/eval_out/scale_w${W} \
      --max_episodes $EPISODES \
      --num_workers $W \
      --task_horizon_steps 800 \
      --no_wandb \
      "$CFG" > /workspace/scale_w${W}.log 2>&1
  RC=$?
  END=$(date +%s)
  kill $MON 2>/dev/null

  DONE=$(grep -cE "completed with success" /workspace/scale_w${W}.log)
  echo "=== DONE workers=$W rc=$RC время=$((END-START))с эпизодов=$DONE $(date +%H:%M:%S) ==="
done
echo "=== ЗАМЕР ЗАВЕРШЁН $(date +%H:%M:%S) ==="
