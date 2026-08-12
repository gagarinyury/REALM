#!/usr/bin/env bash
# Проход по всем десяти задачам REALM, Default, по одному роллауту.
# Повторы намеренно не делаем: прогон детерминирован (seed 1234 + жадное декодирование
# π0-FAST), пять повторов на win2 дали побитово одинаковые метрики — второй роллаут
# той же задачи денег стоит, а информации не даёт.
set -uo pipefail
export MAMBA_ROOT_PREFIX=/micromamba
export PATH=/micromamba/bin:$PATH
export OMNIGIBSON_DATASET_PATH=/workspace/data/og_dataset
export OMNIGIBSON_ASSET_PATH=/workspace/data/assets
export GIBSON_DATASET_PATH=/workspace/data/g_dataset
export OMNIGIBSON_KEY_PATH=/workspace/data/omnigibson.key
export OMNIVERSE_EULA_ACCEPTED=1 OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1
export PYTHONPATH="${PYTHONPATH:-}:/workspace/REALM"

EXP="${1:-stock_default}"
STEPS="${2:-800}"
OUT=/workspace/logs/$EXP.summary.txt
mkdir -p /workspace/logs
echo "REALM сток, все задачи, Default, repeats=1, max_steps=$STEPS — $(date)" | tee "$OUT"

cd /workspace/REALM
for T in 0 1 2 3 4 5 6 7 8 9; do
  t0=$(date +%s)
  micromamba run -n omnigibson python examples/02_evaluate.py \
      --task_id "$T" --perturbation_id 0 --repeats 1 --max_steps "$STEPS" \
      --model_name pi0fast --model_type openpi --port 8000 \
      --experiment_name "$EXP" --log_dir /workspace/logs \
      > "/workspace/logs/${EXP}_task${T}.log" 2>&1
  rc=$?
  echo "task $T : rc=$rc за $(( ($(date +%s) - t0) / 60 )) мин" | tee -a "$OUT"
done
echo "ГОТОВО $(date)" | tee -a "$OUT"
