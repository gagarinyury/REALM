#!/usr/bin/env bash
# Прогон на рендерере filament — «максимальное качество» MolmoSpaces.
#
# Весь день считали на legacy OpenGL: MuJoCo 3.5.0 из extra [mujoco]. У проекта есть
# второй рендерер, filament (PBR, физически корректные материалы и освещение), и
# авторские команды запуска идут именно с ним (--use-filament в gen_eval_cmds.py).
# Включается он не флагом, а самой библиотекой:
#     env.py:35   HAS_FILAMENT = getattr(mujoco, "mjRENDERER", "classic") == "filament"
# то есть нужно их собственное колесо mujoco 3.7.1 из bin/wheels.
#
# ВАЖНО: вместе с рендерером меняется и версия физики (3.5.0 -> 3.7.1), поэтому
# результаты этого прогона нельзя ставить в один ряд с предыдущими без пометки.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
export MLSPACES_CACHE_DIR=/workspace/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/workspace/mlspaces-assets
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/workspace
BENCH=/workspace/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-10k/FrankaPickandPlaceDroidMiniBench/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark
cd /workspace/molmospaces

echo "=== ждём окончания прогона вариаций $(date +%H:%M:%S) ==="
while pgrep -f "eval_main" > /dev/null 2>&1; do sleep 15; done
sleep 5

echo "=== ставим колесо filament $(date +%H:%M:%S) ==="
uv pip install --reinstall bin/wheels/mujoco-3.7.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl 2>&1 | tail -3

echo "=== проверка, что рендерер переключился ==="
./.venv/bin/python -c "
import mujoco
print('mujoco', mujoco.__version__, '| mjRENDERER =', getattr(mujoco, 'mjRENDERER', 'classic'))
import molmo_spaces
from molmo_spaces.env.env import HAS_FILAMENT
print('HAS_FILAMENT =', HAS_FILAMENT)
"

echo "=== прогон 3 эпизодов на filament $(date +%H:%M:%S) ==="
START=$(date +%s)
./.venv/bin/python -m molmo_spaces.evaluation.eval_main \
    --benchmark_dir "$BENCH" \
    --output_dir /workspace/eval_out/filament \
    --max_episodes 3 \
    --num_workers 1 \
    --task_horizon_steps 800 \
    --use-filament \
    --no_wandb \
    molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig > /workspace/filament.log 2>&1
echo "=== ГОТОВО rc=$? время=$(( $(date +%s) - START ))с $(date +%H:%M:%S) ==="
grep -cE "completed with success" /workspace/filament.log | xargs -I{} echo "эпизодов: {}"
