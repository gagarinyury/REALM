#!/usr/bin/env bash
# MolmoSpaces на поде: симуляция переезжает на карту ради EGL-рендеринга.
#
# Причина переезда — замер на evox2 (CPU, MUJOCO_GL=osmesa): из 952.9 мс на шаг
# 919.9 мс уходило на съём показаний камер и лишь 26.1 мс на саму физику. При
# горизонте REALM в 800 шагов это ~12.7 мин на эпизод. EGL-рендеринг на GPU —
# штатный путь MolmoSpaces, он описан в docstring molmo_spaces/evaluation/eval_main.py.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
export DEBIAN_FRONTEND=noninteractive
step() { echo "=== $* $(date +%H:%M:%S) ==="; }

step "1/3 системные библиотеки под EGL"
apt-get update -qq
apt-get install -y -qq git libegl1 libgl1 libglx-mesa0 libosmesa6 libglib2.0-0 >/dev/null

step "2/3 окружение (повтор сборки evox2)"
# Ставим ровно тем же способом, каким собран рабочий стенд на evox2: editable-установка
# через `uv pip install -e`, Python 3.11, extra [mujoco] (там mujoco 3.5.0, mujoco_mjx,
# mujoco_warp, torch — ни curobo, ни tensorflow не установлены).
#
# `uv sync` здесь непригоден: он резолвит универсально, для всего диапазона
# requires-python = ">=3.11", доходит до python 3.15/darwin, где нет колёс
# mujoco~=3.5.0, и падает целиком. Проверено на uv 0.9.0 и 0.12.3, с --python 3.11
# и с --python-platform linux — результат один. `uv pip install` резолвит только
# под текущий интерпретатор и платформу, поэтому проходит.
cd /workspace/molmospaces && uv venv --python 3.11 && uv pip install -e '.[mujoco]'

step "3/3 ассеты (~13 ГБ: сцены, объекты, роботы, захваты, бенчмарки)"
export MLSPACES_CACHE_DIR=/workspace/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/workspace/mlspaces-assets
export MLSPACES_FORCE_INSTALL=True
t0=$(date +%s)
cd /workspace/molmospaces && .venv/bin/python -m molmo_spaces.molmo_spaces_constants
echo "ассеты скачаны за $(( ($(date +%s) - t0) / 60 )) мин"
du -sh /workspace/molmo-spaces-resources
echo "=== ГОТОВО $(date +%H:%M:%S) ==="
