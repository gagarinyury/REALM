#!/usr/bin/env bash
# Установка ТОЛЬКО сервера политики π0-FAST.
#
# Отличие от setup_pod.sh: симуляция здесь не нужна — её считает MolmoSpaces
# на evox2 (CPU), а под держит одну лишь модель. Поэтому нет ни OmniGibson,
# ни его датасетов; шаги 4 и 5 setup_pod.sh перенесены как есть.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
export DEBIAN_FRONTEND=noninteractive
step() { echo "=== $* $(date +%H:%M:%S) ==="; }

step "1/4 git + системное"
apt-get update -qq && apt-get install -y -qq git curl >/dev/null

step "2/4 uv"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }

step "3/4 openpi"
cd /workspace
[ -d openpi ] || git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd /workspace/openpi && git log -1 --format="%h %ad" --date=short && uv sync

step "4/4 чекпоинт pi0_fast_droid_jointpos (10.84 ГБ)"
t0=$(date +%s)
uv run python -c "
from openpi.shared import download
p = download.maybe_download('gs://openpi-assets/checkpoints/pi0_fast_droid_jointpos')
print('чекпоинт:', p)
"
echo "чекпоинт скачан за $(( ($(date +%s) - t0) / 60 )) мин"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "=== ГОТОВО $(date +%H:%M:%S) ==="
