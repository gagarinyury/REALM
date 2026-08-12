#!/usr/bin/env bash
# Разворачивает на поде РОВНО авторскую конфигурацию REALM — без единой нашей правки.
# Запускать на поде:  bash /workspace/setup_pod.sh 2>&1 | tee /workspace/setup.log
#
# Образ пода уже равен базе из .docker/realm.Dockerfile (stanfordvl/omnigibson:1.1.1),
# поэтому здесь воспроизводятся только слои поверх него — те же команды, тот же порядок.
set -euo pipefail
step() { echo -e "\n=== [$(date +%H:%M:%S)] $* ==="; }

export MAMBA_ROOT_PREFIX=/micromamba
export PATH=/micromamba/bin:$PATH
MR="micromamba run -n omnigibson"

step "1/6 клон REALM (апстрим, main)"
cd /workspace
[ -d REALM ] || git clone --depth 1 https://github.com/martin-sedlacek/REALM.git
cd /workspace/REALM && git log -1 --format="%h %ad %s" --date=short

step "2/6 слой из realm.Dockerfile"
$MR pip list 2>/dev/null | grep -qi openpi-client || {
  micromamba install -n omnigibson -y -q -c conda-forge wandb moviepy
  $MR pip install -q /workspace/REALM/packages/openpi-client
  cp /workspace/REALM/realm/misc/modified_entity_prim.py /omnigibson-src/omnigibson/prims/entity_prim.py
  micromamba install -n omnigibson -y -q -c conda-forge dm-control==1.0.27=pyhd8ed1ab_0
  $MR pip install -q --no-cache-dir dm-robotics-controllers dm-robotics-transformations dm-robotics-geometry
  $MR pip install -q --no-cache-dir dm-robotics-moma dm-robotics-manipulation
  $MR pip install -q pandas==2.3.3 pyarrow fastparquet
  $MR pip install -q numpy==1.26.0
}

step "3/6 ассеты BEHAVIOR-1K (замер скорости)"
export OMNIGIBSON_DATASET_PATH=/workspace/data/og_dataset
export OMNIGIBSON_ASSET_PATH=/workspace/data/assets
export GIBSON_DATASET_PATH=/workspace/data/g_dataset
export OMNIGIBSON_KEY_PATH=/workspace/data/omnigibson.key
export OMNIVERSE_EULA_ACCEPTED=1 OMNI_KIT_ALLOW_ROOT=1
mkdir -p /workspace/data
if [ ! -f "$OMNIGIBSON_KEY_PATH" ]; then
  t0=$(date +%s)
  # click.confirm ждёт клавиатуру, а у фонового процесса её нет — отвечаем за него
  cd /omnigibson-src && yes | $MR python -m omnigibson.download_datasets
  echo "ассеты скачаны за $(( ($(date +%s) - t0) / 60 )) мин"
fi
du -sh /workspace/data/* 2>/dev/null || true

step "4/6 uv + openpi"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
export PATH=$HOME/.local/bin:$PATH
cd /workspace
[ -d openpi ] || git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd /workspace/openpi && git log -1 --format="%h %ad" --date=short && uv sync -q

step "5/6 чекпоинт pi0_fast_droid_jointpos (10.8 ГБ, замер скорости)"
t0=$(date +%s)
uv run python -c "
from openpi.shared import download
p = download.maybe_download('gs://openpi-assets/checkpoints/pi0_fast_droid_jointpos')
print('чекпоинт:', p)
"
echo "чекпоинт скачан за $(( ($(date +%s) - t0) / 60 )) мин"

step "6/6 готово"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "дальше: bash /workspace/serve.sh  (сервер политики), затем bash /workspace/run_all.sh"
