#!/usr/bin/env bash
# Сервер политики π0-FAST в авторской конфигурации.
# По README REALM для π0.5 пара такая: pi05_full_droid_finetune + pi05_droid_jointpos.
# Для π0-FAST (модель из статьи, baseline 0.61) по той же логике — pi0_fast_full_droid_finetune.
# Именно на этой паре у нас на win2 политика молчала (action_horizon=16 + max_token_len=180);
# здесь проверяем, воспроизводится ли это на стоке.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd /workspace/openpi
CONFIG="${1:-pi0_fast_full_droid_finetune}"
CKPT="${2:-gs://openpi-assets/checkpoints/pi0_fast_droid_jointpos}"
echo "конфиг=$CONFIG  чекпоинт=$CKPT"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config="$CONFIG" \
    --policy.dir="$CKPT"
