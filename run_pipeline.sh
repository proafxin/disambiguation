#!/usr/bin/env bash
# run_pipeline.sh — train stage2 then launch tensorboard on the latest run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TB_DIR="$SCRIPT_DIR/cache/tensorboard"
TB_PORT="${TB_PORT:-6006}"

echo "=== Stage 2 Training ==="
uv run python disambiguation/signals/train_stage2.py "$@"

echo ""
echo "=== Launching TensorBoard ==="
echo "Log dir : $TB_DIR"
echo "URL     : http://localhost:$TB_PORT"
echo "(Ctrl-C to stop)"
uv run tensorboard --logdir "$TB_DIR" --port "$TB_PORT" --bind_all
