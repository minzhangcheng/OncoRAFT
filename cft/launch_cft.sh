#!/bin/bash
# CFT (Canonical Fine-Tuning) — 5-fold patient-level CV training.
set -e

SCRIPTS_DIR="${CFT_SCRIPTS_DIR:-$(dirname "$0")/scripts}"
CHECKPOINT_DIR="${CFT_CHECKPOINT_DIR:?must be set}"
DATA_FILE="${CFT_DATA_FILE:?must be set}"
LOG_FILE="${CFT_LOG_DIR:-$(pwd)}/cft_training.log"

exec >> "$LOG_FILE" 2>&1

echo ""
echo "============================================================"
echo "$(date): CFT — Canonical Fine-Tuning"
echo "============================================================"

source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pytorch}"

echo "$(date): Environment ready"
echo "  Python: $(which python)"
echo "  Data: $DATA_FILE ($(wc -l < $DATA_FILE) lines)"
echo "  Output: $CHECKPOINT_DIR"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

for FOLD in 0 1 2 3 4; do
    echo ""
    echo "============================================================"
    echo "$(date): ===== Training Fold $FOLD / 4 ====="
    echo "============================================================"
    cd "$SCRIPTS_DIR"
    accelerate launch train_canonical.py \
        --fold $FOLD \
        --output_dir "$CHECKPOINT_DIR" \
        2>&1
    echo "$(date): Fold $FOLD complete!"
done

echo ""
echo "============================================================"
echo "$(date): All 5 folds complete!"
echo "============================================================"
cat "$CHECKPOINT_DIR/5fold_summary.json" 2>/dev/null || echo "Summary not generated yet"
echo "$(date): Done!"
