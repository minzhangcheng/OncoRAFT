#!/bin/bash
# OncoRAFT 5-fold CV training (multitask: LM + score head, LoRA).
#
# Defaults to Llama 3.1 8B Instruct as the base model. Set ONCORAFT_BASE_MODEL
# to swap in a different base.
#
# Required env vars:
#   ONCORAFT_BASE_MODEL       — path to base LLM (e.g. /path/to/llama3.1_8B_instruct)
#   ONCORAFT_TRAINING_DATA    — instruction JSONL with reasoning text + scores
#   ONCORAFT_CHECKPOINT_DIR   — output directory for fold checkpoints
set -e

: "${ONCORAFT_BASE_MODEL:?must be set}"
: "${ONCORAFT_TRAINING_DATA:?must be set}"
: "${ONCORAFT_CHECKPOINT_DIR:?must be set}"

LOG_FILE="${ONCORAFT_LOG_DIR:-$(pwd)}/train.log"
TRAINING_DIR="${ONCORAFT_TRAINING_DIR:-$(dirname "$0")}"

exec >> "$LOG_FILE" 2>&1

source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pytorch}"

echo ""
echo "============================================================"
echo "$(date): Starting OncoRAFT 5-fold CV training"
echo "============================================================"
echo "  Base model:  $ONCORAFT_BASE_MODEL"
echo "  Data:        $ONCORAFT_TRAINING_DATA"
echo "  Output:      $ONCORAFT_CHECKPOINT_DIR"
echo ""

nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

cd "$TRAINING_DIR"

for FOLD in 0 1 2 3 4; do
    echo ""
    echo "============================================================"
    echo "$(date): Training Fold $FOLD / 4"
    echo "============================================================"
    accelerate launch --multi_gpu --num_processes 8 \
        train_5fold.py --fold $FOLD \
        2>&1
    echo "$(date): Fold $FOLD complete!"
done

echo ""
echo "============================================================"
echo "$(date): All 5 folds complete!"
echo "============================================================"
