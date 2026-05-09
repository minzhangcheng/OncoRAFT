#!/bin/bash
# Train ML baseline models with 5-fold CV on MSK data
#
# Required env vars:
#   MSK_DATA_DIR  — directory with MSK-CHORD raw TSV files
#   JSONL_PATH    — instruction JSONL with reasoning text + scores
#   RESULTS_DIR   — output directory for predictions and saved models
set -e

: "${MSK_DATA_DIR:?MSK_DATA_DIR must be set}"
: "${JSONL_PATH:?JSONL_PATH must be set}"
: "${RESULTS_DIR:?RESULTS_DIR must be set}"

LOG_FILE="${ML_BASELINE_LOG:-$RESULTS_DIR/train.log}"
mkdir -p "$RESULTS_DIR"
exec >> "$LOG_FILE" 2>&1

echo ""
echo "============================================================"
echo "$(date): Starting ML baseline training (5-fold CV)"
echo "============================================================"
echo "  MSK data:    $MSK_DATA_DIR"
echo "  JSONL:       $JSONL_PATH"
echo "  Output:      $RESULTS_DIR"
echo ""

cd "$(dirname "$0")"
python -u train_5fold.py 2>&1

echo ""
echo "$(date): Training complete!"
