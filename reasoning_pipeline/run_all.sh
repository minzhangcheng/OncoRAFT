#!/bin/bash
# OncoRAFT Reasoning Pipeline - Full execution script
#
# Prerequisites:
#   export DEEPSEEK_API_KEY=your_key
#   export ONCORAFT_DATA_DIR=/path/to/msk/data
#   export ONCORAFT_OUTPUT_DIR=/path/to/pipeline/output
#
# Pipeline overview:
#   Step 0a: Generate segmentation prompts (diagnosis + treatment -> per-patient prompt files)
#   Step 0b: API segmentation (DeepSeek-chat -> treatment plan JSON per patient)
#   Step 0c: RECIST response assessment (tumor sites + progression -> CR/PR/SD/PD labels)
#   Step 1:  Generate prompts (drug classification + clinical features + RECIST score -> structured prompts)
#   Step 2:  Clean prompts (remove nan, duplicates, invalid groups)
#   Step 3:  Convert to instruction format (instruction/input/output)
#   Step 4:  Generate batch prompts for reasoning
#   Step 5:  Generate reasoning (API or vLLM)
#   Step 6:  Post-process (only needed for API path)

set -e

echo "=== OncoRAFT Reasoning Pipeline ==="
echo "Data dir: ${ONCORAFT_DATA_DIR:?ONCORAFT_DATA_DIR must be set}"
echo ""

# Step 0a: Generate segmentation prompts
echo "[Step 0a] Generating segmentation prompts..."
python step0a_generate_segmentation_prompts.py

# Step 0b: API-based treatment segmentation
echo "[Step 0b] Running API segmentation..."
python step0b_api_segmentation.py

# Step 0c: RECIST response assessment
echo "[Step 0c] Assessing RECIST responses..."
python step0c_recist_response.py

# Step 1: Generate drug response prompts
echo "[Step 1] Generating drug response prompts..."
python step1_generate_prompts.py

# Step 2: Clean prompts
echo "[Step 2] Cleaning prompts..."
python step2_clean_prompts.py

# Step 3: Convert to instruction format
echo "[Step 3] Converting to instruction format..."
python step3_convert_to_instruction.py

# Step 4: Generate batch prompts
echo "[Step 4] Generating batch prompts..."
python step4_generate_batch_prompts.py

# Step 5 + 6: Generate reasoning (choose ONE of the following)
# Option A: DeepSeek API (then post-process with step6)
echo "[Step 5] Generating reasoning via DeepSeek API..."
python step5_api_reasoning.py
echo "[Step 6] Post-processing..."
python step6_postprocess.py

# Option B: Local vLLM (already includes post-processing, skip step6)
# python step5_vllm_reasoning.py --model /path/to/model --tag model-name --tp 4

echo ""
echo "=== Pipeline complete ==="
