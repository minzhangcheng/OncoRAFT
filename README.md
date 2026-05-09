# OncoRAFT: Oncology Reasoning-Aligned Fine-Tuning

OncoRAFT is a framework for predicting drug treatment response in oncology.
See the accompanying manuscript for datasets, metrics, and benchmarking details.

![OncoRAFT overview](Figure%201.png)

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Environment Variables

All paths are configurable via environment variables; there are **no hardcoded
absolute paths**. Below is the canonical set used across modules. Set the ones
you need for the script you are running.

```bash
# === Data ===
export ONCORAFT_DATA_DIR=/path/to/msk_chord_raw          # MSK-CHORD raw TSVs
export ONCORAFT_OUTPUT_DIR=/path/to/pipeline_output      # reasoning_pipeline outputs
export ONCORAFT_TRAINING_DATA=/path/to/instruction.jsonl # training JSONL
export MSK_CLINICAL_SAMPLE_FILE=/path/to/data_clinical_sample.txt

# === Models ===
export ONCORAFT_BASE_MODEL=/path/to/llama3.1_8B_instruct
export ONCORAFT_CHECKPOINT_DIR=/path/to/multitask_5fold  # LoRA + score-head checkpoints
export LLM_DIR=/path/to/all_LLMs                         # parent dir for zero_shot registry

# === External validation ===
export ONCORAFT_VALIDATION_DIR=/path/to/tcga_validation
export ONCORAFT_DRUGBANK=/path/to/drugbank_data.json

# === Figures / analysis ===
export ONCORAFT_FEATURE_MATRIX=/path/to/msk_feature_matrix.csv
export ONCORAFT_RESPONSE_ARRAY=/path/to/score_response_mapping.csv
export ONCORAFT_SCORES_CSV=/path/to/oncoraft_msk_scores.csv
export ONCORAFT_GENERATED_JSONL=/path/to/msk_all_generated.jsonl
export ONCORAFT_TCGA_PREDICTIONS=/path/to/tcga_oncoraft_predictions.csv
export ML_BASELINE_TCGA_PREDICTIONS=/path/to/tcga_ml_predictions.csv
export ANALYSIS_OUTPUT_DIR=/path/to/analysis/results

# === Reasoning API (data construction) ===
export DEEPSEEK_API_KEY=your_api_key
```

Each `config.py` (in `reasoning_pipeline/`, `training/`, `external_validation/`,
`ml_baseline/`, `analysis/`, `cft/scripts/`, `zero_shot/`) lists the env
vars its module reads. Empty strings are used as fallback defaults so missing
configuration fails fast at the call site rather than silently using a wrong
path.

## Pipeline Overview

### 1. Training data construction (`reasoning_pipeline/`)
Processes MSK-CHORD clinical data through a 7-step pipeline:
- Patient segmentation and RECIST response assessment (DeepSeek API)
- Structured prompt generation with clinical context
- Reasoning text generation (DeepSeek API or local vLLM)
- Post-processing and quality filtering

### 2. Model training (`training/`)
5-fold cross-validation with multitask learning:
- **LM Loss**: standard causal language modeling on reasoning text
- **MSE Loss**: score head regression on RECIST-derived response scores
- LoRA fine-tuning on attention + MLP projections

### 3. ML baselines (`ml_baseline/`)
Traditional ML classifiers trained on the same structured feature matrix
for fair comparison.

### 4. External validation (`external_validation/`)
Independent validation on TCGA.

### 5. Canonical Fine-Tuning (`cft/`)
- **CFT (Canonical Fine-Tuning)**: standard causal LM fine-tuning with LoRA
  on the same training JSONL and 5-fold CV split, with the score parsed
  from the generated text at inference time.

### 6. Zero-shot inference (`zero_shot/`)
Benchmarking pretrained LLMs without any fine-tuning.

### 7. Analysis (`analysis/`)
SHAP attributions on a surrogate Random Forest, GSEA (per cancer type and
pan-cancer), reasoning-text word clouds, multi-group survival analysis,
and counterfactual experiments (clinical/biomarker perturbations and
gene-masking on canonical gene-drug pairs).

## Hardware Requirements

- **Training**: 8× NVIDIA A100 80GB (distributed via Accelerate)
- **Inference**: a single GPU with ≥ 24 GB VRAM (e.g., RTX 3090 / 4090)
- **ML Baselines**: CPU only

## Model weights

Direct release of the fine-tuned weights would require formal authorization
from Memorial Sloan Kettering under the MSK-CHORD data-use agreement. We
have raised this with MSK; absent such authorization, we are not at liberty
to release the weights ourselves. This does not affect end-to-end
reproducibility: the MSK-CHORD dataset is publicly available through
cBioPortal, and the complete training and inference pipeline in this
repository allows OncoRAFT to be retrained from scratch using public data
and public code.

## License

Apache License 2.0


## Citation

If you use this code, please cite our paper (forthcoming).

