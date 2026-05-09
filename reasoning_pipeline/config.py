"""
Centralized configuration for the OncoRAFT reasoning data pipeline.

All file paths, API settings, and shared constants are defined here.
To reproduce the pipeline on a new machine, update INPUT_DIR / OUTPUT_DIR.
"""
import os

# ============================================================
# Directories
#   INPUT_DIR  — original MSK-CHORD data (read-only)
#   OUTPUT_DIR — all pipeline outputs (never overwrites input)
# ============================================================
INPUT_DIR = os.environ.get("ONCORAFT_DATA_DIR", "")
OUTPUT_DIR = os.environ.get("ONCORAFT_OUTPUT_DIR", "")
if OUTPUT_DIR:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Legacy alias used by step5_vllm for output file naming
BASE_DIR = OUTPUT_DIR

# ============================================================
# Raw data inputs (read from INPUT_DIR)
# ============================================================
PATIENT_FILE = os.path.join(INPUT_DIR, "data_clinical_patient.txt")
SAMPLE_FILE = os.path.join(INPUT_DIR, "data_clinical_sample.txt")
MUTATION_FILE = os.path.join(INPUT_DIR, "data_mutations.txt")
SV_FILE = os.path.join(INPUT_DIR, "data_sv.txt")
DIAGNOSIS_FILE = os.path.join(INPUT_DIR, "data_timeline_diagnosis.txt")
TREATMENT_FILE = os.path.join(INPUT_DIR, "data_timeline_treatment.txt")
PRIOR_MEDS_FILE = os.path.join(INPUT_DIR, "data_timeline_prior_meds.txt")
SPECIMEN_SURGERY_FILE = os.path.join(INPUT_DIR, "data_timeline_specimen_surgery.txt")
PDL1_FILE = os.path.join(INPUT_DIR, "data_timeline_pdl1.txt")
MMR_FILE = os.path.join(INPUT_DIR, "data_timeline_mmr.txt")
GLEASON_FILE = os.path.join(INPUT_DIR, "data_timeline_gleason.txt")
CA_15_3_FILE = os.path.join(INPUT_DIR, "data_timeline_ca_15-3_labs.txt")
CA_19_9_FILE = os.path.join(INPUT_DIR, "data_timeline_ca_19-9_labs.txt")
CEA_FILE = os.path.join(INPUT_DIR, "data_timeline_cea_labs.txt")
TUMOR_SITES_FILE = os.path.join(INPUT_DIR, "data_timeline_tumor_sites.txt")
PROGRESSION_FILE = os.path.join(INPUT_DIR, "data_timeline_progression.txt")
DRUGBANK_FILE = os.path.join(INPUT_DIR, "..", "drugbank_data.json")

# ============================================================
# Step 0a/0b: Existing intermediate data (read from INPUT_DIR)
# ============================================================
SEGMENTATION_PROMPTS_DIR = os.path.join(INPUT_DIR, "patient_prompts")
TREATMENT_PLANS_DIR = os.path.join(INPUT_DIR, "patient_treatments_api")
TREATMENT_REGIMENS_DIR = os.path.join(INPUT_DIR, "patient_treatments_api_responses")

# ============================================================
# Step 0c: RECIST response assessment
#   DRUG_RESPONSE_FILE — read existing labels from INPUT_DIR
#   RECIST_OUTPUT      — write new labels to OUTPUT_DIR
# ============================================================
DRUG_RESPONSE_FILE = os.path.join(INPUT_DIR, "Recist_plan_responses_adjusted_window.csv")
RECIST_OUTPUT = os.path.join(OUTPUT_DIR, "Recist_plan_responses_adjusted_window.csv")

# ============================================================
# Step 1–6: All outputs → OUTPUT_DIR
# ============================================================
RAW_PROMPTS = os.path.join(OUTPUT_DIR, "treatment_plan_prompts.jsonl")
CLEANED_PROMPTS = os.path.join(OUTPUT_DIR, "treatment_plan_prompts_cleaned.jsonl")
INSTRUCTION_PROMPTS = os.path.join(OUTPUT_DIR, "instruction_tuning_prompts.jsonl")
BATCH_PROMPTS = os.path.join(OUTPUT_DIR, "batch_prompts_for_reasoning.jsonl")
REASONING_DATA = os.path.join(OUTPUT_DIR, "instruction_data_with_reasoning.jsonl")
API_OUTPUTS = os.path.join(OUTPUT_DIR, "api_reasoning_outputs.jsonl")
FINAL_TRAINING_DATA = os.path.join(OUTPUT_DIR, "instruction_data_with_reasoning_postprocessed.jsonl")

# ============================================================
# API Configuration
# ============================================================
# DeepSeek official API (used for both segmentation and reasoning)
API_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE")
API_MODEL_SEGMENTATION = "deepseek-chat"  # for step 0b segmentation
API_MODEL_REASONING = "deepseek-chat"     # for step 5 reasoning

# Generation parameters
SEGMENTATION_TEMPERATURE = 0.3
SEGMENTATION_TOP_P = 0.9
REASONING_TEMPERATURE = 0.6
REASONING_TOP_P = 0.9

# Concurrency
MAX_WORKERS = 128
RATE_LIMIT_DELAY = 0.1  # seconds between API calls

# ============================================================
# System instruction for final training data
# ============================================================
SYSTEM_INSTRUCTION = """You are an expert AI assistant for precision oncology.

Drug Response Inference Task
Based on the patient's genetic profile, clinical and diagnosis Information, and the drug's mechanism of action, please provide:

1. Output a score representing the likelihood of positive treatment response (a number between 0 and 1).
 - Scores closer to 1 indicate higher likelihood of complete response(positive)
 - Scores closer to 0 indicate higher likelihood of disease progression(negative)
2. Key genetic factors influencing this prediction and why they matter for this treatment plan
3. Critical clinical determinants that significantly impact the efficacy of this treatment"""
