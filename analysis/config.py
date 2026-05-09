"""
Shared paths for analysis scripts. All values come from environment variables;
override per-deployment via your shell rc or a .env file.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# MSK clinical sample file (used by multigroup_survival for cancer-type mapping)
CLINICAL_SAMPLE_FILE = os.environ.get('MSK_CLINICAL_SAMPLE_FILE', '')

# Training JSONL with reasoning text + structured input
TRAINING_DATA = os.environ.get('ONCORAFT_TRAINING_DATA', '')

# OncoRAFT 5-fold multitask checkpoint root
ONCORAFT_CHECKPOINT_DIR = os.environ.get('ONCORAFT_CHECKPOINT_DIR', '')

# Generated text from inference (per-sample reasoning + score)
GENERATED_JSONL = os.environ.get('ONCORAFT_GENERATED_JSONL', '')

# Feature matrix and per-sample mapping
FEATURE_MATRIX_CSV = os.environ.get('ONCORAFT_FEATURE_MATRIX', '')
RESPONSE_ARRAY_CSV = os.environ.get('ONCORAFT_RESPONSE_ARRAY', '')
ONCORAFT_SCORES_CSV = os.environ.get('ONCORAFT_SCORES_CSV', '')

# Default output directory
OUTPUT_DIR = os.environ.get('ANALYSIS_OUTPUT_DIR', str(BASE_DIR / 'results'))
