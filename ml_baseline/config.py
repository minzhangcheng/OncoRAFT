"""
Centralized Configuration for ML Baseline
==========================================
All paths, model definitions, hyperparameters, and constants in one place.
Supports environment variable overrides for portability.
"""

import os
from pathlib import Path

# ============================================================================
# Directory Paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent

# MSK data
MSK_DATA_DIR = os.environ.get('MSK_DATA_DIR', '')
JSONL_PATH = os.environ.get('JSONL_PATH', '')

# Output
RESULTS_DIR = os.environ.get('RESULTS_DIR', str(BASE_DIR / 'results'))
MODEL_DIR = os.path.join(RESULTS_DIR, 'saved_models')

# TCGA data
TCGA_DIR = Path(os.environ.get('TCGA_DIR', ''))
TCGA_CLINICAL_FILE = TCGA_DIR / 'data' / 'clinical' / 'all_projects_clinical.csv'
TCGA_TREATMENT_PLANS_FILE = TCGA_DIR / 'data' / 'treatment_plans' / 'all_treatment_plans.json'
TCGA_MUTATION_FILE = TCGA_DIR / 'data' / 'mutations_cbioportal' / 'mc3_mutations.tsv'
TCGA_SURVIVAL_FILE = TCGA_DIR / 'data' / 'survival' / 'TCGA-CDR.xlsx'
TCGA_CBIOPORTAL_DIR = TCGA_DIR / 'data' / 'cbioportal_data'
TCGA_METADATA_FILE = TCGA_DIR / 'outputs' / 'prompts' / 'tcga_prompts_metadata.csv'
TCGA_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'tcga_validation')

# ============================================================================
# Feature Configuration
# ============================================================================
TOP_GENES = [
    'TP53', 'KRAS', 'PIK3CA', 'APC', 'EGFR', 'BRAF', 'PTEN', 'ATM', 'BRCA2', 'BRCA1',
    'CDKN2A', 'RB1', 'NF1', 'STK11', 'KEAP1', 'SMAD4', 'ARID1A', 'KMT2D', 'NOTCH1', 'ERBB2',
    'MYC', 'CCND1', 'CDK4', 'MDM2', 'FGFR1', 'FGFR2', 'FGFR3', 'MET', 'ALK', 'ROS1',
    'RET', 'NTRK1', 'NTRK2', 'NTRK3', 'ESR1', 'AR', 'GATA3', 'FOXA1', 'MAP2K1', 'MAP2K2',
    'NRAS', 'HRAS', 'RAF1', 'AKT1', 'MTOR', 'TSC1', 'TSC2', 'FBXW7', 'CTNNB1', 'KIT',
]

NONSYNONYMOUS_EFFECTS = {
    'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
    'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins',
    'Splice_Site', 'Translation_Start_Site', 'Nonstop_Mutation',
    'missense_variant', 'stop_gained', 'stop_lost',
    'frameshift_variant', 'inframe_insertion', 'inframe_deletion',
    'splice_acceptor_variant', 'splice_donor_variant',
    'start_lost', 'protein_altering_variant',
}

# Cancer type mapping for MSK training (Diagnosis Description field)
CANCER_TYPES = {
    'Breast': ['breast', 'mammary'],
    'NSCLC': ['lung', 'bronchus', 'pulmonary', 'nsclc', 'adenocarcinoma of lung'],
    'Colorectal': ['colorectal', 'colon', 'rectal', 'rectum', 'cecum', 'sigmoid'],
    'Pancreatic': ['pancrea', 'pancreatic'],
    'Prostate': ['prostate'],
}

# Features excluded for fair comparison with OncoRAFT
ONCORAFT_EXCLUDED_FEATURES = [
    'tumor_bone', 'tumor_liver', 'tumor_lung', 'tumor_brain', 'tumor_lymph',
    'pdl1_positive_history',
    'tumor_purity',
]

STRICT_ONCORAFT_CONSISTENT_MODE = True

# ============================================================================
# TCGA-specific Mappings
# ============================================================================
PROJECT_TO_CANCER = {
    'TCGA-LUAD': 'Lung Cancer', 'TCGA-LUSC': 'Lung Cancer',
    'TCGA-BRCA': 'Breast Cancer',
    'TCGA-COAD': 'Colon Cancer', 'TCGA-READ': 'Colon Cancer',
    'TCGA-PRAD': 'Prostate Cancer',
    'TCGA-PAAD': 'Pancreatic Cancer',
}

PROJECT_TO_FEATURE_CANCER = {
    'TCGA-LUAD': 'nsclc', 'TCGA-LUSC': 'nsclc',
    'TCGA-BRCA': 'breast',
    'TCGA-COAD': 'colorectal', 'TCGA-READ': 'colorectal',
    'TCGA-PRAD': 'prostate',
    'TCGA-PAAD': 'pancreatic',
}

SUBTYPE_MAP = {
    'BRCA_LumA':   {'hr_positive': 1, 'her2_positive': 0},
    'BRCA_LumB':   {'hr_positive': 1, 'her2_positive': 0},
    'BRCA_Her2':   {'hr_positive': 0, 'her2_positive': 1},
    'BRCA_Basal':  {'hr_positive': 0, 'her2_positive': 0},
    'BRCA_Normal': {'hr_positive': 0, 'her2_positive': 0},
}

# ============================================================================
# Cross-Validation Configuration
# ============================================================================
N_SPLITS = 5
RANDOM_STATE = 42

# ============================================================================
# Model Definitions
# ============================================================================
# Models that need StandardScaler
MODELS_NEEDING_SCALING = {'LogisticRegression', 'MLP', 'SVM'}


def get_models(random_state=RANDOM_STATE):
    """Return dict of {name: model_instance} for all configured baseline models.

    Import here to keep config lightweight (no heavy imports at module level).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        AdaBoostClassifier, ExtraTreesClassifier,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=random_state,
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=500,
            random_state=random_state,
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=random_state,
        ),
    }

    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,  # updated per-fold
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
        )
    except ImportError:
        print("Warning: XGBoost not installed, will skip XGBoost model")

    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=random_state,
            verbose=-1,
        )
    except ImportError:
        print("Warning: LightGBM not installed, will skip LightGBM model")

    try:
        from catboost import CatBoostClassifier
        models['CatBoost'] = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            auto_class_weights='Balanced',
            random_seed=random_state,
            verbose=0,
        )
    except ImportError:
        print("Warning: CatBoost not installed, will skip CatBoost model")

    return models


# Canonical model name list (order for display/CSV)
MODEL_NAMES = [
    'LogisticRegression', 'RandomForest', 'ExtraTrees', 'AdaBoost',
    'GradientBoosting', 'MLP', 'SVM', 'XGBoost', 'LightGBM', 'CatBoost',
]

