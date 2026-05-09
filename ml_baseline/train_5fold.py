"""
Traditional ML Baseline for OncoRAFT - 5-Fold CV Training
==========================================================
Extract structured features from MSK-CHORD raw data and train 10 ML models.

Target: Drug regimen response (RECIST-based)
- 0.0 = PD (Progressive Disease)
- 0.3 = SD (Stable Disease)
- 0.7 = PR (Partial Response)
- 1.0 = CR (Complete Response)

Binary classification: Positive (>=0.5) vs Negative (<0.5)

Models (10): LR, RF, ExtraTrees, AdaBoost, GB, MLP, SVM, XGBoost, LightGBM, CatBoost
"""

import os
import sys
import json
import re
import copy
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score, confusion_matrix,
)
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import (
    MSK_DATA_DIR, JSONL_PATH, RESULTS_DIR, MODEL_DIR,
    TOP_GENES, CANCER_TYPES, ONCORAFT_EXCLUDED_FEATURES,
    STRICT_ONCORAFT_CONSISTENT_MODE, N_SPLITS, RANDOM_STATE,
    MODELS_NEEDING_SCALING, get_models,
)
from drug_utils import extract_drug_features, normalize_drug, DRUG_CLASSES

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# Step 1: Load Labels from JSONL
# ============================================================================
def load_labels_from_jsonl(jsonl_path):
    """Extract Patient ID, Drug regimen, and RECIST score from JSONL file."""
    print("Loading labels from JSONL...")
    samples = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            inp = data['input']
            out = data['output']

            patient_match = re.search(r'Patient ID: ([^\n]+)', inp)
            patient_id = patient_match.group(1).strip() if patient_match else None

            drug_match = re.search(r'Drug\(s\): ([^\n]+)', inp)
            drugs = drug_match.group(1).strip() if drug_match else None

            score_match = re.search(r'^([\d.]+)', out)
            score = float(score_match.group(1)) if score_match else None

            sample_match = re.search(r'Sample ID: ([^\s]+)', inp)
            sample_id = sample_match.group(1).strip() if sample_match else None

            derived_stage_match = re.search(r'Derived Stage: ([^\n]+)', inp)
            derived_stage = derived_stage_match.group(1).strip() if derived_stage_match else None

            diagnosis_match = re.search(r'Diagnosis Description: ([^\n]+)', inp)
            cancer_raw = diagnosis_match.group(1).strip() if diagnosis_match else None

            cancer_type = 'Other'
            if cancer_raw:
                cancer_raw_lower = cancer_raw.lower()
                for ctype, keywords in CANCER_TYPES.items():
                    if any(kw in cancer_raw_lower for kw in keywords):
                        cancer_type = ctype
                        break

            if patient_id and drugs and score is not None:
                samples.append({
                    'idx': idx,
                    'patient_id': patient_id,
                    'sample_id': sample_id,
                    'drugs': drugs,
                    'drug_list': [d.strip() for d in drugs.split(',')],
                    'score': score,
                    'binary_label': 1 if score >= 0.5 else 0,
                    'cancer_type': cancer_type,
                    'cancer_raw': cancer_raw,
                    'derived_stage': derived_stage,
                })

    print(f"  Loaded {len(samples)} samples with labels")

    df = pd.DataFrame(samples)
    print(f"\n  Cancer type distribution:")
    for ctype, count in df['cancer_type'].value_counts().items():
        print(f"    {ctype}: {count}")

    if 'Other' in df['cancer_type'].values:
        other_count = len(df[df['cancer_type'] == 'Other'])
        if other_count > 100:
            print(f"\n  WARNING: {other_count} samples classified as 'Other'")
            print(f"  Top 10 'Other' cancer_raw values:")
            other_raw = df[df['cancer_type'] == 'Other']['cancer_raw'].value_counts().head(10)
            for raw_type, cnt in other_raw.items():
                print(f"    {raw_type}: {cnt}")

    return df


# ============================================================================
# Step 2: Load MSK Raw Data
# ============================================================================
def load_msk_data(data_dir):
    """Load all relevant MSK-CHORD data files."""
    print("\nLoading MSK raw data...")

    def read_tsv(filename):
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, sep='\t', comment='#', low_memory=False)
            print(f"  {filename}: {len(df)} rows")
            return df
        return None

    data = {
        'patient': read_tsv('data_clinical_patient.txt'),
        'sample': read_tsv('data_clinical_sample.txt'),
        'mutations': read_tsv('data_mutations.txt'),
        'sv': read_tsv('data_sv.txt'),
        'treatment': read_tsv('data_timeline_treatment.txt'),
        'progression': read_tsv('data_timeline_progression.txt'),
        'pdl1': read_tsv('data_timeline_pdl1.txt'),
        'cea': read_tsv('data_timeline_cea_labs.txt'),
        'ca153': read_tsv('data_timeline_ca_15-3_labs.txt'),
        'ca199': read_tsv('data_timeline_ca_19-9_labs.txt'),
        'psa': read_tsv('data_timeline_psa_labs.txt'),
    }

    return data


# ============================================================================
# Step 3: Extract Features
# ============================================================================
def extract_patient_features(msk_data):
    """Extract patient-level features from MSK data."""
    print("\nExtracting patient features...")

    if STRICT_ONCORAFT_CONSISTENT_MODE:
        print("  STRICT_ONCORAFT_CONSISTENT_MODE enabled - excluding features OncoRAFT doesn't use")

    patient_df = msk_data['patient']
    if patient_df is None:
        return {}

    features = {}

    for _, row in patient_df.iterrows():
        pid = row.get('PATIENT_ID')
        if pd.isna(pid):
            continue

        features[pid] = {
            'gender': 1 if row.get('GENDER') == 'Male' else 0,
            'age': row.get('CURRENT_AGE_DEID', np.nan),
            'race_white': 1 if 'White' in str(row.get('RACE', '')) else 0,
            'race_black': 1 if 'Black' in str(row.get('RACE', '')) else 0,
            'race_asian': 1 if 'Asian' in str(row.get('RACE', '')) else 0,
            'ethnicity_hispanic': 1 if 'Spanish' in str(row.get('ETHNICITY', '')) or 'Hispanic' in str(row.get('ETHNICITY', '')) else 0,
            'smoking_current_former': 1 if 'Former' in str(row.get('SMOKING_PREDICTIONS_3_CLASSES', '')) or 'Current' in str(row.get('SMOKING_PREDICTIONS_3_CLASSES', '')) else 0,
            'smoking_never': 1 if 'Never' in str(row.get('SMOKING_PREDICTIONS_3_CLASSES', '')) else 0,
            'stage_4': 1 if 'Stage 4' in str(row.get('STAGE_HIGHEST_RECORDED', '')) else 0,
            'prior_treatment': 1 if 'Prior' in str(row.get('PRIOR_MED_TO_MSK', '')) else 0,
            'hr_positive': 1 if row.get('HR') == 'Yes' else 0,
            'her2_positive': 1 if row.get('HER2') == 'Yes' else 0,
            '_os_months': row.get('OS_MONTHS', np.nan),
            '_os_status': 1 if '1:DECEASED' in str(row.get('OS_STATUS', '')) else 0,
        }

    print(f"  Extracted features for {len(features)} patients")
    return features


def extract_sample_features(msk_data):
    """Extract sample-level features (genomic) from MSK data."""
    print("\nExtracting sample features...")

    sample_df = msk_data['sample']
    if sample_df is None:
        return {}

    features = {}

    for _, row in sample_df.iterrows():
        pid = row.get('PATIENT_ID')
        sid = row.get('SAMPLE_ID')
        if pd.isna(pid) or pd.isna(sid):
            continue

        if pid not in features:
            cancer_type_str = str(row.get('CANCER_TYPE', ''))
            features[pid] = {
                'sample_id': sid,
                'cancer_nsclc': 1 if any(kw.lower() in cancer_type_str.lower()
                                          for kw in ['Non-Small Cell Lung', 'Lung Adenocarcinoma',
                                                     'Lung Squamous', 'NSCLC']) else 0,
                'cancer_breast': 1 if 'Breast' in cancer_type_str else 0,
                'cancer_colorectal': 1 if any(kw in cancer_type_str
                                               for kw in ['Colorectal', 'Colon', 'Rectal']) else 0,
                'cancer_pancreatic': 1 if 'Pancrea' in cancer_type_str else 0,
                'cancer_prostate': 1 if 'Prostate' in cancer_type_str else 0,
                'is_metastasis': 1 if row.get('SAMPLE_TYPE') == 'Metastasis' else 0,
                'msi_stable': 1 if row.get('MSI_TYPE') == 'Stable' else 0,
                'msi_instable': 1 if row.get('MSI_TYPE') == 'Instable' else 0,
                'msi_score': row.get('MSI_SCORE', np.nan) if pd.notna(row.get('MSI_SCORE')) and row.get('MSI_SCORE', -1) >= 0 else np.nan,
                'tmb': row.get('TMB_NONSYNONYMOUS', np.nan),
            }

    print(f"  Extracted features for {len(features)} patients")
    return features


def extract_mutation_features(msk_data, top_genes=TOP_GENES):
    """Extract mutation features - binary indicators for top mutated genes."""
    print("\nExtracting mutation features...")

    mutations_df = msk_data['mutations']
    if mutations_df is None:
        return {}

    patient_mutations = defaultdict(set)

    for _, row in mutations_df.iterrows():
        sample_id = row.get('Tumor_Sample_Barcode', '')
        gene = row.get('Hugo_Symbol', '')

        if pd.isna(sample_id) or pd.isna(gene):
            continue

        patient_id = '-'.join(sample_id.split('-')[:2]) if '-' in sample_id else sample_id
        patient_mutations[patient_id].add(gene)

    features = {}
    for pid, genes in patient_mutations.items():
        features[pid] = {
            f'mut_{gene}': 1 if gene in genes else 0
            for gene in top_genes
        }
        features[pid]['mutation_count'] = len(genes)
        features[pid]['has_any_mutation'] = 1 if len(genes) > 0 else 0

    print(f"  Extracted mutation features for {len(features)} patients")
    return features


def load_treatment_timeline(data_dir):
    """Load MSK treatment timeline and normalize agent names."""
    filepath = os.path.join(data_dir, 'data_timeline_treatment.txt')
    if not os.path.exists(filepath):
        print(f"  WARNING: Treatment timeline not found: {filepath}")
        return {}

    df = pd.read_csv(filepath, sep='\t', comment='#', low_memory=False)
    print(f"  Treatment timeline: {len(df)} rows")

    timeline = defaultdict(list)
    for _, row in df.iterrows():
        pid = row.get('PATIENT_ID')
        agent = row.get('AGENT')
        start = row.get('START_DATE')
        if pd.isna(pid) or pd.isna(agent) or pd.isna(start):
            continue
        try:
            start_days = float(start)
        except (ValueError, TypeError):
            continue
        norm_agent = normalize_drug(str(agent).strip())
        timeline[str(pid)].append((start_days, norm_agent))

    print(f"  Timeline loaded for {len(timeline)} patients")
    return dict(timeline)


def find_regimen_start(patient_id, drug_list, timeline_dict, tolerance_days=14):
    """Find regimen start date by matching drugs in treatment timeline."""
    entries = timeline_dict.get(str(patient_id))
    if not entries:
        return None

    norm_drugs = set()
    for d in drug_list:
        nd = normalize_drug(d.strip())
        if nd:
            norm_drugs.add(nd.upper())

    if not norm_drugs:
        return None

    matched = [(sd, agent) for sd, agent in entries if agent.upper() in norm_drugs]
    if not matched:
        return None

    matched.sort(key=lambda x: x[0])
    clusters = []
    current_cluster = [matched[0]]

    for i in range(1, len(matched)):
        if matched[i][0] - current_cluster[-1][0] <= tolerance_days:
            current_cluster.append(matched[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [matched[i]]
    clusters.append(current_cluster)

    best_cluster = max(clusters, key=lambda c: len(set(a.upper() for _, a in c) & norm_drugs))
    return min(sd for sd, _ in best_cluster)


# ============================================================================
# Step 4: Build Feature Matrix
# ============================================================================
def build_feature_matrix(labels_df, patient_features, sample_features, mutation_features,
                         treatment_timeline=None):
    """Combine all features into a single feature matrix."""
    print("\nBuilding feature matrix...")
    if treatment_timeline is None:
        treatment_timeline = {}

    feature_rows = []

    for _, row in labels_df.iterrows():
        pid = row['patient_id']

        features = extract_drug_features(row['drug_list'])

        if pid in patient_features:
            features.update(patient_features[pid])
        else:
            features.update({
                'gender': np.nan, 'age': np.nan,
                'race_white': 0, 'race_black': 0, 'race_asian': 0,
                'ethnicity_hispanic': 0,
                'smoking_current_former': 0, 'smoking_never': 0,
                'stage_4': 0,
                'prior_treatment': 0,
                'hr_positive': 0, 'her2_positive': 0,
                '_os_months': np.nan, '_os_status': np.nan,
            })

        derived_stage = row.get('derived_stage', None)
        if derived_stage:
            features['stage_4'] = 1 if 'Stage 4' in str(derived_stage) else 0

        if pid in sample_features:
            features.update({k: v for k, v in sample_features[pid].items() if k != 'sample_id'})
        else:
            features.update({
                'cancer_nsclc': 0, 'cancer_breast': 0, 'cancer_colorectal': 0,
                'cancer_pancreatic': 0, 'cancer_prostate': 0,
                'is_metastasis': 0,
                'msi_stable': 0, 'msi_instable': 0, 'msi_score': np.nan,
                'tmb': np.nan,
            })

        if pid in mutation_features:
            features.update(mutation_features[pid])
        else:
            features.update({f'mut_{gene}': 0 for gene in TOP_GENES})
            features['mutation_count'] = 0
            features['has_any_mutation'] = 0

        features['_os_dx_months'] = features.get('_os_months', np.nan)
        features['_os_dx_status'] = features.get('_os_status', np.nan)

        start_days = find_regimen_start(pid, row['drug_list'], treatment_timeline)
        os_dx = features.get('_os_months', np.nan)
        if start_days is not None and not pd.isna(os_dx):
            os_reg = os_dx - start_days / 30.44
            features['_os_reg_months'] = os_reg if os_reg >= 0 else np.nan
        else:
            features['_os_reg_months'] = np.nan
        features['_os_reg_status'] = features.get('_os_status', np.nan)

        features['patient_id'] = pid
        features['drugs'] = row['drugs']
        features['label'] = row['binary_label']
        features['score'] = row['score']
        features['cancer_type'] = row['cancer_type']
        features['sample_idx'] = row['idx']

        feature_rows.append(features)

    df = pd.DataFrame(feature_rows)

    exclude_cols = ['patient_id', 'drugs', 'label', 'score', 'cancer_type', 'sample_idx',
                    'sample_id', '_os_months', '_os_status',
                    '_os_dx_months', '_os_dx_status', '_os_reg_months', '_os_reg_status']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if STRICT_ONCORAFT_CONSISTENT_MODE:
        original_count = len(feature_cols)
        feature_cols = [c for c in feature_cols if c not in ONCORAFT_EXCLUDED_FEATURES]
        excluded_count = original_count - len(feature_cols)
        if excluded_count > 0:
            print(f"  Excluded {excluded_count} features for OncoRAFT consistency")

    print(f"  Total samples: {len(df)}")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    return df, feature_cols


# ============================================================================
# Step 5: Train and Evaluate Models
# ============================================================================
def evaluate_predictions(y_true, y_prob, min_samples=30):
    """Calculate all evaluation metrics."""
    if len(y_true) < min_samples or len(np.unique(y_true)) < 2:
        return None

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        'n_samples': len(y_true),
        'n_positive': int(sum(y_true)),
        'n_negative': int(len(y_true) - sum(y_true)),
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['tp'] = int(tp)
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)

    return metrics


def train_and_evaluate(df, feature_cols, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """Train models using 5-fold cross-validation (patient-level split)."""
    print("\n" + "="*70)
    print("Training Traditional ML Baselines (5-Fold Cross-Validation)")
    print("="*70)

    X_raw = df[feature_cols].values
    y = df['label'].values
    patient_ids = df['patient_id'].values
    unique_patients = df['patient_id'].unique()

    models = get_models(random_state)

    all_fold_results = {name: [] for name in models}
    all_predictions = {name: np.zeros(len(df)) for name in models}
    feature_importance = {name: np.zeros(len(feature_cols)) for name in models}
    fold_assignments = np.zeros(len(df), dtype=int)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_patient_idx, val_patient_idx) in enumerate(kf.split(unique_patients)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        train_patients = set(unique_patients[train_patient_idx])
        val_patients = set(unique_patients[val_patient_idx])

        train_mask = df['patient_id'].isin(train_patients).values
        val_mask = df['patient_id'].isin(val_patients).values

        X_train_raw, X_val_raw = X_raw[train_mask], X_raw[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]

        # Impute missing values based on training set only
        train_df_fold = pd.DataFrame(X_train_raw, columns=feature_cols)
        fill_values = train_df_fold.median()

        X_train = train_df_fold.fillna(fill_values).values
        X_val = pd.DataFrame(X_val_raw, columns=feature_cols).fillna(fill_values).values

        fold_assignments[val_mask] = fold

        print(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        print(f"  Train label dist: {Counter(y_train)}, Val label dist: {Counter(y_val)}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        for name, model in models.items():
            model_fold = copy.deepcopy(model)

            # Update scale_pos_weight for XGBoost based on this fold
            if name == 'XGBoost':
                neg_count = len(y_train[y_train == 0])
                pos_count = len(y_train[y_train == 1])
                model_fold.set_params(scale_pos_weight=neg_count / pos_count if pos_count > 0 else 1.0)

            needs_scaling = name in MODELS_NEEDING_SCALING
            X_tr = X_train_scaled if needs_scaling else X_train
            X_vl = X_val_scaled if needs_scaling else X_val

            model_fold.fit(X_tr, y_train)

            # Save per-fold model
            fold_save_dir = os.path.join(MODEL_DIR, f'fold_{fold}')
            os.makedirs(fold_save_dir, exist_ok=True)
            joblib.dump(model_fold, os.path.join(fold_save_dir, f'{name}.joblib'))

            y_prob = model_fold.predict_proba(X_vl)[:, 1]

            all_predictions[name][val_mask] = y_prob

            metrics = evaluate_predictions(y_val, y_prob)
            if metrics:
                all_fold_results[name].append(metrics)
                print(f"  {name}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}, F1={metrics['f1']:.4f}")

            if hasattr(model_fold, 'feature_importances_'):
                feature_importance[name] += model_fold.feature_importances_ / n_splits
            elif hasattr(model_fold, 'coef_'):
                feature_importance[name] += np.abs(model_fold.coef_[0]) / n_splits

        # Save per-fold preprocessing artifacts
        fold_save_dir = os.path.join(MODEL_DIR, f'fold_{fold}')
        joblib.dump(fill_values, os.path.join(fold_save_dir, 'fill_values.joblib'))
        joblib.dump(scaler, os.path.join(fold_save_dir, 'scaler.joblib'))
        print(f"  Saved fold {fold} models + preprocessing to {fold_save_dir}")

    df = df.copy()
    df['fold'] = fold_assignments

    for name in models:
        df[f'pred_prob_{name}'] = all_predictions[name]

    # ========== Save final models trained on ALL data ==========
    save_dir = MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)

    full_df = pd.DataFrame(X_raw, columns=feature_cols)
    fill_values_all = full_df.median()
    X_all = full_df.fillna(fill_values_all).values

    scaler_all = StandardScaler()
    X_all_scaled = scaler_all.fit_transform(X_all)

    print("\n" + "="*70)
    print("Training & saving final models on ALL data")
    print("="*70)

    for name, model in models.items():
        model_final = copy.deepcopy(model)

        if name == 'XGBoost':
            neg_count = len(y[y == 0])
            pos_count = len(y[y == 1])
            model_final.set_params(scale_pos_weight=neg_count / pos_count if pos_count > 0 else 1.0)

        needs_scaling = name in MODELS_NEEDING_SCALING
        X_fit = X_all_scaled if needs_scaling else X_all
        model_final.fit(X_fit, y)

        model_path = os.path.join(save_dir, f'{name}.joblib')
        joblib.dump(model_final, model_path)
        print(f"  Saved {name} -> {model_path}")

    joblib.dump(scaler_all, os.path.join(save_dir, 'scaler.joblib'))
    joblib.dump(fill_values_all, os.path.join(save_dir, 'fill_values.joblib'))
    joblib.dump(list(feature_cols), os.path.join(save_dir, 'feature_cols.joblib'))
    print(f"  Saved scaler, fill_values, feature_cols -> {save_dir}")

    return df, all_fold_results, feature_importance, feature_cols, models


def compute_hr(df_subset, score_col, time_col='os_months', event_col='os_status',
               percentile=50, min_per_group=15, min_events=5):
    """Compute Cox hazard ratio for high vs low predicted score groups."""
    valid = df_subset[[score_col, time_col, event_col]].dropna()
    valid = valid[valid[time_col] > 0]
    if len(valid) < 30 or valid[event_col].sum() < min_events:
        return {'HR': np.nan, 'HR_CI_low': np.nan, 'HR_CI_high': np.nan,
                'HR_p': np.nan, 'C_index': np.nan}

    threshold = np.percentile(valid[score_col], percentile)
    valid = valid.copy()
    valid['group'] = (valid[score_col] >= threshold).astype(int)

    n_high = valid['group'].sum()
    n_low = len(valid) - n_high
    if n_high < min_per_group or n_low < min_per_group:
        return {'HR': np.nan, 'HR_CI_low': np.nan, 'HR_CI_high': np.nan,
                'HR_p': np.nan, 'C_index': np.nan}

    try:
        c_idx = concordance_index(valid[time_col], valid[score_col], valid[event_col])
    except Exception:
        c_idx = np.nan

    try:
        cph = CoxPHFitter()
        cph.fit(valid[[time_col, event_col, 'group']], duration_col=time_col, event_col=event_col)
        s = cph.summary
        return {
            'HR': s.loc['group', 'exp(coef)'],
            'HR_CI_low': s.loc['group', 'exp(coef) lower 95%'],
            'HR_CI_high': s.loc['group', 'exp(coef) upper 95%'],
            'HR_p': s.loc['group', 'p'],
            'C_index': c_idx,
        }
    except Exception:
        return {'HR': np.nan, 'HR_CI_low': np.nan, 'HR_CI_high': np.nan,
                'HR_p': np.nan, 'C_index': c_idx}


def calculate_metrics_by_cancer_type(df, model_names):
    """Calculate AUROC, AUPRC, and HR for each cancer type."""
    print("\n" + "="*70)
    print("Metrics by Cancer Type (AUROC, AUPRC, HR)")
    print("="*70)

    # Prepare OS columns for HR
    df = df.copy()
    df['os_months'] = df['_os_reg_months'].where(
        df['_os_reg_months'].notna() & (df['_os_reg_months'] > 0),
        df['_os_dx_months'],
    )
    df['os_status'] = df['_os_reg_status']

    results = []

    for cancer_type in sorted(df['cancer_type'].unique()):
        subset = df[df['cancer_type'] == cancer_type]
        y_true = subset['label'].values

        for name in model_names:
            y_prob = subset[f'pred_prob_{name}'].values
            metrics = evaluate_predictions(y_true, y_prob, min_samples=50)

            if metrics:
                metrics['cancer_type'] = cancer_type
                metrics['model'] = name

                # Compute HR
                hr_results = compute_hr(subset, f'pred_prob_{name}')
                metrics.update(hr_results)

                results.append(metrics)

                hr_str = f"HR={hr_results['HR']:.3f}" if not np.isnan(hr_results['HR']) else "HR=N/A"
                ci_str = f"[{hr_results['HR_CI_low']:.3f}-{hr_results['HR_CI_high']:.3f}]" if not np.isnan(hr_results['HR']) else ""
                p_str = f"p={hr_results['HR_p']:.2e}" if not np.isnan(hr_results['HR_p']) else ""
                sig = ''
                if not np.isnan(hr_results['HR_p']):
                    sig = '***' if hr_results['HR_p'] < 0.001 else '**' if hr_results['HR_p'] < 0.01 else '*' if hr_results['HR_p'] < 0.05 else ''

                print(f"  {cancer_type:15s} | {name:18s} | AUROC={metrics['auroc']:.4f}, "
                      f"AUPRC={metrics['auprc']:.4f}, {hr_str} {ci_str} {p_str} {sig}  n={metrics['n_samples']}")

    return pd.DataFrame(results)


def calculate_overall_metrics(all_fold_results):
    """Calculate overall metrics (mean +/- std across folds)."""
    print("\n" + "="*70)
    print("Overall Results (Mean +/- Std across 5 folds)")
    print("="*70)

    final_results = {}

    for name, fold_results in all_fold_results.items():
        if not fold_results:
            continue

        final_results[name] = {}
        print(f"\n{name}:")

        for metric in ['auroc', 'auprc', 'f1', 'accuracy', 'precision', 'recall', 'specificity']:
            values = [r[metric] for r in fold_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            final_results[name][metric] = {'mean': mean_val, 'std': std_val}
            print(f"  {metric:12s}: {mean_val:.4f} +/- {std_val:.4f}")

    return final_results


# ============================================================================
# Step 6: Save Results
# ============================================================================
def save_results(df, metrics_by_cancer, overall_metrics, feature_importance,
                 feature_cols, output_dir):
    """Save all results to files."""
    print(f"\n" + "="*70)
    print(f"Saving results to {output_dir}")
    print("="*70)

    pred_cols = [c for c in df.columns if c.startswith('pred_prob_')]
    model_names = [c.replace('pred_prob_', '') for c in pred_cols]

    for name in model_names:
        pred_df = df[['patient_id', 'drugs', 'cancer_type', 'label', 'score',
                      '_os_dx_months', '_os_dx_status', '_os_reg_months', '_os_reg_status',
                      'fold', f'pred_prob_{name}']].copy()
        pred_df.columns = ['patient_id', 'drugs', 'cancer_type', 'true_label', 'recist_score',
                          'os_dx_months', 'os_dx_status', 'os_reg_months', 'os_reg_status',
                          'fold', 'pred_prob']

        # Use regimen OS as primary; fall back to diagnosis OS when unavailable
        pred_df['os_months'] = pred_df['os_reg_months'].where(
            pred_df['os_reg_months'].notna() & (pred_df['os_reg_months'] > 0),
            pred_df['os_dx_months'],
        )
        pred_df['os_status'] = pred_df['os_reg_status']

        pred_df['pred_label'] = (pred_df['pred_prob'] >= 0.5).astype(int)
        pred_df['risk_group'] = pd.qcut(pred_df['pred_prob'], q=2, labels=['Low', 'High'])

        pred_path = os.path.join(output_dir, f'predictions_{name}.csv')
        pred_df.to_csv(pred_path, index=False)
        print(f"  Saved predictions to {pred_path}")

    if len(metrics_by_cancer) > 0:
        cancer_metrics_path = os.path.join(output_dir, 'metrics_by_cancer_type.csv')
        metrics_by_cancer.to_csv(cancer_metrics_path, index=False)
        print(f"  Saved cancer-specific metrics to {cancer_metrics_path}")

    overall_rows = []
    for name, metrics in overall_metrics.items():
        row = {'model': name}
        for metric, values in metrics.items():
            row[f'{metric}_mean'] = values['mean']
            row[f'{metric}_std'] = values['std']
        overall_rows.append(row)

    overall_df = pd.DataFrame(overall_rows)
    overall_path = os.path.join(output_dir, 'metrics_overall.csv')
    overall_df.to_csv(overall_path, index=False)
    print(f"  Saved overall metrics to {overall_path}")

    for name, importance in feature_importance.items():
        if np.sum(importance) > 0:
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance,
            }).sort_values('importance', ascending=False)

            importance_path = os.path.join(output_dir, f'feature_importance_{name}.csv')
            importance_df.to_csv(importance_path, index=False)
            print(f"  Saved {name} feature importance to {importance_path}")

    feature_matrix_path = os.path.join(output_dir, 'feature_matrix.csv')
    df.to_csv(feature_matrix_path, index=False)
    print(f"  Saved feature matrix to {feature_matrix_path}")

    print("\n" + "="*70)
    print("ML Baseline Performance")
    print("="*70)
    for name, metrics in overall_metrics.items():
        print(f"    {name}: AUROC = {metrics['auroc']['mean']:.4f} +/- {metrics['auroc']['std']:.4f}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("="*70)
    print("Traditional ML Baseline for OncoRAFT (10 Models)")
    print("Predicting Drug Regimen Response (RECIST)")
    print("="*70)

    labels_df = load_labels_from_jsonl(JSONL_PATH)
    msk_data = load_msk_data(MSK_DATA_DIR)

    patient_features = extract_patient_features(msk_data)
    sample_features_ = extract_sample_features(msk_data)
    mutation_features = extract_mutation_features(msk_data)
    treatment_timeline = load_treatment_timeline(MSK_DATA_DIR)

    df, feature_cols = build_feature_matrix(
        labels_df, patient_features, sample_features_, mutation_features,
        treatment_timeline=treatment_timeline,
    )

    df, all_fold_results, feature_importance, feature_cols, models = train_and_evaluate(
        df, feature_cols,
    )

    model_names = list(models.keys())
    metrics_by_cancer = calculate_metrics_by_cancer_type(df, model_names)
    overall_metrics = calculate_overall_metrics(all_fold_results)

    save_results(df, metrics_by_cancer, overall_metrics, feature_importance,
                 feature_cols, RESULTS_DIR)

    print("\n" + "="*70)
    print("DONE! Results saved to:", RESULTS_DIR)
    print("="*70)
    print("\nOutput files:")
    print("  - predictions_*.csv      : Per-sample predictions (for HR/KM analysis)")
    print("  - metrics_by_cancer_type.csv : AUROC/AUPRC per cancer type")
    print("  - metrics_overall.csv    : Overall metrics (mean +/- std)")
    print("  - feature_importance_*.csv : Feature importance rankings")
    print("  - feature_matrix.csv     : Complete feature matrix")

    return df, overall_metrics, metrics_by_cancer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Traditional ML Baseline for OncoRAFT (10 Models)')
    parser.add_argument('--no-strict', action='store_true',
                        help='Disable strict mode to use full feature set')
    args = parser.parse_args()

    if args.no_strict:
        import config
        config.STRICT_ONCORAFT_CONSISTENT_MODE = False
        print("WARNING: STRICT_ONCORAFT_CONSISTENT_MODE disabled - using full feature set")
    else:
        print("STRICT_ONCORAFT_CONSISTENT_MODE enabled (default)")

    main()
