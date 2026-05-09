#!/usr/bin/env python3
"""
TCGA External Validation for ML Baseline Models
Loads saved ML models (trained on MSK-CHORD), extracts the same structured
features from TCGA data, and writes per-sample predictions to CSV.
"""

import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

from config import (
    MODEL_DIR, TCGA_DIR, TCGA_OUTPUT_DIR,
    TCGA_CLINICAL_FILE, TCGA_TREATMENT_PLANS_FILE, TCGA_MUTATION_FILE,
    TCGA_SURVIVAL_FILE, TCGA_CBIOPORTAL_DIR, TCGA_METADATA_FILE,
    TOP_GENES, NONSYNONYMOUS_EFFECTS,
    PROJECT_TO_CANCER, PROJECT_TO_FEATURE_CANCER, SUBTYPE_MAP,
    MODEL_NAMES, MODELS_NEEDING_SCALING,
)
from drug_utils import (
    extract_drug_features, normalize_drug, get_core_combination,
    classify_regimen, DRUG_CLASSES, SUPPORTIVE_DRUGS,
)

OUTPUT_DIR = Path(TCGA_OUTPUT_DIR)
MODEL_DIR_PATH = Path(MODEL_DIR)


# ============================================
# Data Loading
# ============================================

def load_clinical_data():
    print("[1/6] Loading clinical data...")
    df = pd.read_csv(TCGA_CLINICAL_FILE)
    print(f"  {len(df)} records, {df['patient_id'].nunique()} patients")
    return df


def load_survival_data():
    print("[2/6] Loading survival data...")
    df = pd.read_excel(TCGA_SURVIVAL_FILE, sheet_name='TCGA-CDR')
    df = df.rename(columns={'bcr_patient_barcode': 'patient_id'})
    cols = ['patient_id', 'type', 'OS', 'OS.time', 'PFI', 'PFI.time',
            'age_at_initial_pathologic_diagnosis', 'gender', 'race',
            'ajcc_pathologic_tumor_stage', 'clinical_stage']
    df = df[[c for c in cols if c in df.columns]]
    print(f"  {len(df)} records, {df['patient_id'].nunique()} patients")
    return df


def load_treatment_plans():
    print("[3/6] Loading treatment plans...")
    with open(TCGA_TREATMENT_PLANS_FILE) as f:
        plans = json.load(f)
    print(f"  {len(plans)} patients")
    return plans


def load_mutations():
    print("[4/6] Loading mutations (MC3)...")
    top_set = set(TOP_GENES)
    patient_genes = defaultdict(set)
    patient_nonsyn = defaultdict(int)
    patient_total_genes = defaultdict(set)

    for chunk in pd.read_csv(TCGA_MUTATION_FILE, sep='\t', chunksize=500000,
                             usecols=['sample', 'gene', 'effect']):
        chunk['patient_id'] = chunk['sample'].str[:12]
        for _, row in chunk.iterrows():
            pid, gene, effect = row['patient_id'], row['gene'], row['effect']
            if gene in top_set and effect != 'Silent':
                patient_genes[pid].add(gene)
            if effect in NONSYNONYMOUS_EFFECTS:
                patient_nonsyn[pid] += 1
            if effect != 'Silent':
                patient_total_genes[pid].add(gene)

    patient_tmb = {pid: c / 30.0 for pid, c in patient_nonsyn.items()}
    patient_mut_count = {pid: len(g) for pid, g in patient_total_genes.items()}
    print(f"  {len(patient_genes)} patients with mutations")
    return patient_genes, patient_tmb, patient_mut_count


def load_cbioportal_data():
    print("[5/6] Loading cBioPortal data...")
    cbio_dir = Path(TCGA_CBIOPORTAL_DIR)
    pat = defaultdict(dict)
    sam = defaultdict(dict)
    for f in sorted(cbio_dir.glob('*_clinical.json')):
        for item in json.load(open(f)):
            pat[item['patientId']][item['clinicalAttributeId']] = item['value']
    for f in sorted(cbio_dir.glob('*_sample.json')):
        for item in json.load(open(f)):
            pid, attr = item['patientId'], item['clinicalAttributeId']
            if attr not in sam[pid]:
                sam[pid][attr] = item['value']
    print(f"  {len(pat)} patients, {len(sam)} samples")
    return pat, sam


def load_metadata():
    print("[6/6] Loading metadata...")
    df = pd.read_csv(TCGA_METADATA_FILE)
    print(f"  {len(df)} rows, {df['patient_id'].nunique()} patients")
    return df


# ============================================
# Feature Extraction (same structured features as MSK training)
# ============================================

def extract_clinical_features(pid, clin_df, surv_df, cbio_pat, cbio_sam, project_id):
    feats = {}
    clin = clin_df[clin_df['patient_id'] == pid]
    surv = surv_df[surv_df['patient_id'] == pid]

    # Gender
    g = ''
    if len(clin) > 0 and 'gender' in clin.columns:
        g = str(clin.iloc[0]['gender']).lower()
    elif len(surv) > 0 and 'gender' in surv.columns:
        g = str(surv.iloc[0]['gender']).lower()
    feats['gender'] = 1 if 'male' in g and 'female' not in g else 0

    # Age
    age = np.nan
    if len(clin) > 0 and 'age_at_diagnosis' in clin.columns:
        try:
            v = float(clin.iloc[0]['age_at_diagnosis'])
            age = v / 365.25 if v > 200 else v
        except: pass
    if np.isnan(age) and len(clin) > 0 and 'days_to_birth' in clin.columns:
        try: age = abs(float(clin.iloc[0]['days_to_birth'])) / 365.25
        except: pass
    if np.isnan(age) and len(surv) > 0 and 'age_at_initial_pathologic_diagnosis' in surv.columns:
        try: age = float(surv.iloc[0]['age_at_initial_pathologic_diagnosis'])
        except: pass
    if np.isnan(age) and pid in cbio_pat and 'AGE' in cbio_pat[pid]:
        try: age = float(cbio_pat[pid]['AGE'])
        except: pass
    feats['age'] = age

    # Race
    r = ''
    if len(clin) > 0 and 'race' in clin.columns:
        r = str(clin.iloc[0]['race']).lower()
    elif pid in cbio_pat and 'RACE' in cbio_pat[pid]:
        r = cbio_pat[pid]['RACE'].lower()
    feats['race_white'] = 1 if 'white' in r else 0
    feats['race_black'] = 1 if 'black' in r else 0
    feats['race_asian'] = 1 if 'asian' in r else 0

    # Ethnicity
    e = ''
    if len(clin) > 0 and 'ethnicity' in clin.columns:
        e = str(clin.iloc[0]['ethnicity']).lower()
    elif pid in cbio_pat and 'ETHNICITY' in cbio_pat[pid]:
        e = cbio_pat[pid]['ETHNICITY'].lower()
    feats['ethnicity_hispanic'] = 1 if 'hispanic' in e or 'spanish' in e else 0

    # Smoking
    s = ''
    if len(clin) > 0 and 'tobacco_smoking_status' in clin.columns:
        s = str(clin.iloc[0]['tobacco_smoking_status']).lower()
    feats['smoking_current_former'] = 1 if any(k in s for k in ['current', 'former', 'reformed']) else 0
    feats['smoking_never'] = 1 if 'never' in s or s == 'lifelong non-smoker' else 0

    # Stage 4
    st = ''
    if len(surv) > 0 and 'ajcc_pathologic_tumor_stage' in surv.columns:
        st = str(surv.iloc[0]['ajcc_pathologic_tumor_stage']).lower()
    if not st or st == 'nan':
        if len(clin) > 0 and 'ajcc_pathologic_stage' in clin.columns:
            st = str(clin.iloc[0]['ajcc_pathologic_stage']).lower()
    feats['stage_4'] = 1 if 'iv' in st or 'stage 4' in st else 0

    feats['prior_treatment'] = 0

    # HR/HER2
    hr_pos = her2_pos = 0
    cancer = PROJECT_TO_FEATURE_CANCER.get(project_id, '')
    if cancer == 'breast' and pid in cbio_pat:
        sub = cbio_pat[pid].get('SUBTYPE', '')
        if sub in SUBTYPE_MAP:
            hr_pos = SUBTYPE_MAP[sub]['hr_positive']
            her2_pos = SUBTYPE_MAP[sub]['her2_positive']
    feats['hr_positive'] = hr_pos
    feats['her2_positive'] = her2_pos
    return feats


def extract_sample_features(pid, project_id, cbio_sam, tmb_dict, mut_count_dict):
    feats = {}
    c = PROJECT_TO_FEATURE_CANCER.get(project_id, '')
    feats['cancer_nsclc'] = 1 if c == 'nsclc' else 0
    feats['cancer_breast'] = 1 if c == 'breast' else 0
    feats['cancer_colorectal'] = 1 if c == 'colorectal' else 0
    feats['cancer_pancreatic'] = 1 if c == 'pancreatic' else 0
    feats['cancer_prostate'] = 1 if c == 'prostate' else 0

    st = cbio_sam.get(pid, {}).get('SAMPLE_TYPE', '').lower() if pid in cbio_sam else ''
    feats['is_metastasis'] = 1 if 'metast' in st else 0

    msi_score = np.nan; msi_stable = 0; msi_instable = 0
    if pid in cbio_sam:
        sa = cbio_sam[pid]
        if 'MSI_SCORE_MANTIS' in sa:
            try:
                msi_score = float(sa['MSI_SCORE_MANTIS'])
                msi_instable = 1 if msi_score >= 0.4 else 0
                msi_stable = 1 - msi_instable
            except: pass
        if 'MSI_SENSOR_SCORE' in sa and msi_stable == 0 and msi_instable == 0:
            try:
                ms = float(sa['MSI_SENSOR_SCORE'])
                msi_instable = 1 if ms >= 3.5 else 0
                msi_stable = 1 - msi_instable
                if np.isnan(msi_score): msi_score = ms
            except: pass
    feats['msi_stable'] = msi_stable
    feats['msi_instable'] = msi_instable
    feats['msi_score'] = msi_score

    tmb = np.nan
    if pid in cbio_sam and 'TMB_NONSYNONYMOUS' in cbio_sam[pid]:
        try: tmb = float(cbio_sam[pid]['TMB_NONSYNONYMOUS'])
        except: pass
    if np.isnan(tmb) and pid in tmb_dict:
        tmb = tmb_dict[pid]
    feats['tmb'] = tmb
    return feats


def extract_mutation_features(pid, gene_dict, mut_count_dict):
    feats = {}
    genes = gene_dict.get(pid, set())
    for g in TOP_GENES:
        feats[f'mut_{g}'] = 1 if g in genes else 0
    mc = mut_count_dict.get(pid, 0)
    feats['mutation_count'] = mc
    feats['has_any_mutation'] = 1 if mc > 0 else 0
    return feats


def normalize_regimen(drug_str):
    """Normalize drug regimen: normalize names, remove supportive, sort, join."""
    return get_core_combination(drug_str).replace(', ', ' + ')


# ============================================
# Build Features & Predict
# ============================================

def build_feature_matrix():
    """Build feature matrix for all TCGA patient-drug pairs."""
    clin_df = load_clinical_data()
    surv_df = load_survival_data()
    plans = load_treatment_plans()
    pat_genes, pat_tmb, pat_mc = load_mutations()
    cbio_pat, cbio_sam = load_cbioportal_data()
    metadata = load_metadata()
    feature_cols = joblib.load(MODEL_DIR_PATH / 'feature_cols.joblib')
    print(f"\nExpected features: {len(feature_cols)}")

    # Build plan_start mapping
    plan_start = {}
    for pid, plan_list in plans.items():
        for p in plan_list:
            sd = p.get('start_date')
            if sd is not None and str(sd) != 'nan':
                try:
                    val = float(sd)
                    if not np.isnan(val):
                        plan_start[(pid, p['plan_id'])] = val
                except (ValueError, TypeError):
                    pass
    print(f"  Plan start dates available for {len(plan_start)} patient-plan pairs")

    # Recover missing start_dates from raw drug data
    DRUG_FILE = Path(TCGA_DIR) / 'data' / 'drugs' / 'all_projects_drugs.csv'
    if DRUG_FILE.exists():
        drug_df = pd.read_csv(DRUG_FILE)
        recovered_count = 0
        for pid, plan_list in plans.items():
            known_starts = set()
            missing_plans_local = []
            for p in plan_list:
                key = (pid, p['plan_id'])
                if key in plan_start:
                    known_starts.add(plan_start[key])
                else:
                    missing_plans_local.append(p)
            if not missing_plans_local:
                continue
            pat_drugs = drug_df[drug_df['patient_id'] == pid]
            all_dates = pat_drugs['days_to_start'].dropna().unique()
            unused_dates = sorted([d for d in all_dates if d not in known_starts])
            if not unused_dates:
                continue
            for p in missing_plans_local:
                key = (pid, p['plan_id'])
                drugs = [d['agent'] if isinstance(d, dict) else d for d in p.get('drugs', [])]
                direct_dates = []
                for dn in drugs:
                    if not isinstance(dn, str) or dn == 'Unknown':
                        continue
                    prefix = dn[:4]
                    matched = pat_drugs[pat_drugs['drug_name'].str.contains(prefix, case=False, na=False)]
                    for d in matched['days_to_start'].dropna():
                        if d not in known_starts:
                            direct_dates.append(d)
                if direct_dates:
                    plan_start[key] = min(direct_dates)
                    known_starts.add(plan_start[key])
                    recovered_count += 1
                    continue
                if len(missing_plans_local) == 1 and len(unused_dates) == 1:
                    plan_start[key] = unused_dates[0]
                    known_starts.add(plan_start[key])
                    recovered_count += 1
                    continue
                my_num = int(p['plan_id'].split('_')[1])
                before_vals = [plan_start.get((pid, pp['plan_id']), float('nan'))
                               for pp in plan_list if int(pp['plan_id'].split('_')[1]) < my_num]
                after_vals = [plan_start.get((pid, pp['plan_id']), float('nan'))
                              for pp in plan_list if int(pp['plan_id'].split('_')[1]) > my_num]
                lo = max([v for v in before_vals if not math.isnan(v)], default=-float('inf'))
                hi = min([v for v in after_vals if not math.isnan(v)], default=float('inf'))
                candidates = [d for d in unused_dates if lo < d < hi]
                if len(candidates) == 1:
                    plan_start[key] = candidates[0]
                    known_starts.add(plan_start[key])
                    recovered_count += 1
        print(f"  Recovered {recovered_count} missing start_dates from raw drug data")
        print(f"  Plan start dates now available for {len(plan_start)} patient-plan pairs")

    rows = []
    meta_rows = []
    print("\nExtracting features...")
    for idx, row in metadata.iterrows():
        pid = row['patient_id']
        proj = row['cancer_type']
        drug_str = row['drug_name']

        drugs = [d.strip() for d in drug_str.split(',')]
        feats = {}
        feats.update(extract_drug_features(drugs))
        feats.update(extract_clinical_features(pid, clin_df, surv_df, cbio_pat, cbio_sam, proj))
        feats.update(extract_sample_features(pid, proj, cbio_sam, pat_tmb, pat_mc))
        feats.update(extract_mutation_features(pid, pat_genes, pat_mc))
        rows.append(feats)

        sd = plan_start.get((pid, row['plan_id']))
        os_reg_time = None
        if sd is not None and not pd.isna(row['os_time']):
            os_reg_time = row['os_time'] - sd
            if os_reg_time < 0:
                os_reg_time = None

        meta_rows.append({
            'patient_id': pid,
            'plan_id': row['plan_id'],
            'drug_name': drug_str,
            'normalized_regimen': normalize_regimen(drug_str),
            'cancer_type': proj,
            'merged_cancer': PROJECT_TO_CANCER.get(proj, 'Other'),
            'regimen_type': row.get('regimen_type', ''),
            'regimen_name': row.get('regimen_name', ''),
            'os_dx_time': row['os_time'],
            'os_reg_time': os_reg_time,
            'os_time': row['os_time'],
            'os_event': row['os_event'],
            'start_date': sd,
            'pfi_time': row.get('pfi_time', np.nan),
            'pfi_event': row.get('pfi_event', np.nan),
        })
        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(metadata)}...")

    X = pd.DataFrame(rows)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_cols]
    meta = pd.DataFrame(meta_rows)
    print(f"\nFeature matrix: {X.shape}, Meta: {meta.shape}")
    return X, meta, feature_cols


def predict_all_models(X_df):
    """Predict with all saved full-data models."""
    fill_values = joblib.load(MODEL_DIR_PATH / 'fill_values.joblib')
    scaler = joblib.load(MODEL_DIR_PATH / 'scaler.joblib')
    X_filled = X_df.fillna(fill_values).values
    X_scaled = scaler.transform(X_filled)

    preds = {}
    for name in MODEL_NAMES:
        path = MODEL_DIR_PATH / f'{name}.joblib'
        if not path.exists():
            continue
        model = joblib.load(path)
        X_in = X_scaled if name in MODELS_NEEDING_SCALING else X_filled
        prob = model.predict_proba(X_in)[:, 1]
        preds[name] = prob
        print(f"  {name}: mean={prob.mean():.4f}, std={prob.std():.4f}")
    return preds


def predict_all_models_5fold(X_df):
    """Predict with 5-fold models for each algorithm. Returns per-fold + ensemble."""
    feature_cols = joblib.load(MODEL_DIR_PATH / 'feature_cols.joblib')

    preds = {}
    for name in MODEL_NAMES:
        fold_preds = []
        for fold in range(5):
            fold_dir = MODEL_DIR_PATH / f'fold_{fold}'
            if not (fold_dir / f'{name}.joblib').exists():
                continue
            fill_values = joblib.load(fold_dir / 'fill_values.joblib')
            scaler = joblib.load(fold_dir / 'scaler.joblib')
            model = joblib.load(fold_dir / f'{name}.joblib')

            X_filled = X_df.fillna(fill_values).values
            X_in = scaler.transform(X_filled) if name in MODELS_NEEDING_SCALING else X_filled
            prob = model.predict_proba(X_in)[:, 1]
            fold_preds.append(prob)
            preds[f'{name}_fold_{fold}'] = prob

        if fold_preds:
            preds[f'{name}_ensemble'] = np.mean(fold_preds, axis=0)
            print(f"  {name}: ensemble mean={preds[f'{name}_ensemble'].mean():.4f}, "
                  f"std={preds[f'{name}_ensemble'].std():.4f} ({len(fold_preds)} folds)")
    return preds


# ============================================
# Main
# ============================================

def main():
    print("=" * 60)
    print("TCGA External Validation for ML Baseline")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Build features
    X_df, meta_df, feature_cols = build_feature_matrix()
    pd.concat([meta_df, X_df], axis=1).to_csv(OUTPUT_DIR / 'tcga_feature_matrix.csv', index=False)

    # Step 2: Predict with 5-fold ensemble
    print("\nPredicting with 5-fold models (ensemble for external validation)...")
    fold_predictions = predict_all_models_5fold(X_df)

    pred_df = meta_df.copy()
    for name, p in fold_predictions.items():
        pred_df[f'pred_{name}'] = p
    pred_df.to_csv(OUTPUT_DIR / 'tcga_predictions.csv', index=False)
    print(f"\nSaved predictions: {OUTPUT_DIR / 'tcga_predictions.csv'}")
    print("Done!")


if __name__ == '__main__':
    main()
