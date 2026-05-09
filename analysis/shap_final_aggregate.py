#!/usr/bin/env python3
"""
Aggregate raw SHAP matrices into subpopulation importance metrics.
Input: raw_shap/*.npy from shap_final.py
Output: shap_importance.csv
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from config import OUTPUT_DIR as CONFIG_OUTPUT_DIR

OUTPUT_DIR = os.environ.get('SHAP_OUTPUT_DIR', os.path.join(CONFIG_OUTPUT_DIR, 'shap_final'))
RAW_DIR = f'{OUTPUT_DIR}/raw_shap'
N_BOOTSTRAP = 200

# Load feature names and regimen list
with open(f'{RAW_DIR}/feature_names.json') as f:
    FEATURE_NAMES = json.load(f)
with open(f'{RAW_DIR}/regimen_list.json') as f:
    REGIMEN_LIST = json.load(f)

# Binary features
GENE_FEATURES = {f for f in FEATURE_NAMES if f.startswith('mut_')}
BINARY_FEATURES = GENE_FEATURES | {
    'gender', 'stage_4', 'hr_positive', 'her2_positive',
    'is_metastasis', 'smoking_current_former', 'met_liver', 'non_white',
}

# Continuous feature cutoffs (clinical thresholds)
CONTINUOUS_CUTOFFS = {
    'tmb':                       10.0,   # FDA tumor-agnostic pembrolizumab
    'msi_score':                 10.0,   # MSI-H threshold (zero-inflated, median useless)
    'gleason_value':              8.0,   # NCCN high-risk prostate
    'cea_value':       np.log1p(  5.0),  # 5 ng/mL (log1p-transformed)
    'ca19_9_value':    np.log1p( 37.0),  # 37 U/mL (log1p-transformed)
    'pdl1_value':                 0.5,   # PD-L1 positive (binary 0/1 in data)
    'age':                       None,   # use median
    'n_prior_treatment_entries':  1,     # >=1 = has prior treatment entries
}



def aggregate_cell(shap_col, x_col, feat_name, random_state=42):
    n_cohort = len(shap_col)
    mean_shap_full = shap_col.mean()
    mean_abs_shap = np.abs(shap_col).mean()

    # corr_signed_shap: |SHAP| signed by Pearson(x, SHAP) direction
    if np.std(x_col) < 1e-10 or np.std(shap_col) < 1e-10:
        corr_signed = mean_shap_full
    else:
        r, _ = pearsonr(x_col, shap_col)
        corr_signed = np.sign(r) * mean_abs_shap if not np.isnan(r) else mean_shap_full

    # Subpopulation definition
    if feat_name in BINARY_FEATURES:
        subpop_mask = (x_col == 1)
    elif feat_name in CONTINUOUS_CUTOFFS:
        cutoff = CONTINUOUS_CUTOFFS[feat_name]
        if cutoff is None:
            cutoff = np.median(x_col)
        subpop_mask = (x_col >= cutoff)
    else:
        subpop_mask = np.ones(n_cohort, dtype=bool)

    prevalence = subpop_mask.mean()
    n_subpop = subpop_mask.sum()

    if n_subpop == 0:
        return {
            'mean_shap_full': mean_shap_full,
            'mean_abs_shap': mean_abs_shap,
            'corr_signed_shap': corr_signed,
            'mean_shap_subpop': np.nan,
            'n_cohort': n_cohort,
            'n_subpop': 0,
            'prevalence': prevalence,
            'ci_low': np.nan,
            'ci_high': np.nan,
            'masked': True,
            'mask_reason': 'empty_subpop',
        }

    subpop_shap = shap_col[subpop_mask]
    mean_shap_subpop = subpop_shap.mean()

    # Bootstrap 95% CI
    if n_subpop >= 10:
        rng = np.random.default_rng(random_state)
        boot_means = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            idx = rng.integers(0, n_subpop, n_subpop)
            boot_means[b] = subpop_shap[idx].mean()
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    else:
        ci_low, ci_high = np.nan, np.nan

    # Mask cells with n<10 or CI crosses zero (sign uncertain)
    masked = False
    mask_reason = None
    if n_subpop < 10:
        masked = True
        mask_reason = f'n_subpop<10 (n={n_subpop})'
    elif not np.isnan(ci_low) and ci_low * ci_high < 0:
        masked = True
        mask_reason = 'CI_crosses_zero'

    return {
        'mean_shap_full': mean_shap_full,
        'mean_abs_shap': mean_abs_shap,
        'corr_signed_shap': corr_signed,
        'mean_shap_subpop': mean_shap_subpop,
        'n_cohort': n_cohort,
        'n_subpop': n_subpop,
        'prevalence': prevalence,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'masked': masked,
        'mask_reason': mask_reason if mask_reason else '',
    }


def main():
    print(f"Aggregating from {RAW_DIR}...")
    rows = []

    for entry in REGIMEN_LIST:
        cancer = entry['cancer']
        drug = entry['regimen']
        safe_drug = drug.replace(',', '_').replace(' ', '').replace('+', 'plus')
        prefix = f'{RAW_DIR}/{cancer}__{safe_drug}'

        shap_mat = np.load(f'{prefix}__shap.npy')
        X_mat = np.load(f'{prefix}__X.npy')

        for j, feat in enumerate(FEATURE_NAMES):
            result = aggregate_cell(shap_mat[:, j], X_mat[:, j], feat)
            result['cancer'] = cancer
            result['regimen'] = drug
            result['feature'] = feat
            rows.append(result)

        print(f'  {cancer}/{drug}')

    df = pd.DataFrame(rows)
    col_order = ['cancer', 'regimen', 'feature',
                 'mean_shap_full', 'mean_abs_shap', 'corr_signed_shap', 'mean_shap_subpop',
                 'n_cohort', 'n_subpop', 'prevalence', 'ci_low', 'ci_high',
                 'masked', 'mask_reason']
    df = df[col_order]
    df.to_csv(f'{OUTPUT_DIR}/shap_importance.csv', index=False)
    print(f'\nSaved: {OUTPUT_DIR}/shap_importance.csv')
    print(f'  {len(df)} rows ({len(REGIMEN_LIST)} regimens x {len(FEATURE_NAMES)} features)')

    # Quick summary
    n_masked = df['masked'].sum()
    print(f'  Masked cells: {n_masked} ({100*n_masked/len(df):.1f}%)')
    print(f'[DONE] Importance CSV written to {OUTPUT_DIR}/shap_importance.csv.')


if __name__ == '__main__':
    main()
