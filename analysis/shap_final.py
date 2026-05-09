#!/usr/bin/env python3
"""
SHAP analysis on a surrogate RandomForest aligned with the OncoRAFT score head.

For each (cancer, regimen) cohort the script:
  - Builds a structured feature set (mutations + clinical + text-derived
    fields parsed from the patient prompt).
  - Trains a Random Forest surrogate to predict OncoRAFT's score head output.
  - Repeats over multiple random seeds and saves raw per-seed SHAP matrices
    so downstream aggregation (shap_final_aggregate.py) can compute
    bootstrap CIs over feature importance.
"""
import os, json, re, warnings
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from collections import Counter
from config import OUTPUT_DIR as CONFIG_OUTPUT_DIR, TRAINING_DATA, FEATURE_MATRIX_CSV, RESPONSE_ARRAY_CSV
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.environ.get('SHAP_OUTPUT_DIR', os.path.join(CONFIG_OUTPUT_DIR, 'shap_final'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/raw_shap', exist_ok=True)

N_SEEDS = 10
BASE_SEED = 42
MIN_GENE_FREQ = 500

# Cancer / regimen panel used for SHAP analysis
CANCER_REGIMENS = {
    'Breast': ['CAPECITABINE','CYCLOPHOSPHAMIDE, DOXORUBICIN, PACLITAXEL',
               'TAMOXIFEN','LETROZOLE','ANASTROZOLE','PACLITAXEL',
               'LETROZOLE, PALBOCICLIB','CYCLOPHOSPHAMIDE, DOXORUBICIN',
               'PACLITAXEL, PERTUZUMAB, TRASTUZUMAB','FULVESTRANT',
               'FULVESTRANT, PALBOCICLIB','CYCLOPHOSPHAMIDE, FLUOROURACIL, METHOTREXATE',
               'EXEMESTANE','EVEROLIMUS, EXEMESTANE'],
    'NSCLC': ['OSIMERTINIB','CARBOPLATIN, PEMETREXED',
              'CARBOPLATIN, PEMBROLIZUMAB, PEMETREXED','PEMBROLIZUMAB',
              'NIVOLUMAB','BEVACIZUMAB, CARBOPLATIN, PEMETREXED',
              'CISPLATIN, PEMETREXED','CARBOPLATIN, PACLITAXEL',
              'ERLOTINIB','GEMCITABINE','DOCETAXEL, RAMUCIRUMAB',
              'GEMCITABINE, VINORELBINE','DURVALUMAB','DOCETAXEL'],
    'Colorectal': ['FLUOROURACIL, OXALIPLATIN','CAPECITABINE',
                   'BEVACIZUMAB, FLUOROURACIL, IRINOTECAN','FLUOROURACIL, IRINOTECAN',
                   'FLOXURIDINE','CAPECITABINE, OXALIPLATIN',
                   'FLOXURIDINE, FLUOROURACIL, IRINOTECAN','IRINOTECAN',
                   'TRIFLURIDINE + TIPIRACIL','BEVACIZUMAB, FLUOROURACIL, OXALIPLATIN',
                   'FLUOROURACIL','OXALIPLATIN'],
    'Pancreatic': ['FLUOROURACIL, IRINOTECAN, OXALIPLATIN','GEMCITABINE, PACLITAXEL',
                   'CAPECITABINE','FLUOROURACIL, IRINOTECAN',
                   'GEMCITABINE','FLUOROURACIL, OXALIPLATIN'],
    'Prostate': ['ABIRATERONE','BICALUTAMIDE','ENZALUTAMIDE','DOCETAXEL'],
}

# Clinical features used by the surrogate model
FM_CLINICAL = ['tmb', 'msi_score', 'age', 'gender', 'stage_4',
               'hr_positive', 'her2_positive', 'is_metastasis', 'smoking_current_former']

SUPP = {'ZOLEDRONIC ACID','PAMIDRONATE','DENOSUMAB','LEUPROLIDE','GOSERELIN',
        'DEGARELIX','LEUCOVORIN','PREDNISONE','INVESTIGATIONAL','MEGESTROL'}

def norm_drug(d):
    if pd.isna(d): return None
    parts = [x.strip().upper() for x in d.split(',')]
    core = [x for x in parts if x not in SUPP]
    return ', '.join(sorted(core)) if core else None


def extract_features():
    print("Extracting features from text...")
    gene_counter = Counter()
    records = []
    with open(TRAINING_DATA) as f:
        for line in f:
            d = json.loads(line)
            inp = d.get('input', '')
            rec = {}
            pid_m = re.search(r'Patient ID: ([^\n]+)', inp)
            rec['patient_id'] = pid_m.group(1).strip() if pid_m else None

            genes = set()
            if '## Mutations' in inp:
                start = inp.find('## Mutations')
                end = inp.find('## Structural Variants') if '## Structural Variants' in inp else inp.find('\n# ', start+1)
                if end < 0: end = len(inp)
                genes = set(re.findall(r'([A-Z][A-Z0-9-]+)\s+(?:p\.|Splice|Missense|Nonsense|Frame_Shift|In_Frame)',
                                       inp[start:end]))
                gene_counter.update(genes)
            rec['_genes'] = genes

            # Biomarkers from text
            cea = re.findall(r'CEA:\s*([\d.]+)\s*ng/ml', inp)
            rec['cea_raw'] = max(float(v) for v in cea) if cea else np.nan
            ca19 = re.findall(r'CA 19-9:\s*([\d.]+)\s*U(?:nits)?/ml', inp)
            rec['ca19_9_raw'] = max(float(v) for v in ca19) if ca19 else np.nan
            pdl1 = re.search(r'PD-L1 Positive:\s*(Yes|No)', inp)
            rec['pdl1_raw'] = (1 if pdl1.group(1) == 'Yes' else 0) if pdl1 else np.nan
            gleason = re.search(r'Gleason Score:\s*(\d+)', inp)
            rec['gleason_raw'] = int(gleason.group(1)) if gleason else np.nan

            # Metastatic site: liver (from text)
            met_section = ''
            if 'Metastatic Site:' in inp:
                met_section = inp[inp.find('Metastatic Site:'):inp.find('Metastatic Site:')+200].lower()
            rec['met_liver'] = 1 if 'liver' in met_section else 0

            # n_prior_treatment_entries: literal comma-separated count from text
            prior_m = re.search(r'Prior Treatments:\s*([^\n]+)', inp)
            if prior_m:
                items = [x.strip() for x in prior_m.group(1).split(',') if x.strip()]
                rec['n_prior_treatment_entries'] = len(items)
            else:
                rec['n_prior_treatment_entries'] = 0

            records.append(rec)

    selected_genes = sorted([g for g, c in gene_counter.items() if c >= MIN_GENE_FREQ])
    for rec in records:
        genes = rec.pop('_genes')
        for g in selected_genes:
            rec[f'mut_{g}'] = 1 if g in genes else 0

    text_df = pd.DataFrame(records)
    print(f"  {len(selected_genes)} genes, {len(text_df)} samples")
    return text_df, selected_genes


def build_dataset(text_df, selected_genes):
    print("\nBuilding dataset...")
    fm = pd.read_csv(FEATURE_MATRIX_CSV)
    ra = pd.read_csv(RESPONSE_ARRAY_CSV)
    fm = fm.merge(ra, on='sample_idx', how='left')
    fm['core_regimen'] = fm['drugs'].apply(norm_drug)

    gene_cols = [f'mut_{g}' for g in selected_genes]
    for col in gene_cols + ['cea_raw','ca19_9_raw','pdl1_raw','gleason_raw',
                            'met_liver','n_prior_treatment_entries']:
        fm[col] = text_df[col].values

    fm['patient_id'] = text_df['patient_id'].values
    fm['non_white'] = 1 - fm['race_white']

    # Biomarkers: median imputation, no missing indicators
    biomarker_map = {
        'cea':     ('cea_raw',     'cea_value',     True),
        'ca19_9':  ('ca19_9_raw',  'ca19_9_value',  True),
        'pdl1':    ('pdl1_raw',    'pdl1_value',    False),
        'gleason': ('gleason_raw', 'gleason_value', False),
    }
    for key, (raw_col, val_col, do_log) in biomarker_map.items():
        vals = fm[raw_col].copy()
        if do_log:
            vals = np.log1p(vals)
        median_val = vals.median()
        fm[val_col] = vals.fillna(median_val)
        print(f"  {key:10s}: median={median_val:.2f}, missing={fm[raw_col].isna().sum()} ({100*fm[raw_col].isna().mean():.1f}%)")

    # Build full training feature list (mutation + clinical + text-derived)
    ALL_TRAIN = (gene_cols + FM_CLINICAL +
                 ['met_liver', 'cea_value', 'ca19_9_value', 'pdl1_value',
                  'gleason_value', 'non_white', 'n_prior_treatment_entries'])
    print(f"  Training features: {len(ALL_TRAIN)}")
    return fm, gene_cols, ALL_TRAIN


def run_shap(fm, gene_cols, all_train):
    print(f"\nRunning SHAP ({N_SEEDS} seeds)...")
    r2_rows = []
    regimen_list = []

    # Save feature names
    with open(f'{OUTPUT_DIR}/raw_shap/feature_names.json', 'w') as f:
        json.dump(all_train, f)

    for cancer, regimens in CANCER_REGIMENS.items():
        for drug in regimens:
            sub = fm[(fm['cancer_type'] == cancer) & (fm['core_regimen'] == drug)]
            if len(sub) < 200:
                continue
            X = sub[all_train].fillna(0).values
            y = sub['score_ra'].values
            pids = sub['patient_id'].values

            shap_list = []
            oob_list = []
            for s in range(N_SEEDS):
                m = RandomForestRegressor(100, max_depth=10, random_state=BASE_SEED+s,
                                          n_jobs=-1, oob_score=True)
                m.fit(X, y)
                shap_list.append(shap.TreeExplainer(m).shap_values(X))
                oob_list.append(m.oob_score_)

            avg_shap = np.mean(shap_list, axis=0)

            # Pearson r from seed-0 OOB predictions
            m0 = RandomForestRegressor(100, max_depth=10, random_state=BASE_SEED,
                                        n_jobs=-1, oob_score=True)
            m0.fit(X, y)
            r_p = pearsonr(y, m0.oob_prediction_)[0]

            # Save raw SHAP matrices
            safe_drug = drug.replace(',', '_').replace(' ', '').replace('+', 'plus')
            prefix = f'{OUTPUT_DIR}/raw_shap/{cancer}__{safe_drug}'
            np.save(f'{prefix}__shap.npy', avg_shap)
            np.save(f'{prefix}__X.npy', X)
            np.save(f'{prefix}__y.npy', y)
            np.save(f'{prefix}__patient_ids.npy', pids)

            regimen_list.append({'cancer': cancer, 'regimen': drug, 'n': len(sub)})
            r2_rows.append({
                'cancer': cancer, 'regimen': drug, 'n': len(sub),
                'oob_r2': np.mean(oob_list), 'pearson_r': r_p,
            })
            print(f'  {cancer}/{drug:45s} n={len(sub):5d} R2={np.mean(oob_list):.3f} r={r_p:.3f}')

    # Save regimen list
    with open(f'{OUTPUT_DIR}/raw_shap/regimen_list.json', 'w') as f:
        json.dump(regimen_list, f, indent=2)

    r2_df = pd.DataFrame(r2_rows)
    r2_df.to_csv(f'{OUTPUT_DIR}/r2_summary.csv', index=False)
    return r2_df


def main():
    text_df, selected_genes = extract_features()
    fm, gene_cols, all_train = build_dataset(text_df, selected_genes)

    print(f"\n--- Feature set ---")
    print(f"  Genes: {len(gene_cols)}")
    print(f"  FM clinical: {FM_CLINICAL}")
    print(f"  Text-native: met_liver, cea_value, ca19_9_value, pdl1_value, gleason_value, non_white, n_prior_treatment_entries")
    print(f"  Total: {len(all_train)}")

    r2_df = run_shap(fm, gene_cols, all_train)

    print(f'\nMean OOB R2={r2_df["oob_r2"].mean():.4f}  Pearson r={r2_df["pearson_r"].mean():.4f}')
    print(f'Raw SHAP saved to: {OUTPUT_DIR}/raw_shap/')
    print('[DONE] Run shap_final_aggregate.py next.')


if __name__ == '__main__':
    main()
