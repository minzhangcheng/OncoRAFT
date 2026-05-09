#!/usr/bin/env python3
"""
Multi-group analysis: 2-group (50%), 3-group (33%), 4-group (25%).
Cancer-type level + per-regimen level on MSK.
Forest plots + KM curves.
"""
import os, re, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from config import (
    OUTPUT_DIR as _CFG_OUTPUT_DIR,
    FEATURE_MATRIX_CSV,
    RESPONSE_ARRAY_CSV,
    CLINICAL_SAMPLE_FILE,
)
warnings.filterwarnings('ignore')

BASE = os.environ.get('MULTIGROUP_OUTPUT_DIR', os.path.join(_CFG_OUTPUT_DIR, 'multigroup_survival'))
FOREST_DIR = f'{BASE}/forest'
KM_DIR = f'{BASE}/km'
DATA_DIR = f'{BASE}/data'
for d in [FOREST_DIR, KM_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

SUPPORTIVE_DRUGS = {
    'ZOLEDRONIC ACID','PAMIDRONATE','DENOSUMAB','LEUPROLIDE','GOSERELIN',
    'DEGARELIX','LEUCOVORIN','PREDNISONE','INVESTIGATIONAL','MEGESTROL',
}

CANCER_ORDER_MSK = ['Breast','NSCLC','Colorectal','Pancreatic','Prostate']
CANCER_LABELS = {
    'Breast':'Breast Cancer','NSCLC':'Non-small Cell Lung Cancer',
    'Colorectal':'Colon Cancer','Pancreatic':'Pancreas Cancer','Prostate':'Prostate Cancer',
}
CANCER_MAP_CLINICAL = {
    'Breast Cancer':'Breast','Non-Small Cell Lung Cancer':'NSCLC',
    'Colorectal Cancer':'Colorectal','Prostate Cancer':'Prostate','Pancreatic Cancer':'Pancreatic',
}

REGIMEN_ORDER = {
    'Breast': ['CYCLOPHOSPHAMIDE, DOXORUBICIN','CAPECITABINE','CYCLOPHOSPHAMIDE, DOXORUBICIN, PACLITAXEL',
        'CYCLOPHOSPHAMIDE, FLUOROURACIL, METHOTREXATE','PACLITAXEL','FULVESTRANT','LETROZOLE',
        'ANASTROZOLE','TAMOXIFEN','EXEMESTANE','PACLITAXEL, PERTUZUMAB, TRASTUZUMAB',
        'FULVESTRANT, PALBOCICLIB','EVEROLIMUS, EXEMESTANE','LETROZOLE, PALBOCICLIB'],
    'NSCLC': ['CISPLATIN, PEMETREXED','CARBOPLATIN, PEMETREXED','CARBOPLATIN, PACLITAXEL','DOCETAXEL',
        'GEMCITABINE','GEMCITABINE, VINORELBINE','DURVALUMAB','NIVOLUMAB','PEMBROLIZUMAB',
        'ERLOTINIB','OSIMERTINIB','CARBOPLATIN, PEMBROLIZUMAB, PEMETREXED',
        'BEVACIZUMAB, CARBOPLATIN, PEMETREXED','DOCETAXEL, RAMUCIRUMAB'],
    'Colorectal': ['CAPECITABINE, OXALIPLATIN','FLUOROURACIL, OXALIPLATIN',
        'FLOXURIDINE, FLUOROURACIL, IRINOTECAN','CAPECITABINE','FLUOROURACIL, IRINOTECAN',
        'FLOXURIDINE','FLUOROURACIL','IRINOTECAN','OXALIPLATIN','TRIFLURIDINE + TIPIRACIL',
        'BEVACIZUMAB, FLUOROURACIL, IRINOTECAN','BEVACIZUMAB, FLUOROURACIL, OXALIPLATIN'],
    'Pancreatic': ['FLUOROURACIL, OXALIPLATIN','FLUOROURACIL, IRINOTECAN, OXALIPLATIN',
        'FLUOROURACIL, IRINOTECAN','GEMCITABINE, PACLITAXEL','CAPECITABINE','GEMCITABINE'],
    'Prostate': ['DOCETAXEL','ABIRATERONE','BICALUTAMIDE','ENZALUTAMIDE'],
}

KM_COLORS = {
    2: ['#173E64','#E21C16'],
    3: ['#173E64','#7B7B7B','#E21C16'],
    4: ['#173E64','#5B9BD5','#FFA500','#E21C16'],
}
KM_LABELS = {
    2: ['Top 50%','Bottom 50%'],
    3: ['Top 33%','Middle 33%','Bottom 33%'],
    4: ['Top 25%','Q2 (50-75%)','Q3 (25-50%)','Bottom 25%'],
}

MIN_N_MSK = 200
MIN_PER_GROUP = 15


def norm_drug(d):
    if pd.isna(d): return None
    parts = [x.strip().upper() for x in d.split(',')]
    core = [x for x in parts if x not in SUPPORTIVE_DRUGS and 'INVESTIGATIONAL' not in x]
    return ', '.join(sorted(core)) if core else None


# ================================================================
# Cox regression helpers
# ================================================================
def compute_hr_ngroup(scores, times, events, n_groups, min_per_group=MIN_PER_GROUP):
    """Compute HR for top vs bottom group in n-group split."""
    mask = pd.notna(scores) & pd.notna(times) & pd.notna(events) & (times > 0)
    s, t, e = scores[mask].values, times[mask].values, events[mask].values
    if len(s) < 30 or e.sum() < 5:
        return None

    pcts = np.linspace(0, 100, n_groups + 1)
    thresholds = [np.percentile(s, p) for p in pcts]
    # Assign groups: 0=bottom, n_groups-1=top
    groups = np.digitize(s, thresholds[1:-1])
    # Top vs Bottom
    top = groups == (n_groups - 1)
    bottom = groups == 0

    if top.sum() < min_per_group or bottom.sum() < min_per_group:
        return None

    try:
        df = pd.DataFrame({'time': t[top | bottom], 'event': e[top | bottom],
                           'group': (top[top | bottom]).astype(int)})
        cph = CoxPHFitter()
        cph.fit(df, 'time', 'event')
        sm = cph.summary
        return {
            'HR': sm.loc['group', 'exp(coef)'],
            'CI_low': sm.loc['group', 'exp(coef) lower 95%'],
            'CI_high': sm.loc['group', 'exp(coef) upper 95%'],
            'p': sm.loc['group', 'p'],
            'n': len(df), 'n_top': int(top.sum()), 'n_bottom': int(bottom.sum()),
        }
    except:
        return None


# ================================================================
# Forest plot
# ================================================================
def plot_forest_multi(results_dict, labels, title, output_path):
    """Forest plot comparing 2/3/4-group splits."""
    methods = list(results_dict.keys())  # e.g. ['2-group','3-group','4-group']
    colors = {'2-group':'#173E64','3-group':'#E21C16','4-group':'#FFA500'}
    offsets = {m: -0.2 + i * 0.2 for i, m in enumerate(methods)}

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5 + 1)))
    y_pos = np.arange(len(labels))

    for method in methods:
        hrs, ci_lows, ci_highs, ys = [], [], [], []
        for i, label in enumerate(labels):
            if label in results_dict[method] and results_dict[method][label] is not None:
                d = results_dict[method][label]
                hrs.append(d['HR'])
                ci_lows.append(d['CI_low'])
                ci_highs.append(d['CI_high'])
                ys.append(y_pos[i] + offsets[method])

        if not hrs:
            continue
        hrs, ci_lows, ci_highs, ys = map(np.array, [hrs, ci_lows, ci_highs, ys])
        ax.errorbar(hrs, ys, xerr=[hrs - ci_lows, ci_highs - hrs],
                    fmt='o', color=colors.get(method, '#333'), markersize=5,
                    markeredgewidth=0.6, capsize=2, capthick=0.8, elinewidth=0.8,
                    label=method, zorder=3)

    ax.axvline(x=1.0, color='red', linewidth=0.8, linestyle='--', alpha=0.7, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Hazard Ratio', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.xaxis.grid(True, alpha=0.15)
    ax.legend(fontsize=8, loc='lower right', frameon=False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.replace('.pdf', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ================================================================
# KM curves
# ================================================================
def plot_km_ngroup(scores, times, events, n_groups, drug_name, cancer_name, dataset, os_label, min_pg=MIN_PER_GROUP):
    """Plot KM for n-group split."""
    mask = pd.notna(scores) & pd.notna(times) & pd.notna(events) & (times > 0)
    s, t, e = scores[mask].values, times[mask].values, events[mask].values
    if len(s) < 30:
        return

    pcts = np.linspace(0, 100, n_groups + 1)
    thresholds = [np.percentile(s, p) for p in pcts]
    groups = np.digitize(s, thresholds[1:-1])

    # Check all groups have enough samples
    for g in range(n_groups):
        if (groups == g).sum() < min_pg:
            return

    # Cox HR (top vs bottom)
    top = groups == (n_groups - 1)
    bottom = groups == 0
    try:
        df_cox = pd.DataFrame({'time': t[top | bottom], 'event': e[top | bottom],
                               'group': top[top | bottom].astype(int)})
        cph = CoxPHFitter()
        cph.fit(df_cox, 'time', 'event')
        hr = cph.summary.loc['group', 'exp(coef)']
        ci_lo = cph.summary.loc['group', 'exp(coef) lower 95%']
        ci_hi = cph.summary.loc['group', 'exp(coef) upper 95%']
        p = cph.summary.loc['group', 'p']
    except:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = KM_COLORS[n_groups]
    labels = KM_LABELS[n_groups]

    for g in range(n_groups - 1, -1, -1):  # top to bottom
        gmask = groups == g
        kmf = KaplanMeierFitter()
        kmf.fit(t[gmask], e[gmask])
        kmf.plot_survival_function(ax=ax, color=colors[n_groups - 1 - g], linewidth=1.5, ci_show=False)
        n_g = gmask.sum()
        ax.text(0.98, 0.98 - (n_groups - 1 - g) * 0.08,
                f'{labels[n_groups - 1 - g]}, n={n_g}',
                transform=ax.transAxes, ha='right', va='top', fontsize=6,
                color=colors[n_groups - 1 - g], fontweight='bold')

    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '50', '100'], fontsize=8)
    tick_step = 50
    x_max = max(t) * 1.05
    ax.set_xlim(0, x_max)
    ax.set_xticks(list(range(0, int(x_max) + tick_step, tick_step)))
    ax.set_xlabel('Months elapsed', fontsize=9)
    ax.set_ylabel('Percent survival (OS)', fontsize=9)

    drug_disp = drug_name.title() if drug_name.isupper() and len(drug_name) > 3 else drug_name
    ax.set_title(f'{drug_disp} {cancer_name}', fontsize=10, fontweight='bold')

    p_str = f'p={p:.4f}' if p >= 0.00005 else 'p<0.0001'
    ax.text(0.02, 0.02, f'{p_str}\nHR: {hr:.2f}\n(95% CI: {ci_lo:.2f} - {ci_hi:.2f})',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=7, fontweight='bold',
            bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none'))

    ax.get_legend().remove()
    plt.tight_layout()

    safe = re.sub(r'[^\w]', '_', f'{dataset}_{cancer_name}_{drug_name}')
    fname = f'{KM_DIR}/km_{n_groups}group_{os_label}_{safe}'
    fig.savefig(f'{fname}.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save data CSV (GraphPad format)
    csv_data = pd.DataFrame({'Months elapsed': t.astype(int)})
    for g in range(n_groups):
        gmask = groups == g
        col = labels[n_groups - 1 - g]
        vals = pd.Series([''] * len(t))
        vals[gmask] = e[gmask].astype(int).astype(str)
        csv_data[col] = vals.values
    csv_data.to_csv(f'{fname}_data.csv', index=False)


# ================================================================
# Data loading
# ================================================================
def load_msk():
    fm = pd.read_csv(FEATURE_MATRIX_CSV)
    fm['core_regimen'] = fm['drugs'].apply(norm_drug)
    ra = pd.read_csv(RESPONSE_ARRAY_CSV)
    fm = fm.merge(ra, on='sample_idx', how='left')
    clinical = pd.read_csv(CLINICAL_SAMPLE_FILE, sep='\t', comment='#')
    pc = clinical[['PATIENT_ID', 'CANCER_TYPE']].drop_duplicates()
    fm = fm.merge(pc, left_on='patient_id', right_on='PATIENT_ID', how='left')
    fm['cancer_cat'] = fm['CANCER_TYPE'].map(CANCER_MAP_CLINICAL)
    return fm


# ================================================================
# Main analysis
# ================================================================
def analyze_dataset(data, score_col, time_col, event_col, dataset_name, os_label,
                    cancer_order, min_n_cancer=80, min_n_regimen=80):
    """Run full multi-group analysis on one dataset."""
    print(f"\n{'='*60}")
    print(f"{dataset_name} — {os_label}")
    print(f"{'='*60}")

    all_forest_rows = []

    # 1. Cancer-type level
    print("\n--- Cancer-type level ---")
    results_cancer = {m: {} for m in ['2-group', '3-group', '4-group']}
    cancer_labels = []

    for cancer in cancer_order:
        sub = data[data['cancer_cat'] == cancer]
        mask = sub[time_col].notna() & (sub[time_col] > 0) & sub[score_col].notna()
        valid = sub[mask]
        if len(valid) < min_n_cancer:
            continue

        label = CANCER_LABELS.get(cancer, cancer)
        cancer_labels.append(label)

        for ng, method in [(2, '2-group'), (3, '3-group'), (4, '4-group')]:
            hr = compute_hr_ngroup(valid[score_col], valid[time_col], valid[event_col], ng)
            results_cancer[method][label] = hr
            if hr:
                all_forest_rows.append({
                    'dataset': dataset_name, 'os': os_label, 'level': 'cancer',
                    'cancer': cancer, 'regimen': '', 'method': method,
                    'HR': hr['HR'], 'CI_low': hr['CI_low'], 'CI_high': hr['CI_high'],
                    'p': hr['p'], 'n': hr['n'],
                })

            if ng in [3, 4]:
                plot_km_ngroup(valid[score_col], valid[time_col], valid[event_col],
                               ng, 'All Regimens', label, dataset_name, os_label)

        def _fmt(d):
            return f"{d['HR']:.3f}" if d else 'N/A'
        print(f"  {cancer:12s} "
              f"2g={_fmt(results_cancer['2-group'].get(label))} "
              f"3g={_fmt(results_cancer['3-group'].get(label))} "
              f"4g={_fmt(results_cancer['4-group'].get(label))}")

    if cancer_labels:
        plot_forest_multi(results_cancer, cancer_labels,
                          f'{dataset_name} Cancer-Type HR ({os_label})',
                          f'{FOREST_DIR}/forest_cancer_{dataset_name}_{os_label}.pdf')

    # 2. Per-regimen level
    print("\n--- Per-regimen level ---")
    for cancer in cancer_order:
        regs = REGIMEN_ORDER.get(cancer, [])
        sub = data[data['cancer_cat'] == cancer]
        results_reg = {m: {} for m in ['2-group', '3-group', '4-group']}
        reg_labels = []

        for reg in regs:
            rsub = sub[sub['core_regimen'] == reg]
            mask = rsub[time_col].notna() & (rsub[time_col] > 0) & rsub[score_col].notna()
            valid = rsub[mask]
            if len(valid) < min_n_regimen:
                continue

            reg_labels.append(reg)

            for ng, method in [(2, '2-group'), (3, '3-group'), (4, '4-group')]:
                hr = compute_hr_ngroup(valid[score_col], valid[time_col], valid[event_col], ng)
                results_reg[method][reg] = hr
                if hr:
                    all_forest_rows.append({
                        'dataset': dataset_name, 'os': os_label, 'level': 'regimen',
                        'cancer': cancer, 'regimen': reg, 'method': method,
                        'HR': hr['HR'], 'CI_low': hr['CI_low'], 'CI_high': hr['CI_high'],
                        'p': hr['p'], 'n': hr['n'],
                    })

                if ng in [3, 4]:
                    plot_km_ngroup(valid[score_col], valid[time_col], valid[event_col],
                                   ng, reg, CANCER_LABELS.get(cancer, cancer), dataset_name, os_label,
                                   min_pg=10)

        if reg_labels:
            plot_forest_multi(results_reg, reg_labels,
                              f'{CANCER_LABELS.get(cancer,cancer)} ({dataset_name}, {os_label})',
                              f'{FOREST_DIR}/forest_regimen_{dataset_name}_{cancer}_{os_label}.pdf')
            print(f"  {cancer}: {len(reg_labels)} regimens")

    return all_forest_rows


def main():
    all_rows = []
    msk = load_msk()
    for os_label, tc, ec in [('RegOS', '_os_reg_months', '_os_reg_status'),
                              ('DxOS', '_os_dx_months', '_os_dx_status')]:
        rows = analyze_dataset(msk, 'score_ra', tc, ec, 'MSK', os_label,
                               CANCER_ORDER_MSK, min_n_cancer=200, min_n_regimen=200)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(f'{DATA_DIR}/all_multi_group_results.csv', index=False)
    print(f"\nSaved: {DATA_DIR}/all_multi_group_results.csv ({len(df)} rows)")
    print("All done!")


if __name__ == '__main__':
    main()
