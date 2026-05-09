#!/usr/bin/env python3
import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter, CoxPHFitter
from config import (
    OUTPUT_DIR as _CFG_OUTPUT_DIR,
    FEATURE_MATRIX_CSV,
    RESPONSE_ARRAY_CSV,
)
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.environ.get('RESPONSE_RATE_OUTPUT_DIR', os.path.join(_CFG_OUTPUT_DIR, 'response_rate_calibrated'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUPPORTIVE_DRUGS = {
    'ZOLEDRONIC ACID','PAMIDRONATE','DENOSUMAB','LEUPROLIDE','GOSERELIN',
    'DEGARELIX','LEUCOVORIN','PREDNISONE','INVESTIGATIONAL','MEGESTROL',
}

def norm_drug(d):
    if pd.isna(d): return None
    parts = [x.strip().upper() for x in d.split(',')]
    core = [x for x in parts if x not in SUPPORTIVE_DRUGS and 'INVESTIGATIONAL' not in x]
    return ', '.join(sorted(core)) if core else None

REGIMENS = [
    ('Gemcitabine',             'Pancreatic',  'GEMCITABINE',                             5,  '~5%'),
    ('5-FU/LV',                 'Colorectal',  'FLUOROURACIL',                           15,  '~10-20%'),
    ('Nivolumab (2L)',          'NSCLC',       'NIVOLUMAB',                              20,  '~20%'),
    ('Carboplatin/Paclitaxel',  'NSCLC',       'CARBOPLATIN, PACLITAXEL',                20,  '~19-25%'),
    ('FOLFIRINOX',              'Pancreatic',  'FLUOROURACIL, IRINOTECAN, OXALIPLATIN',  30,  '~32%'),
    ('Tamoxifen',               'Breast',      'TAMOXIFEN',                              30,  '~30%'),
    ('Pembrolizumab',           'NSCLC',       'PEMBROLIZUMAB',                          45,  '~45%'),
    ('Bev + Chemo',             'Colorectal',  'BEVACIZUMAB, FLUOROURACIL, IRINOTECAN',  45,  '~45%'),
    ('AC',                      'Breast',      'CYCLOPHOSPHAMIDE, DOXORUBICIN',          55,  '~55%'),
    ('Enzalutamide',            'Prostate',    'ENZALUTAMIDE',                           59,  '~59%'),
]

# Load data
fm = pd.read_csv(FEATURE_MATRIX_CSV)
ra = pd.read_csv(RESPONSE_ARRAY_CSV)
fm = fm.merge(ra, on='sample_idx', how='left')
fm['core_regimen'] = fm['drugs'].apply(norm_drug)


def compute_hr(scores, times, events, top_pct, min_pg=15):
    mask = pd.notna(scores) & pd.notna(times) & pd.notna(events) & (times > 0)
    s, t, e = scores[mask].values, times[mask].values, events[mask].values
    if len(s) < 30 or e.sum() < 5:
        return None
    threshold = np.percentile(s, 100 - top_pct)
    high = s >= threshold
    if high.sum() < min_pg or (~high).sum() < min_pg:
        return None
    try:
        df = pd.DataFrame({'time': t, 'event': e, 'group': high.astype(int)})
        cph = CoxPHFitter()
        cph.fit(df, 'time', 'event')
        sm = cph.summary
        return {
            'HR': sm.loc['group', 'exp(coef)'],
            'CI_low': sm.loc['group', 'exp(coef) lower 95%'],
            'CI_high': sm.loc['group', 'exp(coef) upper 95%'],
            'p': sm.loc['group', 'p'],
            'n': len(s), 'n_high': int(high.sum()), 'n_low': int((~high).sum()),
        }
    except:
        return None


# ============================================================
# Run analysis
# ============================================================
results = []
for display, cancer, regimen, top_pct, orr in REGIMENS:
    sub = fm[(fm['cancer_type'] == cancer) & (fm['core_regimen'] == regimen)]
    hr_cal = compute_hr(sub['score_ra'], sub['_os_reg_months'], sub['_os_reg_status'], top_pct)
    hr_med = compute_hr(sub['score_ra'], sub['_os_reg_months'], sub['_os_reg_status'], 50)

    row = {'display': display, 'cancer': cancer, 'regimen': regimen,
           'top_pct': top_pct, 'known_orr': orr, 'n': len(sub)}
    if hr_cal:
        for k, v in hr_cal.items():
            row['cal_' + k] = v
    if hr_med:
        for k, v in hr_med.items():
            row['med_' + k] = v
    results.append(row)

    if hr_cal:
        p_val = hr_cal['p']
        p_str = 'p<0.001' if p_val < 0.001 else 'p={:.3f}'.format(p_val)
        cal_str = 'HR={:.3f} [{:.3f}-{:.3f}] {}'.format(
            hr_cal['HR'], hr_cal['CI_low'], hr_cal['CI_high'], p_str)
    else:
        cal_str = 'N/A'
    med_str = 'HR={:.3f}'.format(hr_med['HR']) if hr_med else 'N/A'
    print('{:25s} {:12s} Top{:3d}% (ORR{:>10s})  n={:5d}  Cal: {}  Med: {}'.format(
        display, cancer, top_pct, orr, len(sub), cal_str, med_str))

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_DIR, 'response_rate_calibrated_results.csv'), index=False)

# ============================================================
# Forest plot: Calibrated vs Median
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
valid = [r for r in results if 'cal_HR' in r]
n_valid = len(valid)
y_positions = np.arange(n_valid)

for i, row in enumerate(valid):
    y = n_valid - 1 - i
    # Calibrated (red)
    ax.plot(row['cal_HR'], y + 0.12, 'o', color='#E21C16', markersize=7, zorder=3)
    ax.plot([row['cal_CI_low'], row['cal_CI_high']], [y + 0.12]*2,
            '-', color='#E21C16', linewidth=1.8, zorder=2)
    # Median (blue)
    if 'med_HR' in row:
        ax.plot(row['med_HR'], y - 0.12, 'o', color='#173E64', markersize=7, zorder=3)
        ax.plot([row['med_CI_low'], row['med_CI_high']], [y - 0.12]*2,
                '-', color='#173E64', linewidth=1.8, zorder=2)

ax.axvline(x=1, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

labels = []
for row in valid:
    p_val = row['cal_p']
    p_str = 'p<0.001' if p_val < 0.001 else 'p={:.3f}'.format(p_val)
    labels.append('{} ({})\n  Top {}% (ORR{})  HR={:.2f} [{:.2f}-{:.2f}] {}  n={}+{}'.format(
        row['display'], row['cancer'], row['top_pct'], row['known_orr'],
        row['cal_HR'], row['cal_CI_low'], row['cal_CI_high'], p_str,
        row['cal_n_high'], row['cal_n_low']))

ax.set_yticks(y_positions)
ax.set_yticklabels(labels[::-1], fontsize=7)
ax.set_xlabel('Hazard Ratio', fontsize=11, fontweight='bold')
ax.set_title('Response-Rate-Calibrated Threshold vs Median Split (MSK, RegOS)',
             fontsize=12, fontweight='bold')
for spine in ax.spines.values():
    spine.set_visible(True)
ax.xaxis.grid(True, alpha=0.15)

handles = [
    Line2D([0],[0], marker='o', color='#E21C16', linestyle='-', linewidth=1.8,
           markersize=7, label='ORR-Calibrated'),
    Line2D([0],[0], marker='o', color='#173E64', linestyle='-', linewidth=1.8,
           markersize=7, label='Median Split'),
]
ax.legend(handles=handles, loc='lower right', fontsize=9, framealpha=0.9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'forest_calibrated_vs_median.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'forest_calibrated_vs_median.svg'), bbox_inches='tight')
plt.close()
print('\nSaved forest plot')

# ============================================================
# KM panel (2x5)
# ============================================================
COLOR_TOP = '#173E64'
COLOR_BOT = '#E21C16'

fig_km, axes = plt.subplots(2, 5, figsize=(14, 6))
axes_flat = axes.flatten()

for idx, (display, cancer, regimen, top_pct, orr) in enumerate(REGIMENS):
    ax = axes_flat[idx]
    sub = fm[(fm['cancer_type'] == cancer) & (fm['core_regimen'] == regimen)]
    mask = sub['score_ra'].notna() & sub['_os_reg_months'].notna() & sub['_os_reg_status'].notna() & (sub['_os_reg_months'] > 0)
    valid = sub[mask]
    if len(valid) < 30:
        ax.axis('off')
        continue

    s = valid['score_ra'].values
    t = valid['_os_reg_months'].values
    e = valid['_os_reg_status'].values

    threshold = np.percentile(s, 100 - top_pct)
    high = s >= threshold

    kmf_h = KaplanMeierFitter()
    kmf_l = KaplanMeierFitter()
    kmf_h.fit(t[high], e[high])
    kmf_l.fit(t[~high], e[~high])
    kmf_h.plot_survival_function(ax=ax, color=COLOR_TOP, linewidth=1.5, ci_show=False)
    kmf_l.plot_survival_function(ax=ax, color=COLOR_BOT, linewidth=1.5, ci_show=False)

    try:
        df = pd.DataFrame({'time': t, 'event': e, 'group': high.astype(int)})
        cph = CoxPHFitter()
        cph.fit(df, 'time', 'event')
        hr = cph.summary.loc['group', 'exp(coef)']
        ci_lo = cph.summary.loc['group', 'exp(coef) lower 95%']
        ci_hi = cph.summary.loc['group', 'exp(coef) upper 95%']
        p = cph.summary.loc['group', 'p']
    except:
        hr, ci_lo, ci_hi, p = np.nan, np.nan, np.nan, np.nan

    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '50', '100'], fontsize=6)
    x_max = max(t) * 1.05
    ax.set_xlim(0, x_max)
    ax.set_xticks(list(range(0, int(x_max) + 50, 50)))
    ax.tick_params(axis='x', labelsize=6)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('{}\n({})'.format(display, cancer), fontsize=7, fontweight='bold', pad=2)

    n_h, n_l = int(high.sum()), int((~high).sum())
    ax.text(0.98, 0.98, 'Top {}%, n={}'.format(top_pct, n_h),
            transform=ax.transAxes, ha='right', va='top', fontsize=5,
            color=COLOR_TOP, fontweight='bold')
    ax.text(0.98, 0.88, 'Bottom {}%, n={}'.format(100-top_pct, n_l),
            transform=ax.transAxes, ha='right', va='top', fontsize=5,
            color=COLOR_BOT, fontweight='bold')

    p_str = 'p={:.4f}'.format(p) if p >= 0.00005 else 'p<0.0001'
    ax.text(0.02, 0.02,
            '{}\nHR: {:.2f}\n(95% CI: {:.2f} - {:.2f})\nORR: {}'.format(p_str, hr, ci_lo, ci_hi, orr),
            transform=ax.transAxes, ha='left', va='bottom', fontsize=5, fontweight='bold',
            bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none'))
    ax.get_legend().remove()

fig_km.text(0.5, 0.005, 'Months elapsed', ha='center', fontsize=10)
fig_km.text(0.005, 0.5, 'Percent survival (OS)', va='center', rotation='vertical', fontsize=10)
fig_km.suptitle('Response-Rate-Calibrated KM Curves (MSK, RegOS)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0.02, 0.02, 1, 1])
fig_km.savefig(os.path.join(OUTPUT_DIR, 'km_calibrated_all.pdf'), dpi=300, bbox_inches='tight')
fig_km.savefig(os.path.join(OUTPUT_DIR, 'km_calibrated_all.svg'), bbox_inches='tight')
plt.close()
print('Saved KM panel')
print('\n[DONE]')
