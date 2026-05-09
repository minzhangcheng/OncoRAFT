#!/usr/bin/env python3
"""Word cloud from OncoRAFT reasoning text.

Per-cancer panels rendered with raw term frequencies (whitelisted gene,
clinical, drug, and biomarker terms only).
"""

import json, os, re, logging
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WHITELIST_TERMS = {
    'gene_terms': [
        # Curated cancer-gene whitelist (HUGO symbols)
        "AKT1", "ALK", "AMER1", "ANKRD11", "APC", "AR", "ARID1A", "ARID1B",
        "ARID2", "ASXL1", "ASXL2", "ATM", "ATR", "ATRX", "AXIN1", "AXIN2",
        "AXL", "BAP1", "BCOR", "BRAF", "BRCA1", "BRCA2", "BRD4", "CARD11",
        "CBFB", "CDH1", "CDK12", "CDKN2A", "CIC", "CREBBP", "CTCF", "CTNNB1",
        "DDR2", "DICER1", "DNMT1", "DNMT3A", "DOT1L", "DROSHA", "EGFR", "EP300",
        "EPHA3", "EPHA5", "EPHA7", "EPHB1", "ERBB2", "ERBB3", "ERBB4", "ESR1",
        "FANCA", "FAT1", "FBXW7", "FGFR4", "FLT1", "FLT3", "FLT4", "FOXA1",
        "FOXP1", "GATA3", "GLI1", "GNAS", "GRIN2A", "HGF", "IGF1R", "IKZF1",
        "INPP4B", "INPPL1", "IRS1", "IRS2", "JAK1", "JAK2", "JAK3", "KDM5A",
        "KDM5C", "KDM6A", "KDR", "KEAP1", "KIT", "KMT2A", "KMT2B", "KMT2C",
        "KMT2D", "KRAS", "LATS1", "LATS2", "MAP2K4", "MAP3K1", "MDC1", "MED12",
        "MEN1", "MET", "MGA", "MLL3", "MSH3", "MSH6", "MTOR", "NCOR1", "NF1",
        "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "NRAS", "NSD1", "NTRK1", "NTRK3",
        "PAK7", "PALB2", "PBRM1", "PDGFRA", "PDGFRB", "PGR", "PIK3C2G", "PIK3CA",
        "PIK3CG", "PIK3R1", "PLCG2", "POLD1", "POLE", "PREX2", "PRKD1", "PTCH1",
        "PTEN", "PTPRD", "PTPRS", "PTPRT", "RASA1", "RB1", "RBM10", "RECQL4",
        "RET", "RICTOR", "RNF43", "ROS1", "RPTOR", "RUNX1", "SETD2", "SF3B1",
        "SLX4", "SMAD2", "SMAD3", "SMAD4", "SMARCA4", "SMO", "SOX17", "SOX9",
        "SPEN", "SPOP", "STAG2", "STK11", "TBX3", "TCF7L2", "TERT", "TET1",
        "TET2", "TGFBR2", "TP53", "TP53BP1", "TSC1", "TSC2", "WT1", "ZFHX3",
    ],
    'clinical_terms': [
        "ADENOCARCINOMA", "STAGE", "STAGE 4", "STAGE 1-3", "SMOKING", "MSS", "MSI",
        "SMOKER", "METASTASIS", "CEA", "CA_15-3", "CA15-3", "CA_19-9", "CA19-9",
        "GLEASON SCORE", "PD-L1", "PERFORMANCE STATUS", "PERFORMANCE", "PROGNOSIS",
        "TOXICITY", "SURVIVAL", "PROGRESSION", "RECURRENCE", "BIOMARKER", "HISTOLOGY",
        "GRADE", "DIFFERENTIATION", "INVASIVE", "NODAL", "LYMPHATIC",
        "ADJUVANT", "NEOADJUVANT", "PALLIATIVE", "COMBINATION", "MONOTHERAPY",
        "FIRST-LINE", "SECOND-LINE", "REFRACTORY", "RELAPSED",
        "MAINTENANCE", "SALVAGE", "HORMONE", "IMMUNOTHERAPIES", "IMMUNOTHERAPY",
        "HORMONAL", "CHEMORESISTANCE",
        # Biomarkers & pathway names (from text freq scan)
        "TMB", "HER2", "MMR", "TYMS", "DPYD", "DPD", "UGT1A1",
        # Pathway abbreviations
        "PI3K", "MAPK", "AKT", "WNT", "VEGF", "RAS", "MEK", "ERK", "RAF",
        "FGFR", "VEGFR",
    ],
}
TERM_MAPPING = {
    'IMMUNOTHERAPY': ['IMMUNOTHERAPY', 'IMMUNOTHERAPIES'],
    'HORMONE': ['HORMONE', 'HORMONAL'],
    'CA19-9': ['CA_19-9', 'CA19-9'], 'CA15-3': ['CA_15-3', 'CA15-3'],
    'SMOKING': ['SMOKING', 'SMOKER'],
}

_ALL_TERMS, _TERM_TO_CANONICAL = [], {}
for _terms in WHITELIST_TERMS.values():
    for _t in _terms:
        t_upper = _t.upper()
        _ALL_TERMS.append(t_upper)
        canonical = t_upper
        for can, variants in TERM_MAPPING.items():
            if t_upper in [v.upper() for v in variants]:
                canonical = can; break
        _TERM_TO_CANONICAL[t_upper] = canonical
_ALL_TERMS.sort(key=len, reverse=True)
_COMBINED_PATTERN = re.compile(r'\b(?:' + '|'.join(re.escape(t) for t in _ALL_TERMS) + r')\b')

def clean_text(text):
    if not text or pd.isna(text): return ""
    text = re.sub(r'^\d+\.?\d*\s*', '', str(text))
    text = re.sub(r'(0\.0\.0\.0.*)', '', text)
    return re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', text)).strip()

def extract_terms(text):
    counts = defaultdict(int)
    for m in _COMBINED_PATTERN.finditer(text.upper()):
        counts[_TERM_TO_CANONICAL[m.group()]] += 1
    return counts

def load_data():
    from config import GENERATED_JSONL, FEATURE_MATRIX_CSV
    jsonl = GENERATED_JSONL
    fm_path = FEATURE_MATRIX_CSV
    records = []
    with open(jsonl, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try: records.append(json.loads(line))
                except: pass
    inf_df = pd.DataFrame(records)
    fm = pd.read_csv(fm_path, usecols=['patient_id', 'cancer_type'])
    ct_map = {'NSCLC':'Non-Small Cell Lung Cancer','Breast':'Breast Cancer',
              'Colorectal':'Colorectal Cancer','Prostate':'Prostate Cancer',
              'Pancreatic':'Pancreatic Cancer'}
    pct = fm.drop_duplicates(subset='patient_id').copy()
    pct['CANCER_TYPE'] = pct['cancer_type'].map(ct_map)
    return inf_df.merge(pct[['patient_id','CANCER_TYPE']], on='patient_id', how='inner')

CT = ['Non-Small Cell Lung Cancer','Breast Cancer','Colorectal Cancer',
      'Prostate Cancer','Pancreatic Cancer']

def main():
    merged = load_data()
    logger.info(f"Matched {len(merged)} samples")
    from config import OUTPUT_DIR as _CFG_OUTPUT_DIR
    out = os.environ.get('WORDCLOUD_OUTPUT_DIR', os.path.join(_CFG_OUTPUT_DIR, 'wordclouds'))
    os.makedirs(out, exist_ok=True)

    all_data = merged[merged['CANCER_TYPE'].isin(CT)]
    overall = extract_terms(' '.join(clean_text(t) for t in all_data['generated_text'].dropna()))
    per_type, per_n = {}, {}
    for ct in CT:
        sub = merged[merged['CANCER_TYPE'] == ct]
        per_n[ct] = len(sub)
        texts = [clean_text(t) for t in sub['generated_text'].dropna()]
        per_type[ct] = extract_terms(' '.join(texts)) if texts else {}

    _, axes = plt.subplots(2, 3, figsize=(20, 12)); axes = axes.flatten()
    for i, ct in enumerate(CT):
        if not per_type[ct]:
            axes[i].text(0.5,0.5,'No data',ha='center',va='center',transform=axes[i].transAxes)
            axes[i].set_title(ct); axes[i].axis('off'); continue
        scored = per_type[ct]
        wc = WordCloud(width=400,height=300,background_color='white',max_words=50,
                       colormap='Set3',relative_scaling=0.5,min_font_size=6,
                       prefer_horizontal=0.9).generate_from_frequencies(scored)
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].set_title(f'{ct}\n({per_n[ct]} samples)', fontsize=12); axes[i].axis('off')
        top5 = sorted(scored.items(), key=lambda x:x[1], reverse=True)[:5]
        logger.info(f"  {ct} top: {[(t,f'{s:.0f}') for t,s in top5]}")

    if overall:
        scored_all = overall
        wc_all = WordCloud(width=400,height=300,background_color='white',max_words=50,
                           colormap='viridis',relative_scaling=0.5,min_font_size=8,
                           prefer_horizontal=0.9).generate_from_frequencies(scored_all)
        axes[5].imshow(wc_all, interpolation='bilinear')
        axes[5].set_title(f'All Target Cancer Types\n({len(all_data)} samples)', fontsize=12)
        axes[5].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out,'wordclouds_nodrug.png'), bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig(os.path.join(out,'wordclouds_nodrug.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    logger.info("Done: wordclouds_nodrug")

if __name__ == "__main__": main()
