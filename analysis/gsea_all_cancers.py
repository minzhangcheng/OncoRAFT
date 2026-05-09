#!/usr/bin/env python3
"""
GSEA analysis per cancer cohort plus pan-cancer.
Step 1: Extract gene frequencies from OncoRAFT generated reasoning text.
Step 2: Run GSEA with cancer-type-specific gene sets + permutation test.
Step 3: Generate overlapping GSEA plots (3-panel: curves, hit positions, ranking).
"""
import os, re, json, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import false_discovery_control
warnings.filterwarnings('ignore')
np.random.seed(42)

from config import (
    OUTPUT_DIR as _CFG_OUTPUT_DIR,
    FEATURE_MATRIX_CSV,
    RESPONSE_ARRAY_CSV,
    ONCORAFT_SCORES_CSV,
    GENERATED_JSONL,
)

OUTPUT_DIR = os.environ.get('GSEA_OUTPUT_DIR', os.path.join(_CFG_OUTPUT_DIR, 'gsea'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Step 1: Gene frequency extraction
# ============================================================
# Canonical gene list (HUGO symbols from MSK-IMPACT panel + common oncogenes)
GENE_LIST = [
    'TP53','KRAS','PIK3CA','APC','EGFR','BRAF','PTEN','ATM','BRCA2','BRCA1',
    'CDKN2A','RB1','NF1','STK11','KEAP1','SMAD4','ARID1A','KMT2D','NOTCH1',
    'ERBB2','MYC','CCND1','CDK4','MDM2','FGFR1','FGFR2','FGFR3','MET','ALK',
    'ROS1','RET','NTRK1','NTRK2','NTRK3','ESR1','AR','GATA3','FOXA1','MAP2K1',
    'MAP2K2','NRAS','HRAS','RAF1','AKT1','MTOR','TSC1','TSC2','FBXW7','CTNNB1',
    'KIT','PDGFRA','PDGFRB','VHL','MLH1','MSH2','MSH6','PMS2','CHEK2','PALB2',
    'RAD51','BRIP1','SPOP','CHD1','ERG','TMPRSS2','CDH1','AXIN1','AXIN2',
    'TGFBR2','SMAD2','SMAD3','POLE','POLD1','IDH1','IDH2','FGFR4','ERBB3',
    'ERBB4','SRC','JAK1','JAK2','STAT1','STAT3','VEGFA','KDR','FLT1',
    'PGR','TFF1','GREB1','TBX3','RUNX1','MAP3K1','NKX3-1','KLK3',
    'PARP1','BAP1','NFE2L2','CCNE1','CDK6','CDKN1A','CDKN1B','CDKN2B',
    'E2F1','MDM4','BCL2','BAX','MYCN','FOS','JUN','ELK1',
    'NCOA1','NCOA2','NCOA3','EP300','CREBBP','SOS1','GRB2','PTPN11',
    'RPTOR','RICTOR','PDPK1','GSK3B','FOXO1','FOXO3','RNF43',
    'SMURF1','SKI','SKIL','MMP2','MMP9','SNAI1','TWIST1','ZEB1','VIM','FN1',
    'PDCD1','CD274','CTLA4','LAG3','TIGIT','B2M','IDO1',
    'HIF1A','LDHA','PKM','AURKA','WEE1','PLK1',
]

def extract_gene_frequencies(texts, cancer_type=None):
    """Extract gene mention frequencies from reasoning texts."""
    gene_pattern = {g: re.compile(r'\b' + re.escape(g) + r'\b', re.IGNORECASE) for g in GENE_LIST}
    freq = {g: 0 for g in GENE_LIST}
    for text in texts:
        if not isinstance(text, str):
            continue
        for gene, pattern in gene_pattern.items():
            if pattern.search(text):
                freq[gene] += 1
    df = pd.DataFrame([{'Gene': g, 'Frequency': c} for g, c in freq.items()])
    df = df[df['Frequency'] > 0].sort_values('Frequency', ascending=False).reset_index(drop=True)
    return df


def load_texts_by_cancer():
    """Load generated texts and split by cancer type."""
    fm = pd.read_csv(FEATURE_MATRIX_CSV,
                     usecols=['patient_id', 'cancer_type', 'sample_idx'])
    mapped = pd.read_csv(RESPONSE_ARRAY_CSV)
    fm = fm.merge(mapped, on='sample_idx', how='left')

    texts_by_cancer = {c: [] for c in ['Breast', 'NSCLC', 'Colorectal', 'Pancreatic', 'Prostate']}
    all_texts = []

    # Read generated texts - they are in order matching ONCORAFT_SCORES_CSV
    oc = pd.read_csv(ONCORAFT_SCORES_CSV)
    with open(GENERATED_JSONL) as f:
        gen_texts = [json.loads(line) for line in f]

    # Build sample_id -> generated_text mapping
    # oncoraft_msk_scores and msk_all_generated.jsonl are in the same order
    for i, entry in enumerate(gen_texts):
        text = entry.get('generated_text', '')
        pid = entry.get('patient_id', oc.iloc[i]['patient_id'] if i < len(oc) else None)
        all_texts.append(text)

    # Map via RESPONSE_MAPPED: row i in gen_texts -> sample_idx in mapped
    for i, row in mapped.iterrows():
        sidx = row['sample_idx']
        fm_row = fm[fm['sample_idx'] == sidx]
        if len(fm_row) > 0:
            cancer = fm_row.iloc[0]['cancer_type']
            if cancer in texts_by_cancer and i < len(all_texts):
                texts_by_cancer[cancer].append(all_texts[i])

    return texts_by_cancer, all_texts


# ============================================================
# Step 2: Gene set definitions per cancer type
# ============================================================
def get_gene_sets(cancer_type):
    """Return cancer-type-specific gene sets (matching original GPU server definitions)."""
    if cancer_type == 'Breast':
        return {
            'BREAST_CANCER_DRIVERS': {
                'genes': ['ESR1','PGR','ERBB2','PIK3CA','TP53','CDH1','BRCA1','BRCA2',
                          'GATA3','FOXA1','TBX3','RUNX1','MAP3K1','CCND1','MYC','PTEN',
                          'AKT1','MTOR','AR','KMT2C','NF1','ARID1A'],
                'display_name': 'Breast Cancer Drivers', 'color': '#d73027',
            },
            'ONCOGENES': {
                'genes': ['MYC','MYCN','MYCL','ERBB2','EGFR','KRAS','NRAS','HRAS','PIK3CA',
                          'AKT1','AKT2','MTOR','MDM2','MDM4','CCND1','CDK4','CDK6','ALK',
                          'ROS1','MET','FGFR1','FGFR2'],
                'display_name': 'Oncogenes', 'color': '#f46d43',
            },
            'TUMOR_SUPPRESSOR_GENES': {
                'genes': ['TP53','RB1','PTEN','APC','BRCA1','BRCA2','VHL','NF1','NF2',
                          'CDKN2A','CDKN1A','CDKN1B','ATM','CHEK2','MLH1','MSH2','MSH6',
                          'STK11','SMAD4','FBXW7','ARID1A','BAP1'],
                'display_name': 'Tumor Suppressor Genes', 'color': '#fdae61',
            },
            'HORMONE_RECEPTOR_SIGNALING': {
                'genes': ['ESR1','ESR2','PGR','AR','GREB1','TFF1','FOXA1','GATA3','TBX3',
                          'NCOA1','NCOA2','NCOA3','EP300','CREBBP','KDM1A','KDM4B'],
                'display_name': 'Hormone Receptor Signaling', 'color': '#abd9e9',
            },
            'PI3K_AKT_MTOR_COMPREHENSIVE': {
                'genes': ['PIK3CA','PIK3CB','PIK3CD','PIK3CG','PIK3R1','PIK3R2','PIK3R3',
                          'AKT1','AKT2','AKT3','MTOR','RPTOR','RICTOR','PTEN','TSC1','TSC2',
                          'RHEB','PDPK1','GSK3B','FOXO1','FOXO3'],
                'display_name': 'PI3K/AKT/mTOR Signaling', 'color': '#2166ac',
            },
        }
    elif cancer_type == 'NSCLC':
        return {
            'NSCLC_DRIVER_GENES': {
                'genes': ['EGFR','KRAS','ALK','ROS1','BRAF','MET','RET','NTRK1','NTRK2','NTRK3',
                          'TP53','MDM2','MDM4','CDKN1A','ATM','CHEK2',
                          'PIK3CA','AKT1','PTEN','MTOR','STK11','KEAP1','NF1',
                          'RB1','CCND1','CDK4','CDK6','CDKN2A'],
                'display_name': 'NSCLC Driver Genes', 'color': '#d73027',
            },
            'EGFR_SIGNALING': {
                'genes': ['EGFR','ERBB2','ERBB3','ERBB4','EGF','AREG','EREG','BTC',
                          'GRB2','SOS1','SHC1','GAB1','PIK3CA','PIK3R1','AKT1','AKT2',
                          'MTOR','RPS6KB1','EIF4EBP1','GSK3B','FOXO1','FOXO3',
                          'SRC','PTK2','PTEN','INPP4B'],
                'display_name': 'EGFR Signaling', 'color': '#f46d43',
            },
            'RAS_RAF_MEK_ERK': {
                'genes': ['KRAS','NRAS','HRAS','BRAF','RAF1','ARAF',
                          'MAP2K1','MAP2K2','MAPK1','MAPK3','MAPK14',
                          'SOS1','GRB2','PTPN11','NF1','SPRED1','SPRED2',
                          'ELK1','FOS','JUN','MYC','CCND1'],
                'display_name': 'RAS/RAF/MEK/ERK', 'color': '#fdae61',
            },
            'ALK_ROS1_FUSIONS': {
                'genes': ['ALK','ROS1','RET','NTRK1','NTRK2','NTRK3',
                          'MET','FGFR1','FGFR2','FGFR3','PDGFRA','PDGFRB',
                          'KIT','CSF1R','FLT3','AXL','EPHA3','EPHB1',
                          'GRB2','SHC1','PIK3CA','AKT1','MTOR'],
                'display_name': 'ALK/ROS1/Fusion Oncogenes', 'color': '#fee08b',
            },
            'PI3K_AKT_MTOR': {
                'genes': ['PIK3CA','PIK3CB','PIK3CD','PIK3CG','PIK3R1','PIK3R2','PIK3R3',
                          'AKT1','AKT2','AKT3','PTEN','TSC1','TSC2','MTOR','RPTOR',
                          'RICTOR','PDPK1','GSK3A','GSK3B','FOXO1','FOXO3','FOXO4',
                          'RPS6KB1','RPS6KB2','EIF4EBP1','EIF4E'],
                'display_name': 'PI3K/AKT/mTOR', 'color': '#e0f3f8',
            },
            'P53_DNA_DAMAGE': {
                'genes': ['TP53','MDM2','MDM4','ATM','ATR','CHEK1','CHEK2',
                          'CDKN1A','CDKN1B','CDKN2A','BBC3','PMAIP1','BAX','BAK1',
                          'BRCA1','BRCA2','RAD51','RAD50','NBN','MRE11A',
                          'PARP1','XRCC1','XRCC4'],
                'display_name': 'p53/DNA Damage Response', 'color': '#abd9e9',
            },
            'KEAP1_NRF2_OXIDATIVE': {
                'genes': ['KEAP1','NFE2L2','CUL3','RBX1','NQO1','HMOX1','GCLC','GCLM',
                          'GPX1','GPX2','SOD1','SOD2','CAT','PRDX1','PRDX2','PRDX3',
                          'TXNRD1','GSR','G6PD','ALDH3A1','AKR1C1','GSTP1'],
                'display_name': 'KEAP1/NRF2/Oxidative Stress', 'color': '#74add1',
            },
            'CELL_CYCLE_CHECKPOINTS': {
                'genes': ['RB1','CCND1','CCND2','CCND3','CDK4','CDK6','CDK2','CCNE1',
                          'CDKN1A','CDKN1B','CDKN2A','CDKN2B','CDKN2C','CDKN2D',
                          'E2F1','E2F2','E2F3','CHEK1','CHEK2','WEE1','CDC25A',
                          'AURKA','AURKB','PLK1','CDC20','CDCA8'],
                'display_name': 'Cell Cycle Checkpoints', 'color': '#4575b4',
            },
            'IMMUNE_CHECKPOINT_PD1_PDL1': {
                'genes': ['PDCD1','CD274','PDCD1LG2','CTLA4','LAG3','TIGIT','HAVCR2',
                          'ICOS','ICOSLG','CD276','VTCN1','IDO1','IDO2','ADORA2A',
                          'CD80','CD86','B2M','HLA-A','HLA-B','HLA-C','TAP1','TAP2',
                          'IFNG','IFNGR1','JAK1','JAK2','STAT1'],
                'display_name': 'Immune Checkpoint/PD1-PDL1', 'color': '#313695',
            },
            'TUMOR_SUPPRESSOR_GENES': {
                'genes': ['TP53','RB1','PTEN','STK11','CDKN2A','CDKN1A','CDKN1B',
                          'APC','BRCA1','BRCA2','ATM','ATR','CHEK2','NF1','NF2',
                          'VHL','SMAD4','FBXW7','ARID1A','BAP1','KEAP1'],
                'display_name': 'Tumor Suppressor Genes', 'color': '#762a83',
            },
            'ANGIOGENESIS_VEGF': {
                'genes': ['VEGFA','VEGFB','VEGFC','VEGFD','KDR','FLT1','FLT4',
                          'PDGFA','PDGFB','PDGFRA','PDGFRB','FGF1','FGF2','FGFR1','FGFR2',
                          'ANG','ANGPT1','ANGPT2','TEK','HIF1A','EPAS1','VHL',
                          'ARNT','EGLN1','EGLN2','EGLN3'],
                'display_name': 'Angiogenesis/VEGF', 'color': '#5aae61',
            },
            'SMOKING_RELATED_GENES': {
                'genes': ['KEAP1','NFE2L2','AHR','CYP1A1','CYP1B1','GSTP1','GSTM1',
                          'NAT2','NQO1','EPHX1','ALDH2','ADH1B','ADH1C',
                          'TP53','CDKN2A','RB1','FHIT','RASSF1','CDKN1A',
                          'MLH1','MGMT','BRCA1'],
                'display_name': 'Smoking-Related Genes', 'color': '#1a9850',
            },
        }
    elif cancer_type == 'Colorectal':
        return {
            'CRC_DRIVER_GENES': {
                'genes': ['APC','CTNNB1','AXIN1','AXIN2','GSK3B','TCF7L2','LEF1',
                          'TP53','MDM2','MDM4','CDKN1A','ATM','CHEK2',
                          'KRAS','NRAS','HRAS','BRAF','MEK1','MAP2K1','MAP2K2',
                          'PIK3CA','AKT1','PTEN','MTOR',
                          'SMAD4','SMAD2','SMAD3','TGFBR1','TGFBR2',
                          'MLH1','MSH2','MSH6','PMS2','POLE','POLD1'],
                'display_name': 'CRC Driver Genes', 'color': '#d73027',
            },
            'WNT_APC_PATHWAY': {
                'genes': ['APC','CTNNB1','AXIN1','AXIN2','GSK3B','LRP5','LRP6',
                          'TCF7L2','TCF7','LEF1','MYC','CCND1','JUN','FOS',
                          'DKK1','DKK3','SFRP1','SFRP2','WIF1','ZNRF3','RNF43'],
                'display_name': 'Wnt/APC Signaling', 'color': '#f46d43',
            },
            'RAS_MAPK_SIGNALING': {
                'genes': ['KRAS','NRAS','HRAS','BRAF','RAF1','ARAF',
                          'MAP2K1','MAP2K2','MAPK1','MAPK3',
                          'SOS1','GRB2','PTPN11','NF1',
                          'EGFR','ERBB2','ERBB3','MET'],
                'display_name': 'RAS/MAPK Signaling', 'color': '#fdae61',
            },
            'DNA_MISMATCH_REPAIR': {
                'genes': ['MLH1','MSH2','MSH6','MSH3','PMS1','PMS2',
                          'PCNA','RFC1','RFC2','RFC3','RFC4','RFC5',
                          'RPA1','RPA2','RPA3','POLD1','POLE','POLD3','POLD4'],
                'display_name': 'DNA Mismatch Repair', 'color': '#abd9e9',
            },
            'TGF_BETA_SIGNALING': {
                'genes': ['TGFB1','TGFB2','TGFB3','TGFBR1','TGFBR2','TGFBR3',
                          'SMAD2','SMAD3','SMAD4','SMAD7','SMURF1','SMURF2',
                          'SKI','SKIL','BAMBI','LTBP1'],
                'display_name': 'TGF-\u03b2 Signaling', 'color': '#74add1',
            },
            'PI3K_AKT_PATHWAY': {
                'genes': ['PIK3CA','PIK3CB','PIK3CD','PIK3CG','PIK3R1','PIK3R2','PIK3R3',
                          'AKT1','AKT2','AKT3','PTEN','TSC1','TSC2','MTOR','RPTOR',
                          'RICTOR','PDPK1','GSK3B','FOXO1','FOXO3'],
                'display_name': 'PI3K/AKT Pathway', 'color': '#4575b4',
            },
            'TUMOR_SUPPRESSOR_GENES': {
                'genes': ['TP53','APC','PTEN','SMAD4','FBXW7','ATM','BRCA1','BRCA2',
                          'RB1','CDKN2A','CDKN1A','CDKN1B','VHL','NF1','NF2',
                          'STK11','CHEK2','MLH1','MSH2','ARID1A','BAP1'],
                'display_name': 'Tumor Suppressor Genes', 'color': '#313695',
            },
            'ONCOGENES': {
                'genes': ['KRAS','NRAS','BRAF','PIK3CA','MYC','MYCN','MYCL',
                          'EGFR','ERBB2','MET','FGFR1','FGFR2','ALK','ROS1',
                          'MDM2','MDM4','CCND1','CDK4','CDK6','E2F1','E2F3'],
                'display_name': 'Oncogenes', 'color': '#a50026',
            },
            'CRC_METASTASIS_GENES': {
                'genes': ['CDH1','CTNNB1','SNAI1','SNAI2','TWIST1','ZEB1','ZEB2',
                          'VIM','FN1','MMP2','MMP9','MMP7','VEGFA','VEGFR2',
                          'PDGFRA','PDGFRB','FGF2','FGFR1'],
                'display_name': 'CRC Metastasis Genes', 'color': '#762a83',
            },
            'IMMUNE_CHECKPOINT': {
                'genes': ['PDCD1','CD274','PDCD1LG2','CTLA4','LAG3','TIGIT','HAVCR2',
                          'ICOS','ICOSLG','CD276','VTCN1','IDO1','IDO2','CD80','CD86',
                          'B2M','HLA-A','HLA-B','HLA-C'],
                'display_name': 'Immune Checkpoint', 'color': '#5aae61',
            },
        }
    elif cancer_type == 'Pancreatic':
        return {
            'PANCREATIC_DRIVER_GENES': {
                'genes': ['KRAS','TP53','CDKN2A','SMAD4',
                          'GNAS','RNF43','ARID1A','TGFBR2','KDM6A','PREX2',
                          'RREB1','PBRM1','ROBO2','KLF4','TGIF1','RBM10',
                          'BRCA1','BRCA2','PALB2','ATM','CHEK2','MLH1','MSH2','MSH6',
                          'PIK3CA','AKT1','PTEN','MTOR','STK11'],
                'display_name': 'Pancreatic Driver Genes', 'color': '#d73027',
            },
            'KRAS_PATHWAY': {
                'genes': ['KRAS','NRAS','HRAS','BRAF','RAF1','ARAF',
                          'MAP2K1','MAP2K2','MAPK1','MAPK3','MAPK8','MAPK9',
                          'SOS1','SOS2','GRB2','PTPN11','NF1','SPRED1',
                          'EGFR','ERBB2','ERBB3','MET','FGFR1','FGFR2',
                          'JUN','FOS','MYC','ELK1','ETS1'],
                'display_name': 'KRAS Signaling', 'color': '#f46d43',
            },
            'DNA_DAMAGE_REPAIR': {
                'genes': ['BRCA1','BRCA2','PALB2','RAD51','RAD51B','RAD51C','RAD51D',
                          'RAD52','RAD54L','BRIP1','BARD1','XRCC2','XRCC3',
                          'PRKDC','LIG4','XRCC4','XRCC5','XRCC6','NHEJ1',
                          'MLH1','MSH2','MSH6','PMS2','MSH3','PMS1',
                          'ATM','ATR','CHEK1','CHEK2','TP53','MDM2','PARP1','PARP2'],
                'display_name': 'DNA Damage Repair', 'color': '#fdae61',
            },
            'TGF_BETA_SMAD4': {
                'genes': ['TGFB1','TGFB2','TGFB3','TGFBR1','TGFBR2','TGFBR3',
                          'SMAD4','SMAD2','SMAD3','SMAD7','SMAD1','SMAD5','SMAD8',
                          'SMURF1','SMURF2','SKI','SKIL','BAMBI','LTBP1',
                          'ID1','ID2','ID3','RUNX2','RUNX3','CDKN2B','CDKN1A'],
                'display_name': 'TGF-\u03b2/SMAD4 Signaling', 'color': '#abd9e9',
            },
            'P16_INK4A_RB_PATHWAY': {
                'genes': ['CDKN2A','CDKN2B','CDK4','CDK6','RB1','RBL1','RBL2',
                          'E2F1','E2F2','E2F3','E2F4','E2F5','CCND1','CCND2','CCND3',
                          'CCNE1','CCNE2','CDKN1A','CDKN1B','TP53','MDM2','MDM4'],
                'display_name': 'p16/Rb Pathway', 'color': '#74add1',
            },
            'PI3K_AKT_MTOR': {
                'genes': ['PIK3CA','PIK3CB','PIK3CD','PIK3CG','PIK3R1','PIK3R2','PIK3R3',
                          'AKT1','AKT2','AKT3','PTEN','TSC1','TSC2','MTOR','RPTOR',
                          'RICTOR','PDPK1','GSK3B','FOXO1','FOXO3','FOXO4',
                          'RPS6KB1','EIF4E','EIF4EBP1','STK11'],
                'display_name': 'PI3K/AKT/mTOR', 'color': '#4575b4',
            },
            'WNT_SIGNALING': {
                'genes': ['CTNNB1','APC','AXIN1','AXIN2','GSK3B',
                          'TCF7L2','TCF7','LEF1','MYC','CCND1','JUN','FOS',
                          'DKK1','DKK3','SFRP1','SFRP2','WIF1','ZNRF3','RNF43',
                          'LRP5','LRP6','DVL1','DVL2','DVL3'],
                'display_name': 'Wnt Signaling', 'color': '#313695',
            },
            'PANCREATIC_STROMA_ECM': {
                'genes': ['COL1A1','COL1A2','COL3A1','COL4A1','COL5A1','COL6A1',
                          'FN1','LAMB1','LAMC1','VTN','TNC','THBS1','THBS2',
                          'MMP2','MMP9','MMP14','MMP1','MMP3','MMP7','MMP13',
                          'TIMP1','TIMP2','TIMP3','PLAU','SERPINE1',
                          'ACTA2','PDGFRA','PDGFRB','TGFB1','CTGF','LOX','LOXL2'],
                'display_name': 'Stroma & ECM', 'color': '#a50026',
            },
            'HEDGEHOG_SIGNALING': {
                'genes': ['SHH','IHH','DHH','PTCH1','PTCH2','SMO','SUFU',
                          'GLI1','GLI2','GLI3','HHIP','GAS1','CDON',
                          'MYCN','CCND1','CCND2','BCL2','FOXM1','SNAI1'],
                'display_name': 'Hedgehog Signaling', 'color': '#762a83',
            },
            'PANCREATIC_METABOLISM': {
                'genes': ['GLUT1','HK1','HK2','PFKP','ALDOA','TPI1','GAPDH',
                          'PGK1','PGAM1','ENO1','PKM','LDHA','LDHB','PDK1',
                          'ACLY','FASN','SCD1','ACSL1','CPT1A',
                          'GLUL','GLS','ASNS','PHGDH','PSAT1','PSPH',
                          'MYC','TP53','HIF1A','SREBF1','PPARA','PPARG'],
                'display_name': 'Metabolic Reprogramming', 'color': '#5aae61',
            },
        }
    elif cancer_type == 'Prostate':
        return {
            'PROSTATE_CANCER_DRIVERS': {
                'genes': ['AR','KLK3','NKX3-1','TMPRSS2','FKBP5','KLK2',
                          'TP53','RB1','PTEN','NKX3-1',
                          'BRCA1','BRCA2','ATM','CHEK2','MLH1','MSH2',
                          'PIK3CA','AKT1','MTOR','PTEN',
                          'ERG','ETV1','ETV4','ETV5','FLI1',
                          'MYC','SPOP','FOXA1','CHD1'],
                'display_name': 'Prostate Cancer Drivers', 'color': '#d73027',
            },
            'ANDROGEN_RECEPTOR_SIGNALING': {
                'genes': ['AR','KLK3','KLK2','TMPRSS2','FKBP5','NKX3-1','ACPP',
                          'STEAP2','PART1','NCOA1','NCOA2','NCOA3','NCOA4',
                          'CREBBP','EP300','SRC','FOXA1','GATA2','HOXB13',
                          'CYP17A1','CYP19A1','HSD3B1','HSD17B3','SRD5A1','SRD5A2'],
                'display_name': 'Androgen Receptor Signaling', 'color': '#f46d43',
            },
            'ETS_FUSION_GENES': {
                'genes': ['ERG','ETV1','ETV4','ETV5','FLI1','TMPRSS2',
                          'SLC45A3','NDRG1','CANT1','HNRNPA2B1',
                          'ELK1','ELK4','GABPA','SPI1','SPDEF'],
                'display_name': 'ETS Fusion Genes', 'color': '#fdae61',
            },
            'DNA_DAMAGE_REPAIR': {
                'genes': ['BRCA1','BRCA2','ATM','ATR','CHEK1','CHEK2','TP53',
                          'RAD50','RAD51','RAD51B','RAD51C','RAD51D','RAD52',
                          'PALB2','BRIP1','BARD1','NBN','MRE11A','FANCA','FANCC',
                          'PARP1','XRCC2','MLH1','MSH2','MSH6','PMS2'],
                'display_name': 'DNA Damage Repair', 'color': '#abd9e9',
            },
            'PI3K_AKT_MTOR_PATHWAY': {
                'genes': ['PIK3CA','PIK3CB','PIK3CD','PIK3CG','PIK3R1','PIK3R2','PIK3R3',
                          'AKT1','AKT2','AKT3','PTEN','TSC1','TSC2','MTOR','RPTOR',
                          'RICTOR','PDPK1','GSK3B','FOXO1','FOXO3','SGK1'],
                'display_name': 'PI3K/AKT/mTOR Pathway', 'color': '#74add1',
            },
            'CELL_CYCLE_REGULATION': {
                'genes': ['RB1','CCND1','CCND2','CCND3','CDK4','CDK6','CDK2','CDKN1A',
                          'CDKN1B','CDKN2A','CDKN2B','E2F1','E2F2','E2F3','TP53',
                          'MDM2','MDM4','CHEK1','CHEK2','ATM','ATR'],
                'display_name': 'Cell Cycle Regulation', 'color': '#4575b4',
            },
            'NEUROENDOCRINE_DIFFERENTIATION': {
                'genes': ['CHGA','CHGB','SYP','NCAM1','ENO2','INSM1','SOX2',
                          'RB1','TP53','MYCN','AURKA','PLCB4','SRRM4',
                          'ASCL1','ONECUT2','FOXA2','BRN2','REST'],
                'display_name': 'Neuroendocrine Differentiation', 'color': '#313695',
            },
            'WNT_SIGNALING': {
                'genes': ['CTNNB1','APC','AXIN1','AXIN2','GSK3B','TCF7L2','LEF1',
                          'MYC','CCND1','JUN','FOS','DKK1','DKK3','SFRP1',
                          'WIF1','ZNRF3','RNF43','LRP5','LRP6'],
                'display_name': 'Wnt Signaling', 'color': '#a50026',
            },
            'PROSTATE_METASTASIS_GENES': {
                'genes': ['CDH1','CTNNB1','SNAI1','SNAI2','TWIST1','ZEB1','ZEB2',
                          'VIM','FN1','MMP2','MMP9','VEGFA','VEGFR2','PDGFRA',
                          'RANK','RANKL','PTHrP','RUNX2','DKK1','CXCR4'],
                'display_name': 'Prostate Metastasis Genes', 'color': '#762a83',
            },
            'IMMUNE_MICROENVIRONMENT': {
                'genes': ['PDCD1','CD274','PDCD1LG2','CTLA4','LAG3','TIGIT','HAVCR2',
                          'ICOS','ICOSLG','CD276','VTCN1','IDO1','CD80','CD86',
                          'B2M','HLA-A','HLA-B','HLA-C','IFNG','TNF','IL6'],
                'display_name': 'Immune Microenvironment', 'color': '#5aae61',
            },
        }
    else:  # Pan-cancer
        return {
            'CORE_ONCOGENES': {
                'genes': ['MYC','MYCN','MYCL','KRAS','NRAS','HRAS','BRAF','EGFR','ERBB2','ERBB3',
                          'PIK3CA','AKT1','AKT2','MTOR','MDM2','MDM4','CCND1','CDK4','CDK6',
                          'ALK','ROS1','MET','FGFR1','FGFR2','RET'],
                'display_name': 'Core Oncogenes', 'color': '#d73027',
            },
            'TUMOR_SUPPRESSORS': {
                'genes': ['TP53','RB1','PTEN','APC','BRCA1','BRCA2','VHL','NF1','NF2','ATM',
                          'CDKN2A','CDKN1A','CDKN1B','CHEK2','MLH1','MSH2','MSH6','PMS2',
                          'STK11','SMAD4','FBXW7','ARID1A','BAP1'],
                'display_name': 'Tumor Suppressors', 'color': '#f46d43',
            },
            'PI3K_AKT_MTOR': {
                'genes': ['PIK3CA','PIK3CB','PIK3CD','PIK3CG','PIK3R1','PIK3R2','PIK3R3',
                          'AKT1','AKT2','AKT3','MTOR','RPTOR','RICTOR','PTEN','TSC1','TSC2',
                          'RHEB','PDPK1','GSK3B','FOXO1','FOXO3'],
                'display_name': 'PI3K/AKT/mTOR Signaling', 'color': '#fdae61',
            },
            'RAS_MAPK_SIGNALING': {
                'genes': ['KRAS','NRAS','HRAS','BRAF','RAF1','ARAF','MAP2K1','MAP2K2',
                          'MAPK1','MAPK3','SOS1','GRB2','PTPN11','NF1','EGFR','ERBB2','MET'],
                'display_name': 'RAS/MAPK Signaling', 'color': '#fee08b',
            },
            'WNT_BETA_CATENIN': {
                'genes': ['APC','CTNNB1','AXIN1','AXIN2','GSK3B','TCF7L2','LEF1','MYC','CCND1',
                          'DKK1','SFRP1','WIF1','ZNRF3','RNF43','LRP5','LRP6'],
                'display_name': 'Wnt/\u03b2-Catenin Signaling', 'color': '#e0f3f8',
            },
            'TGF_BETA_SIGNALING': {
                'genes': ['TGFB1','TGFB2','TGFBR1','TGFBR2','SMAD2','SMAD3','SMAD4','SMAD7',
                          'SMURF1','SMURF2','SKI','SKIL','BAMBI'],
                'display_name': 'TGF-\u03b2 Signaling', 'color': '#abd9e9',
            },
            'CELL_CYCLE_CONTROL': {
                'genes': ['RB1','CCND1','CCND2','CCND3','CDK4','CDK6','CDK2','CDKN1A','CDKN1B',
                          'CDKN2A','CDKN2B','E2F1','E2F2','E2F3','CHEK1','CHEK2','WEE1'],
                'display_name': 'Cell Cycle Control', 'color': '#74add1',
            },
            'APOPTOSIS_REGULATION': {
                'genes': ['TP53','BAX','BCL2','BCL2L1','BCL2L11','CASP3','CASP8','CASP9',
                          'CYCS','FAS','FADD','BBC3','PMAIP1','MDM2','MDM4','ATM'],
                'display_name': 'Apoptosis Regulation', 'color': '#4575b4',
            },
            'DNA_REPAIR_PATHWAYS': {
                'genes': ['BRCA1','BRCA2','ATM','ATR','CHEK1','CHEK2','TP53','RAD50','RAD51',
                          'RAD51B','RAD51C','RAD51D','RAD52','PALB2','BRIP1','BARD1','NBN',
                          'MRE11A','FANCA','FANCC','PARP1','XRCC2','MLH1','MSH2','MSH6',
                          'POLD1','POLE'],
                'display_name': 'DNA Repair Pathways', 'color': '#313695',
            },
            'RECEPTOR_TYROSINE_KINASES': {
                'genes': ['EGFR','ERBB2','ERBB3','ERBB4','MET','ALK','ROS1','RET','NTRK1',
                          'NTRK2','NTRK3','FGFR1','FGFR2','FGFR3','FGFR4','PDGFRA','PDGFRB',
                          'KDR','KIT','CSF1R','IGF1R'],
                'display_name': 'Receptor Tyrosine Kinases', 'color': '#762a83',
            },
            'HORMONE_SIGNALING': {
                'genes': ['ESR1','ESR2','PGR','AR','GREB1','TFF1','FOXA1','GATA3','TBX3',
                          'NCOA1','NCOA2','NCOA3','EP300','CREBBP'],
                'display_name': 'Hormone Signaling', 'color': '#5aae61',
            },
            'IMMUNE_CHECKPOINT': {
                'genes': ['PDCD1','CD274','PDCD1LG2','CTLA4','LAG3','TIGIT','HAVCR2',
                          'ICOS','ICOSLG','CD276','VTCN1','IDO1','IDO2','CD80','CD86',
                          'B2M','HLA-A','HLA-B','HLA-C'],
                'display_name': 'Immune Checkpoint', 'color': '#1a9850',
            },
            'METASTASIS_INVASION': {
                'genes': ['CDH1','CTNNB1','SNAI1','SNAI2','TWIST1','ZEB1','ZEB2',
                          'VIM','FN1','MMP2','MMP9','MMP7','VEGFA','VEGFR2',
                          'PDGFRA','PDGFRB','FGF2'],
                'display_name': 'Metastasis & Invasion', 'color': '#998ec3',
            },
            'METABOLIC_REPROGRAMMING': {
                'genes': ['MTOR','AKT1','PIK3CA','PTEN','TSC1','TSC2',
                          'HIF1A','VHL','LDHA','PKM','PFKFB3'],
                'display_name': 'Metabolic Reprogramming', 'color': '#bf812d',
            },
        }


# ============================================================
# Step 3: GSEA calculation
# ============================================================
def prepare_ranking(gene_freq_df):
    log2 = np.log2(gene_freq_df['Frequency'].values + 1)
    noise = np.random.normal(0, 0.001, len(log2))
    return pd.DataFrame({
        'Gene': gene_freq_df['Gene'].values,
        'Score': log2 + noise
    }).sort_values('Score', ascending=False).reset_index(drop=True)


def calc_enrichment(ranking, gene_set_list):
    gene_scores = dict(zip(ranking['Gene'], ranking['Score']))
    sorted_genes = ranking['Gene'].tolist()
    N = len(sorted_genes)
    gs = set(gene_set_list)
    found = [g for g in gs if g in gene_scores]
    if not found:
        return None
    hit_idx = [i for i, g in enumerate(sorted_genes) if g in gs]
    Nr = sum(abs(gene_scores[g]) for g in found)
    running = []
    s = 0
    for i, g in enumerate(sorted_genes):
        if g in gs:
            s += abs(gene_scores[g]) / Nr if Nr else 1 / len(found)
        else:
            s -= 1 / (N - len(found))
        running.append(s)
    max_es = max(running, key=abs)
    return {'hit_indices': hit_idx, 'running': running, 'es': max_es,
            'max_pos': running.index(max_es), 'found': found}


def permutation_test(ranking, gene_set_list, n_perm=1000):
    orig = calc_enrichment(ranking, gene_set_list)
    if not orig:
        return None, None, None
    es = orig['es']
    scores = ranking['Score'].values
    names = ranking['Gene'].values
    null = []
    for _ in range(n_perm):
        perm = pd.DataFrame({'Gene': names, 'Score': np.random.permutation(scores)})
        perm = perm.sort_values('Score', ascending=False).reset_index(drop=True)
        r = calc_enrichment(perm, gene_set_list)
        if r:
            null.append(r['es'])
    if not null:
        return es, None, None
    if es >= 0:
        p = sum(1 for x in null if x >= es) / len(null)
        pos = [x for x in null if x >= 0]
        nes = es / np.mean(pos) if pos else 0
    else:
        p = sum(1 for x in null if x <= es) / len(null)
        neg = [x for x in null if x < 0]
        nes = es / abs(np.mean(neg)) if neg else 0
    return es, nes, p


# ============================================================
# Step 4: Plot
# ============================================================
def plot_gsea(ranking, pathway_results, cancer_name, output_path):
    gene_scores = ranking['Score'].values
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 0.8], hspace=0.05)

    # Top: enrichment curves
    ax = fig.add_subplot(gs[0])
    for name, d in pathway_results.items():
        nes = d.get('nes', 0)
        p = d.get('p_value', 1)
        p_str = 'p<0.001' if p < 0.001 else 'p={:.3f}'.format(p)
        ax.plot(range(len(d['running'])), d['running'], lw=3, color=d['color'], alpha=0.8,
                label='{} (NES={:.2f}, {})'.format(d['display_name'], nes, p_str))
        ax.scatter([d['max_pos']], [d['es']], color=d['color'], s=100, zorder=5)
    ax.axhline(0, color='black', ls='--', alpha=0.5, lw=1)
    ax.set_ylabel('Running Enrichment Score', fontsize=14, fontweight='bold')
    ax.set_title('GSEA: Top Enriched Pathways in {}'.format(cancer_name),
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlim(0, len(gene_scores))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True)
    ax.set_xticklabels([])

    # Info box
    info = 'NES    P-val   FDR\n' + '-'*22 + '\n'
    for d in pathway_results.values():
        info += '{:5.2f}  {:6.3f}  {:5.3f}\n'.format(
            d.get('nes',0), d.get('p_value',1), d.get('fdr',1))
    ax.text(0.02, 0.98, info, transform=ax.transAxes, va='top', fontsize=9,
            fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.9))

    # Middle: hit positions
    ax2 = fig.add_subplot(gs[1])
    n_pw = len(pathway_results)
    for i, (name, d) in enumerate(pathway_results.items()):
        y = n_pw - 1 - i
        p = d.get('p_value', 1)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        for h in d['hit_indices']:
            ax2.vlines(h, y-0.4, y+0.4, colors=d['color'], alpha=0.8, lw=2)
    ax2.set_xlim(0, len(gene_scores))
    ax2.set_ylim(-0.5, n_pw-0.5)
    labels = []
    for d in pathway_results.values():
        p = d.get('p_value', 1)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        labels.append('{} {}'.format(d['display_name'], sig))
    ax2.set_yticks(list(range(n_pw)))
    ax2.set_yticklabels(labels[::-1], fontsize=10)
    ax2.set_ylabel('Gene Sets', fontsize=12, fontweight='bold')
    ax2.set_xticklabels([])
    for i in range(n_pw-1):
        ax2.axhline(i+0.5, color='gray', ls='-', alpha=0.3, lw=0.5)

    # Bottom: ranking metric
    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(range(len(gene_scores)), gene_scores, 0, color='lightgray', alpha=0.7)
    ax3.axhline(0, color='black', ls='-', alpha=0.5)
    ax3.set_xlim(0, len(gene_scores))
    ax3.set_xlabel('Rank in Ordered Gene List', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Ranked List\nMetric', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.98, 0.95, '*** p<0.001, ** p<0.01, * p<0.05', transform=ax3.transAxes,
             ha='right', va='top', fontsize=9, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

    for ext in ['pdf', 'svg']:
        fig.savefig('{}.{}'.format(output_path, ext), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: {}.pdf'.format(output_path))


# ============================================================
# Main
# ============================================================
def run_cancer_from_freq(cancer_type, freq_path, n_top=5, n_perm=1000):
    """Run GSEA using pre-computed gene frequency file."""
    print('\n' + '='*60)
    print('GSEA: {} (from pre-computed frequencies)'.format(cancer_type))
    print('='*60)

    freq_df = pd.read_csv(freq_path)
    freq_df = freq_df[freq_df['Frequency'] > 0].sort_values('Frequency', ascending=False).reset_index(drop=True)
    print('  {} genes with >0 frequency'.format(len(freq_df)))

    gene_sets = get_gene_sets(cancer_type)
    ranking = prepare_ranking(freq_df)

    stats = []
    for pw_name, pw_data in gene_sets.items():
        es, nes, p = permutation_test(ranking, pw_data['genes'], n_perm)
        if es is not None:
            stats.append({
                'pathway': pw_name, 'es': es, 'nes': nes, 'p_value': p,
                'abs_nes': abs(nes) if nes else 0,
                'display_name': pw_data['display_name'],
            })

    stats.sort(key=lambda x: x['abs_nes'], reverse=True)

    pvals = [s['p_value'] for s in stats if s['p_value'] is not None]
    if len(pvals) > 1:
        fdrs = false_discovery_control(pvals)
        fi = 0
        for s in stats:
            if s['p_value'] is not None:
                s['fdr'] = fdrs[fi]; fi += 1
            else:
                s['fdr'] = 1.0
    else:
        for s in stats:
            s['fdr'] = s.get('p_value', 1.0)

    print('\n  Top pathways (by |NES|):')
    for i, s in enumerate(stats[:n_top]):
        p_str = '<0.001' if s['p_value'] < 0.001 else '{:.3f}'.format(s['p_value'])
        print('    {} {:<35s} NES={:.3f}  p={}  FDR={:.3f}'.format(
            i+1, s['display_name'], s['nes'], p_str, s['fdr']))

    top = stats[:n_top]
    pathway_results = {}
    for s in top:
        pw = gene_sets[s['pathway']]
        e = calc_enrichment(ranking, pw['genes'])
        if e:
            pathway_results[s['pathway']] = {
                **e, **pw, 'nes': s['nes'], 'p_value': s['p_value'], 'fdr': s['fdr'],
            }

    res_df = pd.DataFrame([{
        'Pathway': d['display_name'], 'NES': d['nes'], 'ES': d['es'],
        'P_value': d['p_value'], 'FDR': d['fdr'],
        'Genes_in_Set': len(d['genes']), 'Genes_Found': len(d['found']),
    } for d in pathway_results.values()])
    res_df.to_csv(os.path.join(OUTPUT_DIR, 'gsea_results_{}.csv'.format(cancer_type.lower())), index=False)

    cancer_labels = {
        'Breast': 'Breast Cancer', 'NSCLC': 'Non-small Cell Lung Cancer',
        'Colorectal': 'Colorectal Cancer', 'Pancreatic': 'Pancreatic Cancer',
        'Prostate': 'Prostate Cancer', 'PanCancer': 'Pan-Cancer (All Types)',
    }
    plot_gsea(ranking, pathway_results, cancer_labels.get(cancer_type, cancer_type),
              os.path.join(OUTPUT_DIR, 'gsea_{}'.format(cancer_type.lower())))

    return pathway_results


def run_cancer(cancer_type, texts, n_top=5, n_perm=1000):
    print('\n' + '='*60)
    print('GSEA: {}'.format(cancer_type))
    print('='*60)

    # Extract gene frequencies
    freq_df = extract_gene_frequencies(texts)
    freq_path = os.path.join(OUTPUT_DIR, 'gene_frequencies_{}.csv'.format(cancer_type.lower()))
    freq_df.to_csv(freq_path, index=False)
    print('  {} genes with >0 frequency from {} texts'.format(len(freq_df), len(texts)))

    # Get gene sets
    gene_sets = get_gene_sets(cancer_type)
    ranking = prepare_ranking(freq_df)

    # Compute enrichment + permutation test
    stats = []
    for pw_name, pw_data in gene_sets.items():
        es, nes, p = permutation_test(ranking, pw_data['genes'], n_perm)
        if es is not None:
            stats.append({
                'pathway': pw_name, 'es': es, 'nes': nes, 'p_value': p,
                'abs_nes': abs(nes) if nes else 0,
                'display_name': pw_data['display_name'],
            })

    stats.sort(key=lambda x: x['abs_nes'], reverse=True)

    # FDR
    pvals = [s['p_value'] for s in stats if s['p_value'] is not None]
    if len(pvals) > 1:
        fdrs = false_discovery_control(pvals)
        fi = 0
        for s in stats:
            if s['p_value'] is not None:
                s['fdr'] = fdrs[fi]; fi += 1
            else:
                s['fdr'] = 1.0
    else:
        for s in stats:
            s['fdr'] = s.get('p_value', 1.0)

    # Print results
    print('\n  Top pathways (by |NES|):')
    for i, s in enumerate(stats[:n_top]):
        p_str = '<0.001' if s['p_value'] < 0.001 else '{:.3f}'.format(s['p_value'])
        print('    {} {:<35s} NES={:.3f}  p={}  FDR={:.3f}'.format(
            i+1, s['display_name'], s['nes'], p_str, s['fdr']))

    # Compute enrichment data for top pathways
    top = stats[:n_top]
    pathway_results = {}
    for s in top:
        pw = gene_sets[s['pathway']]
        e = calc_enrichment(ranking, pw['genes'])
        if e:
            pathway_results[s['pathway']] = {
                **e, **pw, 'nes': s['nes'], 'p_value': s['p_value'], 'fdr': s['fdr'],
            }

    # Save results CSV
    res_df = pd.DataFrame([{
        'Pathway': d['display_name'], 'NES': d['nes'], 'ES': d['es'],
        'P_value': d['p_value'], 'FDR': d['fdr'],
        'Genes_in_Set': len(d['genes']), 'Genes_Found': len(d['found']),
    } for d in pathway_results.values()])
    res_df.to_csv(os.path.join(OUTPUT_DIR, 'gsea_results_{}.csv'.format(cancer_type.lower())), index=False)

    # Plot
    cancer_labels = {
        'Breast': 'Breast Cancer', 'NSCLC': 'Non-small Cell Lung Cancer',
        'Colorectal': 'Colorectal Cancer', 'Pancreatic': 'Pancreatic Cancer',
        'Prostate': 'Prostate Cancer', 'PanCancer': 'Pan-Cancer (All Types)',
    }
    plot_gsea(ranking, pathway_results, cancer_labels.get(cancer_type, cancer_type),
              os.path.join(OUTPUT_DIR, 'gsea_{}'.format(cancer_type.lower())))

    return pathway_results


def main():
    print('Loading texts by cancer type...')
    texts_by_cancer, all_texts = load_texts_by_cancer()
    for c, t in texts_by_cancer.items():
        print('  {}: {} texts'.format(c, len(t)))
    print('  Total: {} texts'.format(len(all_texts)))

    N_PERM = 1000

    # Use pre-computed per-cancer gene frequency files
    OLD_FREQ_DIR = os.environ.get('GENE_FREQUENCY_DIR', os.path.join(OUTPUT_DIR, 'gene_freq'))
    FREQ_MAP = {
        'Breast': 'gene_frequencies_breast_cancer.csv',
        'NSCLC': 'gene_frequencies_non_small_cell_lung_cancer.csv',
        'Colorectal': 'gene_frequencies_colorectal_cancer.csv',
        'Pancreatic': 'gene_frequencies_pancreatic_cancer.csv',
        'Prostate': 'gene_frequencies_prostate_cancer.csv',
        'PanCancer': 'gene_frequencies_overall.csv',
    }

    for cancer in ['Breast', 'NSCLC', 'Colorectal', 'Pancreatic', 'Prostate', 'PanCancer']:
        freq_path = os.path.join(OLD_FREQ_DIR, FREQ_MAP[cancer])
        n_top = 6 if cancer == 'PanCancer' else 5
        run_cancer_from_freq(cancer, freq_path, n_top=n_top, n_perm=N_PERM)

    print('\n\nAll GSEA analyses complete!')
    print('Results saved to: {}'.format(OUTPUT_DIR))


if __name__ == '__main__':
    main()
