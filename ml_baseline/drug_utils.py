#!/usr/bin/env python3
"""
Unified Drug Normalization & Classification Utilities

Three levels of drug analysis:
  1. Single drug:      normalize_drug("NABPACLITAXEL") → "NAB-PACLITAXEL"
  2. Combination:       get_core_combination("CARBOPLATIN,PACLITAXEL,LEUCOVORIN")
                        → "CARBOPLATIN, PACLITAXEL"
  3. Regimen class:     classify_regimen("DOXORUBICIN,CYCLOPHOSPHAMIDE")
                        → "AC/EC based Chemotherapy"
"""

import pandas as pd

# ============================================================
# 1. Salt/formulation suffixes to strip
# ============================================================
DRUG_SUFFIXES = [
    'HYDROCHLORIDE', 'HCL', 'MESYLATE', 'MALEATE', 'DIMALEATE',
    'SUCCINATE', 'TARTRATE', 'SULFATE', 'CITRATE', 'FUMARATE',
    'ACETATE', 'BESYLATE', 'TOSYLATE', 'DITOSYLATE', 'DISODIUM',
    'SODIUM', 'PHOSPHATE', 'MALATE', 'SMALATE', 'LIPOSOME',
    'POLIGLUMEX', 'DIMETHYL SULFOXIDE', 'CALCIUM',
]

# ============================================================
# 2. Drug aliases → canonical names
#    After suffix stripping, map remaining variants.
#    Canonical names MUST appear in DRUG_CLASSES or regimen sets.
# ============================================================
DRUG_ALIASES = {
    # Taxane formulations → canonical
    'NABPACLITAXEL':                    'NAB-PACLITAXEL',
    'NAB PACLITAXEL':                   'NAB-PACLITAXEL',
    'PACLITAXEL PROTEIN-BOUND':         'NAB-PACLITAXEL',
    'PACLITAXEL ALBUMIN-BOUND':         'NAB-PACLITAXEL',
    'PACLITAXEL LOADED POLYMERIC MICELLE': 'NAB-PACLITAXEL',
    'PACLITAXEL TREVATIDE':             'PACLITAXEL',

    # Doxorubicin formulations → canonical
    'PEGYLATED LIPOSOMAL DOXORUBICIN':  'DOXORUBICIN',
    'LIPOSOMAL DOXORUBICIN':            'DOXORUBICIN',
    'DOXORUBICIN LIPOSOMAL':            'DOXORUBICIN',
    'DOXIL':                            'DOXORUBICIN',

    # Fluorouracil aliases
    '5-FLUOROURACIL':                   'FLUOROURACIL',
    '5-FU':                             'FLUOROURACIL',
    '5FU':                              'FLUOROURACIL',

    # Irinotecan formulations
    'IRINOTECAN LIPOSOMAL':             'IRINOTECAN',
    'LIPOSOMAL IRINOTECAN':             'IRINOTECAN',
    'ONIVYDE':                          'IRINOTECAN',

    # Biosimilar suffixes for antibodies
    'TRASTUZUMAB-ANNS':                 'TRASTUZUMAB',
    'TRASTUZUMAB-DKST':                 'TRASTUZUMAB',
    'TRASTUZUMAB-DTTB':                 'TRASTUZUMAB',
    'TRASTUZUMAB-PKRB':                 'TRASTUZUMAB',
    'TRASTUZUMAB-QYYP':                 'TRASTUZUMAB',
    'TRASTUZUMAB-HYALURONIDASE':        'TRASTUZUMAB',
    'TRASTUZUMAB/HYALURONIDASE-ZZXF':   'TRASTUZUMAB',
    'PERTUZUMAB-TRASTUZUMAB-HYALURONIDASE-ZZXF': 'PERTUZUMAB',
    'BEVACIZUMAB-AWWB':                 'BEVACIZUMAB',
    'BEVACIZUMAB-BVZR':                 'BEVACIZUMAB',
    'RITUXIMAB-ABBS':                   'RITUXIMAB',
    'RITUXIMAB-PVVR':                   'RITUXIMAB',

    # ADC names (keep as distinct canonical names for regimen classification)
    'T-DM1':                            'TRASTUZUMAB EMTANSINE',
    'ADO-TRASTUZUMAB EMTANSINE':        'TRASTUZUMAB EMTANSINE',
    'ALDO-TRASTUZUMAB EMTANSINE':       'TRASTUZUMAB EMTANSINE',
    'T-DXD':                            'TRASTUZUMAB DERUXTECAN',
    'FAM-TRASTUZUMAB DERUXTECAN':       'TRASTUZUMAB DERUXTECAN',
    'FAM-TRASTUZUMAB DERUXTECAN-NXKI':  'TRASTUZUMAB DERUXTECAN',
    'ENFORTUMAB VEDOTIN-EJFV':          'ENFORTUMAB VEDOTIN',
    'SACITUZUMAB GOVITECAN-HZIY':       'SACITUZUMAB GOVITECAN',

    # Growth factor support (normalize before checking SUPPORTIVE_DRUGS)
    'FILGRASTIM-SNDZ':                  'FILGRASTIM',
    'FILGRASTIM-AAFI':                  'FILGRASTIM',
    'PEGFILGRASTIM-JMDB':               'PEGFILGRASTIM',
    'PEGFILGRASTIM-BMEZ':               'PEGFILGRASTIM',
    'PEGFILGRASTIM-CBQV':               'PEGFILGRASTIM',

    # Combo drugs (single product containing multiple active agents)
    'HYALURONIDASE/PERTUZUMAB/TRASTUZUMAB': 'TRASTUZUMAB',
    'HYALURONIDASE-TRASTUZUMAB':            'TRASTUZUMAB',
    'PERTUZUMAB/TRASTUZUMAB/HYALURONIDASE': 'TRASTUZUMAB',

    # Trastuzumab/Rituximab combo products
    'TRASTUZUMAB/HYALURONIDASE-OYSK':   'TRASTUZUMAB',
    'RITUXIMAB AND HYALURONIDASE HUMAN': 'RITUXIMAB',

    # DACARBAZINE international trade names and synonyms
    'DTIC':                             'DACARBAZINE',
    'DTIC-DOME':                        'DACARBAZINE',
    'DTICDOME':                         'DACARBAZINE',
    'DACARBAZINA':                      'DACARBAZINE',
    'DACARBAZINE  DTIC':                'DACARBAZINE',
    'DACATIC':                          'DACARBAZINE',
    'DAKARBAZIN':                       'DACARBAZINE',
    'DETICENE':                         'DACARBAZINE',
    'DETIMEDAC':                        'DACARBAZINE',
    'DIC':                              'DACARBAZINE',
    'FAULDETIC':                        'DACARBAZINE',
    'IMIDAZOLE CARBOXAMIDE':            'DACARBAZINE',
    'WR139007':                         'DACARBAZINE',
    'BIOCARBAZINE':                     'DACARBAZINE',

    # Thalidomide trade names
    'THALOMID':                         'THALIDOMIDE',
    'DISTAVAL':                         'THALIDOMIDE',
    'KEVADON':                          'THALIDOMIDE',
    'NEUROSEDYN':                       'THALIDOMIDE',
    'PANTOSEDIV':                       'THALIDOMIDE',
    'SEDALIS':                          'THALIDOMIDE',
    'SEDOVAL K17':                      'THALIDOMIDE',
    'SOFTENON':                         'THALIDOMIDE',
    'SYNOVIR':                          'THALIDOMIDE',
    'TALIMOL':                          'THALIDOMIDE',

    # Trifluridine/tipiracil combo (Lonsurf) — all naming variants
    'TRIFLURIDINE AND TIPIRACIL':       'TRIFLURIDINE/TIPIRACIL',
    'TRIFLURIDINE + TIPIRACIL':         'TRIFLURIDINE/TIPIRACIL',
    'TIPIRACIL-TRIFLURIDINE':           'TRIFLURIDINE/TIPIRACIL',
    'TRIFLURIDINE-TIPIRACIL':           'TRIFLURIDINE/TIPIRACIL',
    'TRIFLURIDINE/TIPIRACIL':           'TRIFLURIDINE/TIPIRACIL',
    'TIPIRACIL + TRIFLURIDINE':         'TRIFLURIDINE/TIPIRACIL',
    'LONSURF':                          'TRIFLURIDINE/TIPIRACIL',

    # Goserelin typo
    'GOSERLIN':                         'GOSERELIN',

    # Cytarabine formulation
    'CYTARABINE LIPOSOMAL':             'CYTARABINE',

    # Ziv-aflibercept (VEGF trap)
    'ZIV AFLIBERCEPT':                  'ZIV-AFLIBERCEPT',
    'ZIV-AFLIBERCEPT':                  'ZIV-AFLIBERCEPT',

    # Lutetium variants (with/without hyphen)
    'LUTETIUM LU 177 DOTATATE':         'LUTETIUM LU-177 DOTATATE',

    # Tegafur combo (S-1)
    'TEGAFURGIMERACILOTERACIL POTASSIUM': 'TEGAFUR/GIMERACIL/OTERACIL',

    # Other aliases
    'CABOZANTINIB S-MALATE':            'CABOZANTINIB',
}

# ============================================================
# 3. Supportive / auxiliary drugs to remove for regimen classification
# ============================================================
SUPPORTIVE_DRUGS = {
    # Bone agents
    'ZOLEDRONIC ACID', 'PAMIDRONATE', 'DENOSUMAB',
    # GnRH agonists/antagonists (when used as ADT support)
    'LEUPROLIDE', 'GOSERELIN', 'DEGARELIX',
    # Chemo adjuncts
    'LEUCOVORIN', 'PREDNISONE', 'DEXAMETHASONE',
    # Antiemetics / supportive
    'ONDANSETRON', 'GRANISETRON', 'APREPITANT', 'MESNA',
    # Growth factors
    'FILGRASTIM', 'PEGFILGRASTIM',
    # Investigational placeholders
    'INVESTIGATIONAL', 'INVESTIGATIONAL DRUG',
    # Other
    'MEGESTROL',
}

# ============================================================
# 4. Drug class sets for ML feature extraction
#    Every canonical drug name should be found here.
# ============================================================
DRUG_CLASSES = {
    'Chemo': {
        'CARBOPLATIN', 'CISPLATIN', 'OXALIPLATIN',
        'PACLITAXEL', 'DOCETAXEL', 'NAB-PACLITAXEL',
        'GEMCITABINE', 'CAPECITABINE', 'FLUOROURACIL',
        'IRINOTECAN', 'ETOPOSIDE',
        'DOXORUBICIN', 'EPIRUBICIN',
        'CYCLOPHOSPHAMIDE', 'METHOTREXATE',
        'VINORELBINE', 'PEMETREXED', 'TOPOTECAN',
        'ERIBULIN', 'IXABEPILONE',
        'VINBLASTINE', 'VINCRISTINE', 'MITOMYCIN',
        'BENDAMUSTINE', 'TEMOZOLOMIDE',
        # LEUCOVORIN: folate modulator, also in SUPPORTIVE_DRUGS for regimen classification
        # but kept here for ML feature consistency with MSK-CHORD training
        'LEUCOVORIN',
    },
    'Targeted': {
        # HER2
        'TRASTUZUMAB', 'PERTUZUMAB', 'LAPATINIB', 'TUCATINIB', 'NERATINIB',
        # ADCs
        'TRASTUZUMAB EMTANSINE', 'TRASTUZUMAB DERUXTECAN',
        'SACITUZUMAB GOVITECAN', 'ENFORTUMAB VEDOTIN',
        # EGFR TKIs
        'ERLOTINIB', 'GEFITINIB', 'OSIMERTINIB', 'AFATINIB',
        # ALK/ROS1
        'CRIZOTINIB', 'ALECTINIB', 'CERITINIB', 'LORLATINIB', 'BRIGATINIB',
        # NTRK/RET
        'ENTRECTINIB', 'LAROTRECTINIB', 'SELPERCATINIB', 'PRALSETINIB',
        # RAF/MEK
        'VEMURAFENIB', 'DABRAFENIB', 'TRAMETINIB', 'COBIMETINIB',
        # CDK4/6
        'PALBOCICLIB', 'RIBOCICLIB', 'ABEMACICLIB',
        # mTOR/PI3K
        'EVEROLIMUS', 'TEMSIROLIMUS', 'ALPELISIB',
        # Multi-kinase / VEGF
        'SORAFENIB', 'SUNITINIB', 'PAZOPANIB', 'REGORAFENIB', 'CABOZANTINIB',
        'LENVATINIB', 'AXITINIB',
        'BEVACIZUMAB', 'RAMUCIRUMAB',
        # Anti-EGFR antibodies
        'CETUXIMAB', 'PANITUMUMAB',
        # PARP
        'OLAPARIB', 'RUCAPARIB', 'TALAZOPARIB', 'NIRAPARIB',
        # Other
        'RITUXIMAB',
    },
    'Immuno': {
        'NIVOLUMAB', 'PEMBROLIZUMAB', 'ATEZOLIZUMAB', 'DURVALUMAB',
        'AVELUMAB', 'IPILIMUMAB', 'TREMELIMUMAB', 'CEMIPLIMAB',
    },
    'Hormone': {
        'TAMOXIFEN', 'LETROZOLE', 'ANASTROZOLE', 'EXEMESTANE', 'FULVESTRANT',
        'ENZALUTAMIDE', 'ABIRATERONE',
        'BICALUTAMIDE', 'DAROLUTAMIDE', 'APALUTAMIDE',
        # GnRH agonists/antagonists — also in SUPPORTIVE_DRUGS for regimen classification
        # but kept here for ML feature consistency with MSK-CHORD training
        'LEUPROLIDE', 'GOSERELIN', 'DEGARELIX',
        'MEGESTROL',
    },
}

# Flatten for quick lookup: drug → class
_DRUG_TO_CLASS = {}
for _cls, _drugs in DRUG_CLASSES.items():
    for _d in _drugs:
        _DRUG_TO_CLASS[_d] = _cls

# ============================================================
# 5. Regimen classification constants
# ============================================================
ENDOCRINE_THERAPIES = {'TAMOXIFEN', 'ANASTROZOLE', 'LETROZOLE', 'EXEMESTANE', 'FULVESTRANT'}
CDK46_INHIBITORS    = {'PALBOCICLIB', 'RIBOCICLIB', 'ABEMACICLIB'}
HER2_TARGETS        = {'TRASTUZUMAB', 'PERTUZUMAB'}
HER2_ADCS           = {'TRASTUZUMAB EMTANSINE', 'TRASTUZUMAB DERUXTECAN'}
TNBC_ADCS           = {'SACITUZUMAB GOVITECAN'}
PARP_INHIBITORS     = {'OLAPARIB', 'TALAZOPARIB', 'RUCAPARIB', 'NIRAPARIB'}
PLATINUMS           = {'CISPLATIN', 'CARBOPLATIN', 'OXALIPLATIN'}
TAXANES             = {'PACLITAXEL', 'DOCETAXEL', 'NAB-PACLITAXEL'}
ANTHRACYCLINES      = {'DOXORUBICIN', 'EPIRUBICIN'}
IMMUNO_THERAPIES    = {'PEMBROLIZUMAB', 'ATEZOLIZUMAB', 'NIVOLUMAB', 'IPILIMUMAB',
                       'DURVALUMAB', 'AVELUMAB', 'CEMIPLIMAB', 'TREMELIMUMAB'}


# ============================================================
# Level 1: Single Drug Normalization
# ============================================================
def normalize_drug(name):
    """
    Canonical drug name: uppercase → strip salt suffixes → apply aliases.

    Examples:
        "nabpaclitaxel"               → "NAB-PACLITAXEL"
        "Afatinib Dimaleate"          → "AFATINIB"
        "Pegylated Liposomal Doxorubicin" → "DOXORUBICIN"
        "5-FU"                        → "FLUOROURACIL"
        "Trastuzumab-DKST"            → "TRASTUZUMAB"
    """
    name = name.strip().upper()
    # Strip salt / formulation suffixes (iterate — multiple suffixes possible)
    changed = True
    while changed:
        changed = False
        for suffix in DRUG_SUFFIXES:
            if name.endswith(' ' + suffix):
                name = name[:-len(suffix) - 1].strip()
                changed = True
    # Apply alias mapping
    name = DRUG_ALIASES.get(name, name)
    return name


# ============================================================
# Level 2: Core Drug Combination
# ============================================================
def get_core_combination(drug_combo_str):
    """
    Normalize all drugs, remove supportive agents, sort alphabetically.

    Input:  "CARBOPLATIN, NABPaclitaxel, Leucovorin"
    Output: "CARBOPLATIN, NAB-PACLITAXEL"
    """
    if not drug_combo_str or (isinstance(drug_combo_str, float) and pd.isna(drug_combo_str)):
        return "Unknown"
    drugs = {normalize_drug(d) for d in drug_combo_str.split(',')}
    core = drugs - SUPPORTIVE_DRUGS
    if not core:
        return "Supportive Care Only"
    return ', '.join(sorted(core))


# ============================================================
# Level 3: Regimen Class
# ============================================================
def classify_regimen(drug_combo_str):
    """
    Assign a high-level clinical regimen class to a drug combination.
    Drugs are normalized first, then supportive drugs removed, then rules applied.

    Examples:
        "DOXORUBICIN, CYCLOPHOSPHAMIDE"       → "AC/EC based Chemotherapy"
        "CARBOPLATIN, PACLITAXEL"              → "Platinum + Taxane"
        "LETROZOLE, PALBOCICLIB"               → "Endocrine Therapy + CDK4/6 Inhibitor"
        "PEMBROLIZUMAB"                        → "Immuno Monotherapy"
    """
    if not drug_combo_str or (isinstance(drug_combo_str, float) and pd.isna(drug_combo_str)):
        return "Unknown"

    original_drugs = {normalize_drug(d) for d in drug_combo_str.split(',')}
    core_drugs = original_drugs - SUPPORTIVE_DRUGS

    if not core_drugs:
        return "Supportive Care / ADT Monotherapy"

    # Rule engine: most specific → most general
    # --- Breast: Endocrine combinations ---
    if any(cdk in core_drugs for cdk in CDK46_INHIBITORS):
        return "Endocrine Therapy + CDK4/6 Inhibitor"
    if (any(et in core_drugs for et in ENDOCRINE_THERAPIES) and
            any(t in core_drugs for t in ('EVEROLIMUS', 'ALPELISIB'))):
        return "Endocrine Therapy + Other Targeted (mTOR/PI3K)"

    # --- ADCs ---
    if any(adc in core_drugs for adc in HER2_ADCS):
        return "HER2 Antibody-Drug Conjugate"
    if any(adc in core_drugs for adc in TNBC_ADCS):
        return "TNBC Antibody-Drug Conjugate (Sacituzumab)"

    # --- PARP ---
    if any(parp in core_drugs for parp in PARP_INHIBITORS):
        return "PARP Inhibitor"

    # --- HER2 targeted ---
    if {'TRASTUZUMAB', 'PERTUZUMAB'}.issubset(core_drugs):
        return "Dual HER2 Blockade (Trastuzumab+Pertuzumab) based"
    if 'TRASTUZUMAB' in core_drugs:
        return "Trastuzumab-based Regimen"

    # --- Standard chemo regimens ---
    if {'FLUOROURACIL', 'IRINOTECAN', 'OXALIPLATIN'}.issubset(core_drugs):
        return 'FOLFIRINOX'
    if {'FLUOROURACIL', 'OXALIPLATIN'}.issubset(core_drugs):
        return 'FOLFOX based'
    if {'FLUOROURACIL', 'IRINOTECAN'}.issubset(core_drugs):
        return 'FOLFIRI based'
    if {'CAPECITABINE', 'OXALIPLATIN'}.issubset(core_drugs):
        return 'CAPOX based'
    if any(a in core_drugs for a in ANTHRACYCLINES) and 'CYCLOPHOSPHAMIDE' in core_drugs:
        return "AC/EC based Chemotherapy"
    if {'DOCETAXEL', 'CYCLOPHOSPHAMIDE'}.issubset(core_drugs):
        return "TC Chemotherapy"
    if {'CYCLOPHOSPHAMIDE', 'METHOTREXATE', 'FLUOROURACIL'}.issubset(core_drugs):
        return "CMF Chemotherapy"

    # --- Platinum combos ---
    if any(p in core_drugs for p in PLATINUMS) and any(t in core_drugs for t in TAXANES):
        if any(i in core_drugs for i in IMMUNO_THERAPIES):
            return "Platinum + Taxane + Immunotherapy"
        return "Platinum + Taxane"
    if any(p in core_drugs for p in PLATINUMS) and 'PEMETREXED' in core_drugs:
        if any(i in core_drugs for i in IMMUNO_THERAPIES):
            return "Platinum + Pemetrexed + Immunotherapy"
        return "Platinum + Pemetrexed"

    # --- Other combinations ---
    if 'GEMCITABINE' in core_drugs and any(t in core_drugs for t in TAXANES):
        return "Gemcitabine + Taxane"

    # --- Prostate ---
    if 'ABIRATERONE' in core_drugs:
        return "Abiraterone based"
    if 'ENZALUTAMIDE' in core_drugs:
        return "Enzalutamide based"

    # --- Monotherapy ---
    if len(core_drugs) == 1:
        drug = list(core_drugs)[0]
        if drug in ENDOCRINE_THERAPIES:
            return "Endocrine Monotherapy"
        if drug in IMMUNO_THERAPIES:
            return "Immuno Monotherapy"
        return f"{drug} Monotherapy"

    return "Other Combination"


# ============================================================
# ML Feature Extraction (12 drug features)
# ============================================================
def extract_drug_features(drug_input):
    """
    Extract 12 drug features from a drug list or comma-separated string.
    Normalizes all drug names first to ensure consistency across datasets.

    Parameters:
        drug_input: list of drug names OR comma-separated string

    Returns:
        dict with 12 features:
          drug_count, has_chemo, has_targeted, has_immuno, has_hormone,
          chemo_count, targeted_count, immuno_count, hormone_count,
          is_combination, is_chemo_immuno, is_targeted_chemo
    """
    if isinstance(drug_input, str):
        drug_list = [d.strip() for d in drug_input.split(',')]
    else:
        drug_list = list(drug_input)

    # Normalize all drug names
    normalized = [normalize_drug(d) for d in drug_list]
    # drug_count includes ALL drugs (consistent with MSK-CHORD training)
    n = len(drug_list)

    has_c = has_t = has_i = has_h = 0
    cc = tc = ic = hc = 0

    for d in normalized:
        cls = _DRUG_TO_CLASS.get(d)
        if cls == 'Chemo':
            has_c = 1; cc += 1
        elif cls == 'Targeted':
            has_t = 1; tc += 1
        elif cls == 'Immuno':
            has_i = 1; ic += 1
        elif cls == 'Hormone':
            has_h = 1; hc += 1

    return {
        'drug_count': n,
        'has_chemo': has_c,
        'has_targeted': has_t,
        'has_immuno': has_i,
        'has_hormone': has_h,
        'chemo_count': cc,
        'targeted_count': tc,
        'immuno_count': ic,
        'hormone_count': hc,
        'is_combination': 1 if n > 1 else 0,
        'is_chemo_immuno': 1 if has_c and has_i else 0,
        'is_targeted_chemo': 1 if has_t and has_c else 0,
    }
