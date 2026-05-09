#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build TCGA prompts in MSK-compatible format.

Integrates:
- Clinical information
- WES mutations (from MAF)
- Treatment regimen
- Drug mechanism (from DrugBank)
- HR/HER2 status (breast cancer)
- Gleason score (prostate cancer)

Output format matches MSK-CHORD so the OncoRAFT model can score the prompts directly.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import re
from collections import defaultdict

# ============================================
# Configuration — import from centralized config
# ============================================
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    VALIDATION_BASE, DRUGBANK_FILE,
    TCGA_DATA_DIR as DATA_DIR,
    TCGA_CLINICAL_DIR as CLINICAL_DIR,
    TCGA_MUTATION_DIR as MUTATION_DIR,
    TCGA_TREATMENT_PLAN_DIR as TREATMENT_PLAN_DIR,
    TCGA_SURVIVAL_FILE as SURVIVAL_FILE,
    TCGA_CBIOPORTAL_DIR as CBIOPORTAL_DIR,
    TCGA_PROMPT_OUTPUT_DIR as PROMPT_OUTPUT_DIR,
    INSTRUCTION,
)

BASE_DIR = VALIDATION_BASE
OUTPUT_DIR = VALIDATION_BASE / "outputs"
PROMPT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# Important cancer genes (used to prioritize mutations in display)
# ============================================
IMPORTANT_GENES = {
    "TP53", "KRAS", "EGFR", "BRAF", "PIK3CA", "PTEN", "APC",
    "BRCA1", "BRCA2", "ATM", "RB1", "CDKN2A", "MYC", "ERBB2",
    "ALK", "ROS1", "MET", "RET", "NRAS", "STK11", "KEAP1",
    "NF1", "ARID1A", "SMAD4", "FBXW7", "CTNNB1", "IDH1", "IDH2",
    "FGFR1", "FGFR2", "FGFR3", "NOTCH1", "NOTCH2", "AR", "ESR1",
    "CDK4", "CDK6", "CCND1", "MDM2", "TERT"
}

# ============================================
# Drug Name Standardization
# ============================================
DRUG_SUBSTITUTIONS = {
    "PACLITAXEL PROTEIN-BOUND": "PACLITAXEL",
    "FAM-TRASTUZUMAB DERUXTECAN": "TRASTUZUMAB DERUXTECAN",
    "DOXORUBICIN LIPOSOMAL": "DOXORUBICIN",
    "DOXORUBICIN HYDROCHLORIDE": "DOXORUBICIN",
    "GEMCITABINE HYDROCHLORIDE": "GEMCITABINE",
    "IRINOTECAN HYDROCHLORIDE": "IRINOTECAN",
    "EPIRUBICIN HYDROCHLORIDE": "EPIRUBICIN",
}

# ============================================
# Smoking History Standardization (TCGA → MSK format)
# ============================================
SMOKING_HISTORY_MAP = {
    "current smoker": "Former/Current Smoker",
    "current reformed smoker for < or = 15 yrs": "Former/Current Smoker",
    "current reformed smoker for > 15 yrs": "Former/Current Smoker",
    "current reformed smoker, duration not specified": "Former/Current Smoker",
    "lifelong non-smoker": "Never",
    "lifelong non-smoker (<100 cigarettes smoked in lifetime)": "Never",
    "not reported": "Unknown",
}

# ============================================
# Ethnicity Standardization (TCGA → MSK format)
# ============================================
ETHNICITY_MAP = {
    "not hispanic or latino": "Non-Spanish; Non-Hispanic",
    "hispanic or latino": "Spanish NOS; Hispanic NOS, Latino NOS",
    "not reported": "Unknown",
}

# ============================================
# Stage Standardization (TCGA AJCC → MSK simplified)
# ============================================
def standardize_stage(stage_str):
    """Convert TCGA AJCC stage (e.g. 'Stage IIB') to MSK format (e.g. 'Stage 1-3')."""
    if not stage_str or str(stage_str).lower() in ('nan', 'none', 'unknown', 'not reported', ''):
        return "Unknown"
    s = str(stage_str).strip()
    # Handle TCGA special values
    if s.startswith('[') and s.endswith(']'):
        return "Unknown"
    s_upper = s.upper().replace("STAGE ", "")
    if s_upper.startswith("IV") or s_upper == "4":
        return "Stage 4"
    elif s_upper.startswith("X"):
        return "Unknown"
    elif s_upper.startswith("0") or s_upper.startswith("I") or s_upper.startswith("II") or s_upper.startswith("III") \
         or s_upper in ("1", "2", "3"):
        return "Stage 1-3"
    return s

# ============================================
# MSI Score Classification
# ============================================
def classify_msi(mantis_score, sensor_score=None):
    """Classify MSI status from MANTIS/SENSOR scores.
    MANTIS >= 0.4 → MSI-H; < 0.4 → MSS (Stable)
    MSIsensor >= 3.5 → MSI-H
    """
    try:
        mantis = float(mantis_score)
        if mantis >= 0.4:
            return "Instable", mantis
        else:
            return "Stable", mantis
    except (ValueError, TypeError):
        pass
    try:
        sensor = float(sensor_score)
        if sensor >= 3.5:
            return "Instable", sensor
        else:
            return "Stable", sensor
    except (ValueError, TypeError):
        pass
    return "Unknown", None

# ============================================
# Cancer Type Detailed Standardization
# ============================================
CANCER_TYPE_DETAILED_MAP = {
    # BRCA PAM50 subtypes
    "BRCA_LumA": "Breast Invasive Ductal Carcinoma (Luminal A)",
    "BRCA_LumB": "Breast Invasive Ductal Carcinoma (Luminal B)",
    "BRCA_Basal": "Breast Invasive Ductal Carcinoma (Basal-like)",
    "BRCA_Her2": "Breast Invasive Ductal Carcinoma (HER2-enriched)",
    "BRCA_Normal": "Breast Invasive Ductal Carcinoma (Normal-like)",
    # Lung
    "LUAD": "Lung Adenocarcinoma",
    "LUSC": "Lung Squamous Cell Carcinoma",
    # CRC
    "COAD": "Colon Adenocarcinoma",
    "READ": "Rectal Adenocarcinoma",
    # Others
    "PRAD": "Prostate Adenocarcinoma",
    "PAAD": "Pancreatic Adenocarcinoma",
}

def _safe_str(val):
    """Convert value to string, returning None for NaN/None."""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    s = str(val).strip()
    if s.lower() in ('nan', 'none', ''):
        return None
    return s


class TCGAPromptBuilder:
    """TCGA prompt builder (MSK-compatible format)."""
    
    def __init__(self):
        self.output_dir = PROMPT_OUTPUT_DIR
        
        # Load DrugBank data
        self.drugbank_data = self._load_drugbank()
        
        # Data containers
        self.clinical_data = {}      # GDC clinical data
        self.cbioportal_data = {}    # cBioPortal supplementary data
        self.survival_data = {}      # CDR survival data
        self.mutations = {}          # MAF mutations
        self.treatment_plans = {}    # Treatment plans
        
        print("TCGAPromptBuilder initialized")
    
    def _load_drugbank(self) -> dict:
        """Load DrugBank drug information."""
        if DRUGBANK_FILE.exists():
            with open(DRUGBANK_FILE, 'r') as f:
                data = json.load(f)
                
                # Convert list-format DrugBank to dict for quick lookup
                if isinstance(data, list):
                    drugbank_dict = {}
                    for item in data:
                        # Try multiple name fields
                        name = item.get("name") or item.get("drug_name") or item.get("generic_name")
                        if name:
                            drugbank_dict[name.lower()] = item
                    print(f"Loaded DrugBank data: {len(drugbank_dict)} drugs (converted from list)")
                    return drugbank_dict
                elif isinstance(data, dict):
                    # If already a dict, lowercase keys for case-insensitive matching
                    drugbank_dict = {k.lower(): v for k, v in data.items()}
                    print(f"Loaded DrugBank data: {len(drugbank_dict)} drugs")
                    return drugbank_dict
                else:
                    print(f"Warning: Unknown DrugBank format: {type(data)}")
                    return {}
        print("Warning: DrugBank file not found")
        return {}
    
    def load_all_data(self):
        """Load all input data sources."""
        print("\n" + "=" * 60)
        print("Loading all data sources...")
        print("=" * 60)
        
        # 1. Load GDC clinical data
        self._load_gdc_clinical()
        
        # 2. Load cBioPortal data (HR/HER2, Gleason, etc.)
        self._load_cbioportal_data()
        
        # 3. Load CDR survival data
        self._load_survival_data()
        
        # 4. Load MAF mutations
        self._load_mutation_data()
        
        # 5. Load treatment plans
        self._load_treatment_plans()
        
        print("\nData loading complete!")
        print(f"  Clinical: {len(self.clinical_data)} patients")
        print(f"  cBioPortal: {len(self.cbioportal_data)} patients")
        print(f"  Survival: {len(self.survival_data)} patients")
        print(f"  Mutations: {len(self.mutations)} patients")
        print(f"  Treatment Plans: {len(self.treatment_plans)} patients")
    
    def _load_gdc_clinical(self):
        """Load GDC clinical data."""
        print("\nLoading GDC clinical data...")
        
        # Load merged clinical file if present
        combined_file = CLINICAL_DIR / "all_projects_clinical.csv"
        if combined_file.exists():
            df = pd.read_csv(combined_file)
            for _, row in df.iterrows():
                patient_id = row.get("patient_id")
                if patient_id:
                    self.clinical_data[patient_id] = row.to_dict()
            print(f"  Loaded {len(self.clinical_data)} patients from GDC")
        else:
            # Otherwise load per-cancer files
            for csv_file in CLINICAL_DIR.glob("TCGA-*_clinical.csv"):
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    patient_id = row.get("patient_id")
                    if patient_id:
                        self.clinical_data[patient_id] = row.to_dict()
            print(f"  Loaded {len(self.clinical_data)} patients from individual files")
    
    def _load_cbioportal_data(self):
        """Load cBioPortal data (HR/HER2, Gleason, subtype, etc.)."""
        print("\nLoading cBioPortal data...")
        
        if not CBIOPORTAL_DIR.exists():
            print("  cBioPortal directory not found, skipping...")
            return
        
        all_attributes = set()

        # Load both clinical and sample JSON files (sample has MSI data)
        json_files = list(CBIOPORTAL_DIR.glob("*_clinical.json")) + \
                     list(CBIOPORTAL_DIR.glob("*_sample.json"))

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Group records by patient ID
                for item in data:
                    patient_id = item.get("patientId")
                    attr_id = item.get("clinicalAttributeId")
                    value = item.get("value")
                    
                    if attr_id:
                        all_attributes.add(attr_id)
                    
                    if patient_id and attr_id:
                        if patient_id not in self.cbioportal_data:
                            self.cbioportal_data[patient_id] = {}
                        self.cbioportal_data[patient_id][attr_id] = value
            except Exception as e:
                print(f"  Error loading {json_file.name}: {e}")
        
        print(f"  Loaded {len(self.cbioportal_data)} patients from cBioPortal")
        
        # Print available attributes for debugging
        relevant_attrs = [a for a in all_attributes if any(
            keyword in a.upper() for keyword in 
            ['ER', 'PR', 'HER2', 'HORMONE', 'RECEPTOR', 'GLEASON', 'SUBTYPE', 'MSI']
        )]
        if relevant_attrs:
            print(f"  Relevant attributes found: {sorted(relevant_attrs)}")
    
    def _load_survival_data(self):
        """Load CDR survival data."""
        print("\nLoading CDR survival data...")
        
        if SURVIVAL_FILE.exists():
            df = pd.read_excel(SURVIVAL_FILE)
            for _, row in df.iterrows():
                patient_id = row.get("bcr_patient_barcode")
                if patient_id:
                    self.survival_data[patient_id] = {
                        "OS": row.get("OS"),
                        "OS.time": row.get("OS.time"),
                        "PFI": row.get("PFI"),
                        "PFI.time": row.get("PFI.time"),
                        "DSS": row.get("DSS"),
                        "DSS.time": row.get("DSS.time"),
                        "vital_status": row.get("vital_status"),
                        "tumor_status": row.get("tumor_status"),
                        "age_at_diagnosis": row.get("age_at_initial_pathologic_diagnosis"),
                        "gender": row.get("gender"),
                        "race": row.get("race"),
                        "ajcc_stage": row.get("ajcc_pathologic_tumor_stage"),
                        "clinical_stage": row.get("clinical_stage"),
                        "histological_type": row.get("histological_type"),
                        "histological_grade": row.get("histological_grade"),
                        "cancer_type": row.get("type")
                    }
            print(f"  Loaded {len(self.survival_data)} patients from CDR")
        else:
            print("  CDR file not found!")
    
    def _load_mutation_data(self):
        """Load mutation data (supports multiple file layouts)."""
        print("\nLoading mutation data...")
        
        # Layout 1: TSV files named TCGA-*.somaticmutation_wxs.tsv
        tsv_files = list(MUTATION_DIR.glob("TCGA-*.somaticmutation_wxs.tsv"))
        
        if tsv_files:
            print(f"  Found {len(tsv_files)} TSV mutation files")
            
            for tsv_file in tsv_files:
                print(f"  Processing {tsv_file.name}...")
                
                try:
                    df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
                    print(f"    Rows: {len(df)}")
                    
                    # Extract patient ID (first 12 chars of sample / Tumor_Sample_Barcode)
                    if 'sample' in df.columns:
                        df['patient_id'] = df['sample'].str[:12]
                    elif 'Tumor_Sample_Barcode' in df.columns:
                        df['patient_id'] = df['Tumor_Sample_Barcode'].str[:12]
                    else:
                        print(f"    Warning: No sample ID column found")
                        continue
                    
                    unique_patients = df['patient_id'].nunique()
                    print(f"    Unique patients: {unique_patients}")
                    
                    # Group by patient
                    for patient_id, group in df.groupby('patient_id'):
                        mutations = []
                        
                        for _, row in group.iterrows():
                            gene = row.get('gene', '')
                            effect = row.get('effect', '')
                            aa_change = row.get('Amino_Acid_Change', '')
                            
                            # Skip synonymous mutations
                            if 'synonymous' in str(effect).lower():
                                continue
                            
                            # Build mutation string
                            if aa_change and not pd.isna(aa_change) and str(aa_change) != '':
                                mutation_str = f"{gene} {aa_change}"
                            elif effect:
                                mutation_str = f"{gene} ({effect})"
                            else:
                                mutation_str = str(gene)
                            
                            mutations.append({
                                'gene': gene,
                                'variant_classification': effect,
                                'hgvsp': aa_change,
                                'mutation_str': mutation_str,
                                'vaf': row.get('dna_vaf', None)
                            })
                        
                        if mutations:
                            if patient_id in self.mutations:
                                self.mutations[patient_id].extend(mutations)
                            else:
                                self.mutations[patient_id] = mutations
                                
                except Exception as e:
                    print(f"    Error: {e}")
            
            print(f"  Loaded mutations for {len(self.mutations)} patients from TSV files")
            return
        
        # Layout 2: cBioPortal-format mutations
        cbioportal_mutations = DATA_DIR / "mutations_cbioportal"
        if cbioportal_mutations.exists():
            combined_file = cbioportal_mutations / "all_mutations.csv"
            if combined_file.exists():
                print(f"  Loading from cBioPortal format: {combined_file}")
                df = pd.read_csv(combined_file)
                
                for patient_id, group in df.groupby("patient_id"):
                    mutations = []
                    for _, row in group.iterrows():
                        gene = row.get("gene", "")
                        variant_class = row.get("variant_classification", "")
                        hgvsp = row.get("hgvsp", "")
                        
                        # Skip silent mutations
                        if "silent" in str(variant_class).lower() or "synonymous" in str(variant_class).lower():
                            continue
                        
                        # Build mutation string
                        if hgvsp and not pd.isna(hgvsp) and str(hgvsp) != "":
                            mutation_str = f"{gene} {hgvsp}"
                        else:
                            mutation_str = f"{gene} ({variant_class})"
                        
                        mutations.append({
                            "gene": gene,
                            "variant_classification": variant_class,
                            "hgvsp": hgvsp,
                            "mutation_str": mutation_str
                        })
                    
                    if mutations:
                        self.mutations[patient_id] = mutations
                
                print(f"  Loaded mutations for {len(self.mutations)} patients from cBioPortal")
                return
        
        # Layout 3: MAF format
        maf_files = list(MUTATION_DIR.glob("*.maf"))
        if not maf_files:
            print(f"  No mutation files found in {MUTATION_DIR}")
            return
        
        print(f"  Found {len(maf_files)} MAF files")
        
        for maf_file in maf_files:
            print(f"  Processing {maf_file.name}...")
            
            try:
                df = pd.read_csv(maf_file, sep='\t', comment='#', low_memory=False)
                print(f"    Total rows: {len(df)}")
                
                if "Tumor_Sample_Barcode" in df.columns:
                    df["patient_id"] = df["Tumor_Sample_Barcode"].str[:12]
                    unique_patients = df["patient_id"].nunique()
                    print(f"    Unique patients: {unique_patients}")
                else:
                    continue
                
                for patient_id, group in df.groupby("patient_id"):
                    mutations = []
                    
                    for _, row in group.iterrows():
                        gene = row.get("Hugo_Symbol", "")
                        variant_class = row.get("Variant_Classification", "")
                        hgvsp = row.get("HGVSp_Short", row.get("Amino_Acid_Change", ""))
                        
                        if variant_class in ["Silent", "Intron", "3'UTR", "5'UTR", "IGR", 
                                            "3'Flank", "5'Flank", "RNA"]:
                            continue
                        
                        if hgvsp and not pd.isna(hgvsp):
                            hgvsp_clean = str(hgvsp).replace("p.", "")
                            mutation_str = f"{gene} p.{hgvsp_clean}"
                        else:
                            mutation_str = f"{gene} ({variant_class})"
                        
                        mutations.append({
                            "gene": gene,
                            "variant_classification": variant_class,
                            "hgvsp": hgvsp,
                            "mutation_str": mutation_str
                        })
                    
                    if mutations:
                        if patient_id in self.mutations:
                            self.mutations[patient_id].extend(mutations)
                        else:
                            self.mutations[patient_id] = mutations
                        
            except Exception as e:
                print(f"    Error: {e}")
        
        print(f"  Total patients with mutations: {len(self.mutations)}")
    
    def _load_treatment_plans(self):
        """Load treatment plans."""
        print("\nLoading treatment plans...")
        
        # Load merged treatment plans if present
        combined_file = TREATMENT_PLAN_DIR / "all_treatment_plans.json"
        if combined_file.exists():
            with open(combined_file) as f:
                self.treatment_plans = json.load(f)
            print(f"  Loaded {len(self.treatment_plans)} patients with treatment plans")
        else:
            print("  Treatment plans file not found!")
    
    def get_drug_info(self, drug_name: str) -> dict:
        """Look up drug information."""
        # Normalize drug name
        drug_upper = drug_name.upper()
        if drug_upper in DRUG_SUBSTITUTIONS:
            drug_name = DRUG_SUBSTITUTIONS[drug_upper]
        
        # Look up in DrugBank (case-insensitive)
        drug_lower = drug_name.lower()
        
        if drug_lower in self.drugbank_data:
            info = self.drugbank_data[drug_lower]
            return {
                "description": self._extract_first_sentence(
                    info.get("description") or info.get("desc") or ""
                ),
                "mechanism_of_action": self._extract_first_sentence(
                    info.get("mechanism-of-action") or info.get("mechanism_of_action") or 
                    info.get("moa") or ""
                )
            }
        
        # Fall back to partial match
        for db_name, info in self.drugbank_data.items():
            if drug_lower in db_name or db_name in drug_lower:
                return {
                    "description": self._extract_first_sentence(
                        info.get("description") or info.get("desc") or ""
                    ),
                    "mechanism_of_action": self._extract_first_sentence(
                        info.get("mechanism-of-action") or info.get("mechanism_of_action") or 
                        info.get("moa") or ""
                    )
                }
        
        # Not found — return default placeholder
        return {
            "description": f"{drug_name} is an anticancer agent.",
            "mechanism_of_action": "Mechanism of action not available in database."
        }
    
    def _extract_first_sentence(self, text: str) -> str:
        """Extract the first sentence and remove citation markers."""
        if not text:
            return ""
        
        # Strip citation markers
        text = re.sub(r'\[[A-Z][0-9]+(?:,[A-Z][0-9]+)*\]', '', text)
        
        # Extract first sentence
        match = re.search(r'^(.*?[.!?])(?:\s|$)', text)
        if match:
            return match.group(1).strip()
        
        return text.split('\n')[0].strip()[:500]
    
    def calculate_tmb(self, patient_id: str) -> Optional[float]:
        """Compute tumor mutational burden (non-synonymous mutations per Mb)."""
        mutations = self.mutations.get(patient_id, [])
        if not mutations:
            return None
        
        # Non-synonymous mutation classes (supports MAF and effect/VEP labels)
        nonsynonymous_classes = [
            # MAF format
            "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
            "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
            "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation",
            # Effect format (VEP/SnpEff)
            "missense_variant", "stop_gained", "stop_lost", 
            "frameshift_variant", "inframe_insertion", "inframe_deletion",
            "splice_acceptor_variant", "splice_donor_variant",
            "start_lost", "protein_altering_variant"
        ]
        
        nonsynonymous_count = sum(
            1 for m in mutations
            if m.get("variant_classification") in nonsynonymous_classes or
               any(vc in str(m.get("variant_classification", "")).lower() 
                   for vc in ["missense", "nonsense", "frameshift", "stop_gained", 
                             "stop_lost", "splice", "inframe"])
        )
        
        # Exome size ~30 Mb
        tmb = nonsynonymous_count / 30.0
        return round(tmb, 2)
    
    def get_hr_her2_status(self, patient_id: str) -> Tuple[str, str]:
        """Resolve breast HR/HER2 status (primarily inferred from SUBTYPE)."""
        cbio = self.cbioportal_data.get(patient_id, {})
        
        # First try direct fields
        er_status = cbio.get("ER_STATUS_BY_IHC") or cbio.get("ER_STATUS") or ""
        pr_status = cbio.get("PR_STATUS_BY_IHC") or cbio.get("PR_STATUS") or ""
        her2_status = cbio.get("HER2_STATUS_BY_IHC") or cbio.get("HER2_FISH_STATUS") or cbio.get("HER2_STATUS") or ""
        
        hr_status = "Unknown"
        her2 = "Unknown"
        
        # If a direct status is available, use it
        if er_status or pr_status:
            er_pos = "positive" in str(er_status).lower() or "pos" in str(er_status).lower()
            pr_pos = "positive" in str(pr_status).lower() or "pos" in str(pr_status).lower()
            if er_pos or pr_pos:
                hr_status = "Positive"
            elif "negative" in str(er_status).lower() or "neg" in str(er_status).lower():
                hr_status = "Negative"
        
        if her2_status:
            if "positive" in str(her2_status).lower() or "amplified" in str(her2_status).lower():
                her2 = "Positive"
            elif "negative" in str(her2_status).lower() or "not amplified" in str(her2_status).lower():
                her2 = "Negative"
        
        # Otherwise infer from SUBTYPE (TCGA Pan-Cancer Atlas labels)
        subtype = str(cbio.get("SUBTYPE", "")).upper()
        
        if subtype:
            # PAM50 molecular subtypes
            if "LUMA" in subtype or "LUMINAL A" in subtype:
                # Luminal A: HR+/HER2-
                if hr_status == "Unknown":
                    hr_status = "Positive"
                if her2 == "Unknown":
                    her2 = "Negative"
            elif "LUMB" in subtype or "LUMINAL B" in subtype:
                # Luminal B: HR+, HER2 may be + or -
                if hr_status == "Unknown":
                    hr_status = "Positive"
                # Leave HER2 as Unknown unless explicitly known
            elif "HER2" in subtype and "ENRICHED" in subtype:
                # HER2-enriched: typically HR-/HER2+
                if her2 == "Unknown":
                    her2 = "Positive"
                if hr_status == "Unknown":
                    hr_status = "Negative"
            elif "BASAL" in subtype:
                # Basal-like: typically Triple Negative (HR-/HER2-)
                if hr_status == "Unknown":
                    hr_status = "Negative"
                if her2 == "Unknown":
                    her2 = "Negative"
            elif "NORMAL" in subtype:
                # Normal-like: cannot determine
                pass
        
        return hr_status, her2
    
    def get_gleason_score(self, patient_id: str) -> Optional[str]:
        """Resolve prostate Gleason score."""
        cbio = self.cbioportal_data.get(patient_id, {})
        clinical = self.clinical_data.get(patient_id, {})
        survival = self.survival_data.get(patient_id, {})
        
        # Try several attribute names (cBioPortal)
        gleason = cbio.get("GLEASON_SCORE") or cbio.get("GLEASON_SCORE_COMBINED")
        
        if gleason:
            return str(gleason)
        
        # Combine primary + secondary if available
        primary = cbio.get("GLEASON_PATTERN_PRIMARY") or cbio.get("GLEASON_PRIMARY")
        secondary = cbio.get("GLEASON_PATTERN_SECONDARY") or cbio.get("GLEASON_SECONDARY")
        if primary and secondary:
            return f"{primary}+{secondary}"
        
        # Pull from CDR or GDC clinical data
        cdr_gleason = survival.get("gleason_score") or clinical.get("gleason_score")
        if cdr_gleason:
            return str(cdr_gleason)
        
        # Infer from histological_grade (prostate-specific fallback)
        grade = clinical.get("tumor_grade") or survival.get("histological_grade", "")
        if grade and "gleason" in str(grade).lower():
            return str(grade)
        
        return None
    
    def build_prompt(self, patient_id: str, treatment_plan: Dict, prior_plans: list = None) -> Optional[Dict]:
        """
        Build a prompt for one patient × treatment regimen (MSK format).
        """
        # Pull clinical data
        clinical = self.clinical_data.get(patient_id, {})
        survival = self.survival_data.get(patient_id, {})
        cbio = self.cbioportal_data.get(patient_id, {})
        
        if not clinical and not survival:
            return None
        
        # Get drug list
        drugs = treatment_plan.get("drugs", [])
        drug_names = [d.get("agent", "") for d in drugs if d.get("agent")]
        
        if not drug_names:
            return None
        
        # Get cancer type
        project_id = clinical.get("project_id", "")
        cancer_type_code = survival.get("cancer_type", "")
        
        # ============================================
        # Build prompt (MSK format)
        # ============================================
        prompt = f"Patient ID: {patient_id}\n\n"
        prompt += f"Drug(s): {', '.join(d.upper() for d in drug_names)}\n\n"

        # ============================================
        # # Drug Information
        # ============================================
        prompt += "# Drug Information\n"
        for drug_name in drug_names:
            drug_info = self.get_drug_info(drug_name)
            prompt += f"## {drug_name.upper()}\n"
            prompt += f"Description: {drug_info['description']}\n"
            prompt += f"Mechanism of Action: {drug_info['mechanism_of_action']}\n\n"
        
        # ============================================
        # # Clinical and Diagnosis Information (combined)
        # ============================================
        prompt += "# Clinical and Diagnosis Information\n"
        
        # Gender
        gender = clinical.get("gender") or survival.get("gender") or cbio.get("SEX") or "Unknown"
        gender = gender.capitalize() if isinstance(gender, str) else "Unknown"
        prompt += f"Gender: {gender}\n"
        
        # Age (handle NaN properly)
        age = _safe_str(clinical.get("age_at_diagnosis")) or \
              _safe_str(survival.get("age_at_diagnosis")) or \
              _safe_str(cbio.get("AGE"))
        if age:
            try:
                age_val = float(age)
                if age_val > 365:
                    age_val = int(age_val / 365)
                prompt += f"Age: {float(age_val)}\n"
            except (ValueError, TypeError):
                prompt += "Age: Unknown\n"
        else:
            prompt += "Age: Unknown\n"
        
        # Race
        race = clinical.get("race") or survival.get("race") or cbio.get("RACE") or "Unknown"
        prompt += f"Race: {race.title() if isinstance(race, str) else 'Unknown'}\n"
        
        # Ethnicity (standardized to MSK format)
        ethnicity_raw = _safe_str(clinical.get("ethnicity")) or _safe_str(cbio.get("ETHNICITY")) or "Unknown"
        ethnicity = ETHNICITY_MAP.get(ethnicity_raw.lower(), ethnicity_raw) if ethnicity_raw else "Unknown"
        prompt += f"Ethnicity: {ethnicity}\n"

        # Smoking History (standardize to MSK format, fix NaN bug)
        smoking_raw = _safe_str(clinical.get("tobacco_smoking_status"))
        if smoking_raw:
            smoking = SMOKING_HISTORY_MAP.get(smoking_raw.lower(), smoking_raw)
        else:
            smoking = "Unknown"
        prompt += f"Smoking History: {smoking}\n"

        # Stage (at diagnosis) - grouped to match MSK training format
        stage_raw = (_safe_str(clinical.get("ajcc_pathologic_stage")) or
                     _safe_str(survival.get("ajcc_stage")) or
                     _safe_str(cbio.get("AJCC_PATHOLOGIC_TUMOR_STAGE")))
        stage_grouped = standardize_stage(stage_raw)
        prompt += f"Stage (at diagnosis): {stage_grouped}\n\n"

        # MMR Status (derive from MSI; MSI-H → MMR deficient)
        prompt += "## MMR Status\n"
        mantis_pre = cbio.get("MSI_SCORE_MANTIS")
        sensor_pre = cbio.get("MSI_SENSOR_SCORE")
        msi_type_pre, _ = classify_msi(mantis_pre, sensor_pre)
        if msi_type_pre == "Instable":
            prompt += "MMR Absent: True\n"
        elif msi_type_pre == "Stable":
            prompt += "MMR Absent: False\n"
        prompt += "\n"

        # Laboratory Results (TCGA lacks lab data; empty section matches MSK structure)
        prompt += "## Laboratory Results\n\n"

        # Prior Medication Status and Prior Treatments (inferred from plan ordering)
        if prior_plans:
            prior_drug_names = []
            for pp in prior_plans:
                for d in pp.get("drugs", []):
                    agent = d.get("agent", "")
                    if agent:
                        prior_drug_names.append(agent.upper())
            prompt += "Prior Medication Status: Yes\n"
            if prior_drug_names:
                prompt += f"Prior Treatments: {', '.join(prior_drug_names)}\n"
        else:
            prompt += "Prior Medication Status: No\n"
        
        # HR/HER2 status (breast cancer)
        if "BRCA" in project_id or cancer_type_code == "BRCA":
            hr_status, her2_status = self.get_hr_her2_status(patient_id)
            prompt += f"HR Status: {hr_status}\n"
            prompt += f"HER2 Status: {her2_status}\n"
        
        # Gleason score (prostate cancer)
        if "PRAD" in project_id or cancer_type_code == "PRAD":
            gleason = self.get_gleason_score(patient_id)
            if gleason:
                prompt += f"Gleason Score: {gleason}\n"
        
        prompt += "\n"
        
        # Diagnosis Description (MSK format with ICD-O codes)
        primary_diagnosis = _safe_str(clinical.get("primary_diagnosis")) or \
                           _safe_str(survival.get("histological_type")) or "Unknown"
        primary_site = _safe_str(clinical.get("primary_site")) or "Unknown"
        # Try to get ICD-O codes from GDC clinical data
        morphology_code = _safe_str(clinical.get("morphology")) or _safe_str(clinical.get("icd_o_3_histology"))
        site_code = _safe_str(clinical.get("tissue_or_organ_of_origin")) or _safe_str(clinical.get("icd_o_3_site"))
        if morphology_code and site_code:
            prompt += f"Diagnosis Description: {primary_diagnosis.upper()} | {primary_site.upper()} ({morphology_code} | {site_code})\n"
        elif morphology_code:
            prompt += f"Diagnosis Description: {primary_diagnosis.upper()} | {primary_site.upper()} ({morphology_code})\n"
        else:
            # Fallback: use common ICD-O code mapping for known TCGA histologies
            icdo_map = {
                "adenocarcinoma, nos": "M8140/3",
                "infiltrating duct carcinoma, nos": "M8500/3",
                "squamous cell carcinoma, nos": "M8070/3",
                "lobular carcinoma, nos": "M8520/3",
                "mucinous adenocarcinoma": "M8480/3",
                "papillary adenocarcinoma, nos": "M8260/3",
                "acinar cell carcinoma": "M8550/3",
            }
            site_code_map = {
                "bronchus and lung": "C34.9", "breast, nos": "C50.9",
                "colon, nos": "C18.9", "rectum, nos": "C20.9",
                "prostate gland": "C61.9", "pancreas, nos": "C25.9",
                "ascending colon": "C18.2", "sigmoid colon": "C18.7",
                "transverse colon": "C18.4", "descending colon": "C18.6",
                "cecum": "C18.0", "rectosigmoid junction": "C19.9",
                "hepatic flexure of colon": "C18.3", "splenic flexure of colon": "C18.5",
                "upper lobe, lung": "C34.1", "lower lobe, lung": "C34.3",
                "middle lobe, lung": "C34.2", "overlapping lesion of lung": "C34.8",
            }
            m_code = icdo_map.get(primary_diagnosis.lower(), "")
            s_code = site_code_map.get(primary_site.lower(), "")
            if m_code and s_code:
                prompt += f"Diagnosis Description: {primary_diagnosis.upper()} | {primary_site.upper()} ({m_code} | {s_code})\n"
            elif m_code:
                prompt += f"Diagnosis Description: {primary_diagnosis.upper()} | {primary_site.upper()} ({m_code})\n"
            else:
                prompt += f"Diagnosis Description: {primary_diagnosis.upper()} | {primary_site.upper()}\n"

        # AJCC Stage (raw value)
        if stage_raw:
            prompt += f"AJCC Stage: {stage_raw}\n"

        # Clinical Group + Pathological Group - derived from raw AJCC stage suffix (e.g. Stage IIB → 2B)
        # MSK has Clinical Group in ~85% of samples; derive from clinical_stage or AJCC pathologic
        clinical_stage_raw = _safe_str(survival.get("clinical_stage")) or _safe_str(clinical.get("ajcc_clinical_stage"))
        if clinical_stage_raw:
            cs = str(clinical_stage_raw).upper().replace("STAGE ", "").strip()
            for rom, num in [("IV", "4"), ("III", "3"), ("II", "2"), ("I", "1")]:
                if cs.startswith(rom):
                    suffix = cs[len(rom):]
                    prompt += f"Clinical Group: {num}{suffix}\n"
                    break
        elif stage_raw:
            # Fallback: use pathologic stage as Clinical Group (better than omitting)
            s = str(stage_raw).upper().replace("STAGE ", "").strip()
            for rom, num in [("IV", "4"), ("III", "3"), ("II", "2"), ("I", "1")]:
                if s.startswith(rom):
                    suffix = s[len(rom):]
                    prompt += f"Clinical Group: {num}{suffix}\n"
                    break

        if stage_raw:
            s = str(stage_raw).upper().replace("STAGE ", "").strip()
            for rom, num in [("IV", "4"), ("III", "3"), ("II", "2"), ("I", "1")]:
                if s.startswith(rom):
                    suffix = s[len(rom):]
                    prompt += f"Pathological Group: {num}{suffix}\n"
                    break

        # Derived Stage (grouped)
        prompt += f"Derived Stage: {stage_grouped}\n"

        # Summary (mapped to MSK-style detailed descriptions)
        if stage_raw and ("IV" in str(stage_raw).upper() or "4" in str(stage_raw)):
            summary = "Distant metastases/systemic disease"
        elif stage_raw and ("III" in str(stage_raw).upper() or "3" in str(stage_raw)):
            summary = "Regional, lymph nodes only"
        elif stage_raw and ("II" in str(stage_raw).upper() or "2" in str(stage_raw)):
            summary = "Regional, direct extension only"
        elif stage_raw and ("I" in str(stage_raw).upper() or "1" in str(stage_raw) or "0" in str(stage_raw)):
            summary = "Localized"
        else:
            summary = "Unknown"
        prompt += f"Summary: {summary}\n"
        
        # ============================================
        # # Sample and Genetic Profile (combined)
        # ============================================
        prompt += "\n# Sample and Genetic Profile\n"
        
        # Sample info (add temporal reference matching MSK format: "## Sample P-XXX (-29 days before treatment)")
        sample_id = f"{patient_id}-01A"  # TCGA standard format
        # TCGA samples are typically collected at diagnosis (before treatment)
        prompt += f"\n## Sample {sample_id} (at diagnosis)\n"
        
        # Cancer Type
        cancer_type_map = {
            "TCGA-LUAD": ("Lung Cancer", "Lung Adenocarcinoma"),
            "TCGA-LUSC": ("Lung Cancer", "Lung Squamous Cell Carcinoma"),
            "TCGA-BRCA": ("Breast Cancer", "Invasive Breast Carcinoma"),
            "TCGA-COAD": ("Colorectal Cancer", "Colon Adenocarcinoma"),
            "TCGA-READ": ("Colorectal Cancer", "Rectal Adenocarcinoma"),
            "TCGA-PRAD": ("Prostate Cancer", "Prostate Adenocarcinoma"),
            "TCGA-PAAD": ("Pancreatic Cancer", "Pancreatic Adenocarcinoma")
        }
        cancer_type_info = cancer_type_map.get(project_id, ("Unknown", primary_diagnosis))
        prompt += f"Cancer Type: {cancer_type_info[0]}\n"
        
        # Cancer Type Detailed (full name, mapped from subtype code)
        subtype = cbio.get("SUBTYPE")
        if subtype and subtype in CANCER_TYPE_DETAILED_MAP:
            cancer_type_detailed = CANCER_TYPE_DETAILED_MAP[subtype]
        elif subtype:
            prefix = subtype.split("_")[0]
            cancer_type_detailed = CANCER_TYPE_DETAILED_MAP.get(prefix, cancer_type_info[1])
        else:
            cancer_type_detailed = cancer_type_info[1]
        prompt += f"Cancer Type Detailed: {cancer_type_detailed}\n"

        # Primary Site
        prompt += f"Primary Site: {primary_site.title()}\n"

        # Sample type (TCGA samples are mostly Primary)
        prompt += "Sample Type: Primary\n"

        # MSI Status (numeric score + text classification, matching MSK format)
        mantis_score = cbio.get("MSI_SCORE_MANTIS")
        sensor_score = cbio.get("MSI_SENSOR_SCORE")
        msi_type, msi_score = classify_msi(mantis_score, sensor_score)
        msi_raw_status = cbio.get("MSI_STATUS")
        if msi_raw_status:
            prompt += f"MSI Comment: {msi_raw_status}\n"
        else:
            prompt += "MSI Comment: Not Available\n"
        if msi_score is not None:
            prompt += f"MSI Score: {msi_score}\n"
        prompt += f"MSI Type: {msi_type}\n"
        
        # TMB
        tmb = self.calculate_tmb(patient_id)
        if tmb is not None:
            prompt += f"TMB (nonsynonymous): {tmb:.2f}\n"
        
        # ============================================
        # ## Mutations
        # ============================================
        prompt += "\n## Mutations\n"

        mutations = self.mutations.get(patient_id, [])
        if mutations:
            # Add sample reference line before listing mutations (matches MSK format)
            prompt += f"Sample {sample_id} (at diagnosis):\n"

            # Prioritize mutations in important cancer genes
            important_mutations = [m for m in mutations if m["gene"] in IMPORTANT_GENES]
            other_mutations = [m for m in mutations if m["gene"] not in IMPORTANT_GENES]

            # Combine mutation lists
            mutation_strs = [m["mutation_str"] for m in important_mutations]

            if len(mutation_strs) < 15:
                remaining = 15 - len(mutation_strs)
                mutation_strs.extend([m["mutation_str"] for m in other_mutations[:remaining]])

            prompt += f"{', '.join(mutation_strs[:18])}\n"
        else:
            prompt += "No mutations detected\n"
        
        # ============================================
        # ## Structural Variants
        # ============================================
        prompt += "\n## Structural Variants\n"
        prompt += "No structural variants detected\n"
        
        # ============================================
        # Build output record (without Prediction Task section)
        # ============================================
        return {
            "patient_id": patient_id,
            "plan_id": treatment_plan.get("plan_id", "plan_1"),
            "drug_name": ", ".join(drug_names),
            "instruction": INSTRUCTION,
            "input": prompt,
            "score": None,
            # Attach survival data
            "os_time": survival.get("OS.time"),
            "os_event": survival.get("OS"),
            "pfi_time": survival.get("PFI.time"),
            "pfi_event": survival.get("PFI"),
            "cancer_type": project_id or cancer_type_code,
            "regimen_type": treatment_plan.get("regimen_type", "Unknown"),
            "regimen_name": treatment_plan.get("regimen_name", "Unknown")
        }
    
    def build_all_prompts(self) -> List[Dict]:
        """Build prompts for all patients."""
        prompts = []
        
        # Patients with treatment plans
        patients_with_plans = set(self.treatment_plans.keys())
        # Patients with clinical or survival data
        patients_with_data = set(self.clinical_data.keys()) | set(self.survival_data.keys())
        # Intersect
        valid_patients = patients_with_plans & patients_with_data
        
        print(f"\nBuilding prompts for {len(valid_patients)} patients...")
        
        for patient_id in tqdm(valid_patients, desc="Building prompts"):
            plans = self.treatment_plans.get(patient_id, [])
            
            for plan_idx, plan in enumerate(plans):
                prompt_data = self.build_prompt(patient_id, plan, prior_plans=plans[:plan_idx])
                if prompt_data:
                    prompts.append(prompt_data)
        
        print(f"Generated {len(prompts)} prompts")
        return prompts
    
    def save_prompts(self, prompts: List[Dict], filename: str = "tcga_prompts.jsonl"):
        """Save prompts as JSONL."""
        
        # Inference JSONL
        inference_file = self.output_dir / filename
        metadata_file = self.output_dir / filename.replace(".jsonl", "_metadata.csv")
        
        inference_prompts = []
        metadata_records = []
        
        for p in prompts:
            # Inference format (matches MSK; includes regimen_id/patient_id/cancer_type for the inference script)
            regimen_id = f"{p['patient_id']}_{p['plan_id']}"
            inference_prompts.append({
                "instruction": p["instruction"],
                "input": p["input"],
                "output": "",
                "patient_id": p["patient_id"],
                "regimen_id": regimen_id,
                "cancer_type": p["cancer_type"],
            })
            
            # Metadata
            metadata_records.append({
                "patient_id": p["patient_id"],
                "plan_id": p["plan_id"],
                "drug_name": p["drug_name"],
                "os_time": p["os_time"],
                "os_event": p["os_event"],
                "pfi_time": p["pfi_time"],
                "pfi_event": p["pfi_event"],
                "cancer_type": p["cancer_type"],
                "regimen_type": p["regimen_type"],
                "regimen_name": p["regimen_name"]
            })
        
        # SaveJSONL
        with open(inference_file, 'w') as f:
            for item in inference_prompts:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved inference prompts to: {inference_file}")
        
        # Save metadata CSV
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_csv(metadata_file, index=False)
        print(f"Saved metadata to: {metadata_file}")
        
        # Save a sample prompt for inspection
        if prompts:
            sample_file = self.output_dir / "sample_prompt.txt"
            with open(sample_file, 'w') as f:
                f.write(prompts[0]["input"])
            print(f"Saved sample prompt to: {sample_file}")
        
        return inference_file, metadata_file


def main():
    print("=" * 60)
    print("TCGA Prompt Builder (MSK-Compatible Format)")
    print("=" * 60)
    
    builder = TCGAPromptBuilder()
    
    # Load all data
    builder.load_all_data()
    
    # Build prompts
    prompts = builder.build_all_prompts()
    
    # Save
    builder.save_prompts(prompts)
    
    # Summary stats
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total prompts generated: {len(prompts)}")
    
    # Count by cancer type
    cancer_counts = defaultdict(int)
    regimen_counts = defaultdict(int)
    for p in prompts:
        cancer_counts[p["cancer_type"]] += 1
        regimen_counts[p["regimen_type"]] += 1
    
    print("\nBy cancer type:")
    for ct, count in sorted(cancer_counts.items(), key=lambda x: -x[1]):
        print(f"  {ct}: {count}")
    
    print("\nBy regimen type:")
    for rt, count in sorted(regimen_counts.items(), key=lambda x: -x[1]):
        print(f"  {rt}: {count}")


if __name__ == "__main__":
    main()