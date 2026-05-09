"""
Step 1: Generate drug response prediction prompts from MSK-CHORD data.

For each patient-treatment plan pair:
  - Classifies drugs (substitution, manual descriptions, DrugBank lookup)
  - Extracts pre-treatment clinical features (mutations, SVs, labs, status)
  - Assigns RECIST-derived response score
  - Emits a structured prompt with all patient context
"""
import pandas as pd
import json
import os
import re
import concurrent.futures
import glob
from tqdm import tqdm
from collections import defaultdict

from config import (
    BASE_DIR, PATIENT_FILE, SAMPLE_FILE, MUTATION_FILE, SV_FILE,
    DIAGNOSIS_FILE, TREATMENT_FILE, PRIOR_MEDS_FILE, DRUG_RESPONSE_FILE,
    SPECIMEN_SURGERY_FILE, PDL1_FILE, MMR_FILE,
    GLEASON_FILE, CA_15_3_FILE, CA_19_9_FILE, CEA_FILE, DRUGBANK_FILE,
    TREATMENT_REGIMENS_DIR, RAW_PROMPTS,
)
from utils import (
    extract_days_from_string, read_tsv_file,
    DRUG_SUBSTITUTIONS, MANUAL_DESCRIPTIONS, DRUGS_TO_SKIP_INFO, RECIST_SCORE_MAP,
)

DEBUG_LIMIT = 500


def extract_first_sentence(text):
    """Extract the first sentence from a text and remove citations."""
    if not text:
        return ""
    text = re.sub(r'\[[A-Z][0-9]+(?:,[A-Z][0-9]+)*\]', '', text)
    match = re.search(r'^(.*?[.!?])(?:\s|$)', text)
    if match:
        return match.group(1).strip()
    return text.split('\n')[0].strip()


# ============================================================
# Data loading
# ============================================================
def load_data():
    """Load all 16 clinical data files + DrugBank in parallel."""
    print("Loading data files...")

    data_files = {
        'patient_df': PATIENT_FILE,
        'sample_df': SAMPLE_FILE,
        'mutation_df': MUTATION_FILE,
        'sv_df': SV_FILE,
        'diagnosis_df': DIAGNOSIS_FILE,
        'treatment_df': TREATMENT_FILE,
        'prior_meds_df': PRIOR_MEDS_FILE,
        'drug_response_df': DRUG_RESPONSE_FILE,
        'specimen_surgery_df': SPECIMEN_SURGERY_FILE,
        'pdl1_df': PDL1_FILE,
        'mmr_df': GLEASON_FILE.replace('gleason', 'mmr'),  # use MMR_FILE
        'gleason_df': GLEASON_FILE,
        'ca_15_3_df': CA_15_3_FILE,
        'ca_19_9_df': CA_19_9_FILE,
        'cea_df': CEA_FILE,
    }
    # Fix: use actual config paths
    data_files['mmr_df'] = MMR_FILE

    results = {}
    with tqdm(total=len(data_files) + 1, desc="Loading files") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            def _read(key, path):
                if key == 'drug_response_df':
                    return pd.read_csv(path)
                return read_tsv_file(path)

            future_to_key = {
                executor.submit(_read, key, path): key
                for key, path in data_files.items()
            }
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                    print(f"Loaded {key}: {results[key].shape}")
                except Exception as e:
                    print(f"Error loading {key}: {e}")
                    results[key] = pd.DataFrame()
                pbar.update(1)

        # Load DrugBank
        try:
            with open(DRUGBANK_FILE, 'r') as f:
                results['drugbank_data'] = json.load(f)
                print(f"Loaded drugbank_data: {len(results['drugbank_data'])} entries")
        except Exception as e:
            print(f"Error loading drugbank data: {e}")
            results['drugbank_data'] = []
        pbar.update(1)

    print("Data loading complete.")
    return results


def load_treatment_regimens():
    """Load all treatment regimen JSON files from API responses."""
    regimen_files = glob.glob(os.path.join(TREATMENT_REGIMENS_DIR, "*_api_response.json"))
    regimens = {}
    for fp in tqdm(regimen_files, desc="Loading treatment regimens"):
        try:
            patient_id = os.path.basename(fp).split('_')[0]
            with open(fp, 'r') as f:
                data = json.load(f)
            treatment_data = data.get('extracted_json', data)
            regimens[patient_id] = treatment_data
        except Exception as e:
            print(f"Error loading {fp}: {e}")
    print(f"Loaded {len(regimens)} patient treatment regimens")
    return regimens


# ============================================================
# Clinical feature extraction helpers
# ============================================================
def find_closest_diagnosis(patient_id, treatment_date, diagnosis_df):
    """Find the diagnosis closest to the treatment date."""
    patient_diagnoses = diagnosis_df[diagnosis_df['PATIENT_ID'] == patient_id].copy()
    if patient_diagnoses.empty:
        return None
    patient_diagnoses.loc[:, 'time_diff'] = abs(patient_diagnoses['START_DATE'] - treatment_date)
    return patient_diagnoses.loc[patient_diagnoses['time_diff'].idxmin()]


def find_pre_treatment_samples(patient_id, treatment_date, specimen_surgery_df,
                               max_days_after=3, max_results=3):
    """Find samples before treatment (or up to max_days_after treatment)."""
    patient_specimens = specimen_surgery_df[specimen_surgery_df['PATIENT_ID'] == patient_id].copy()
    if patient_specimens.empty:
        return []
    valid = patient_specimens[patient_specimens['START_DATE'] <= (treatment_date + max_days_after)].copy()
    if valid.empty:
        return []
    valid.loc[:, 'time_diff'] = abs(valid['START_DATE'] - treatment_date)
    valid = valid.sort_values(['time_diff'])

    results = []
    for _, specimen in valid.head(max_results).iterrows():
        td = specimen['START_DATE'] - treatment_date
        timing = f"{int(td)} days before treatment" if td < 0 else f"{int(td)} days after treatment start"
        results.append({'sample_id': specimen['SAMPLE_ID'], 'start_date': specimen['START_DATE'],
                        'time_diff': td, 'timing_desc': timing})
    return results


def find_pre_treatment_lab_results(patient_id, treatment_date, lab_df, lab_type,
                                   max_days_after=3, max_results=3):
    """Find lab results before treatment (or up to max_days_after treatment)."""
    if lab_df.empty:
        return []
    patient_labs = lab_df[lab_df['PATIENT_ID'] == patient_id].copy()
    if patient_labs.empty:
        return []
    valid = patient_labs[patient_labs['START_DATE'] <= (treatment_date + max_days_after)].copy()
    if valid.empty:
        return []
    valid.loc[:, 'time_diff'] = abs(valid['START_DATE'] - treatment_date)
    valid = valid.sort_values(['time_diff'])

    results = []
    for _, lab in valid.head(max_results).iterrows():
        td = lab['START_DATE'] - treatment_date
        timing = f"{int(td)} days before treatment" if td < 0 else f"{int(td)} days after treatment start"
        results.append({'test': lab_type, 'result': lab.get('RESULT', None),
                        'unit': lab.get('LR_UNIT_MEASURE', ''), 'date': lab['START_DATE'],
                        'time_diff': td, 'timing_desc': timing})
    return results


def find_pre_treatment_status(patient_id, treatment_date, status_df, status_type,
                              max_days_after=3, max_results=3):
    """Find status records before treatment (or up to max_days_after treatment)."""
    if status_df.empty:
        return []
    patient_status = status_df[status_df['PATIENT_ID'] == patient_id].copy()
    if patient_status.empty:
        return []
    valid = patient_status[patient_status['START_DATE'] <= (treatment_date + max_days_after)].copy()
    if valid.empty:
        return []
    valid.loc[:, 'time_diff'] = abs(valid['START_DATE'] - treatment_date)
    valid = valid.sort_values(['time_diff'])

    results = []
    for _, status in valid.head(max_results).iterrows():
        td = status['START_DATE'] - treatment_date
        timing = f"{int(td)} days before treatment" if td < 0 else f"{int(td)} days after treatment start"
        result_dict = {'type': status_type, 'date': status['START_DATE'],
                       'time_diff': td, 'timing_desc': timing}
        if status_type == 'PD-L1':
            result_dict['positive'] = status.get('PDL1_POSITIVE', 'Unknown')
        elif status_type == 'MMR':
            result_dict['mmr_absent'] = status.get('MMR_ABSENT', 'Unknown')
        results.append(result_dict)
    return results


def get_mutations_with_timing(sample_info_list, mutation_df):
    """Get mutations for multiple samples with timing information."""
    all_mutations = []
    for si in sample_info_list:
        sample_mutations = mutation_df[mutation_df['Tumor_Sample_Barcode'] == si['sample_id']]
        mutations = []
        for _, row in sample_mutations.iterrows():
            gene = row['Hugo_Symbol']
            mut = row['HGVSp_Short'] if not pd.isna(row['HGVSp_Short']) else row['Variant_Classification']
            mutations.append(f"{gene} {mut}")
        if mutations:
            all_mutations.append({'sample_id': si['sample_id'], 'mutations': mutations,
                                  'timing_desc': si['timing_desc']})
    return all_mutations


def get_structural_variants_with_timing(sample_info_list, sv_df):
    """Get structural variants for multiple samples with timing information."""
    all_svs = []
    for si in sample_info_list:
        sample_svs = sv_df[sv_df['Sample_Id'] == si['sample_id']]
        svs = []
        for _, row in sample_svs.iterrows():
            s1 = row['Site1_Hugo_Symbol']
            s2 = row['Site2_Hugo_Symbol']
            cls = row['Class']
            sv_info = f"{s1} {cls}" if s1 == s2 else f"{s1}-{s2} {cls}"
            event = row.get('Event_Info', '')
            if event and not pd.isna(event):
                sv_info += f" ({event})"
            svs.append(sv_info)
        if svs:
            all_svs.append({'sample_id': si['sample_id'], 'structural_variants': svs,
                            'timing_desc': si['timing_desc']})
    return all_svs


def get_sample_info_with_timing(sample_info_list, sample_df):
    """Get sample information for multiple samples with timing."""
    results = []
    for si in sample_info_list:
        sample_data = sample_df[sample_df['SAMPLE_ID'] == si['sample_id']]
        if not sample_data.empty:
            results.append({'sample_id': si['sample_id'], 'sample_data': sample_data.iloc[0],
                            'timing_desc': si['timing_desc']})
    return results


def get_prior_medication_status(patient_id, treatment_date, prior_meds_df, treatment_df):
    """Get prior medication status for a patient before a specific treatment."""
    patient_prior_meds = prior_meds_df[prior_meds_df['PATIENT_ID'] == patient_id]
    prior_to_msk = "Unknown"
    if not patient_prior_meds.empty:
        for _, row in patient_prior_meds.iterrows():
            if 'PRIOR_MED_TO_MSK' in row:
                if row['PRIOR_MED_TO_MSK'] == 'Prior medications to MSK':
                    prior_to_msk = "Yes"
                    break
                elif row['PRIOR_MED_TO_MSK'] == 'No prior medications':
                    prior_to_msk = "No"
                    break

    filtered_df = treatment_df[~treatment_df['AGENT'].isin(DRUGS_TO_SKIP_INFO)]
    patient_treatments = filtered_df[filtered_df['PATIENT_ID'] == patient_id]
    prior_treatments = patient_treatments[patient_treatments['START_DATE'] < treatment_date]
    prior_msk_drugs = prior_treatments['AGENT'].unique().tolist() if not prior_treatments.empty else []

    return prior_to_msk, prior_msk_drugs


def get_gleason_score(patient_id, treatment_date, gleason_df):
    """Get the closest Gleason score for prostate cancer patients."""
    if gleason_df.empty:
        return None
    patient_gleason = gleason_df[gleason_df['PATIENT_ID'] == patient_id].copy()
    if patient_gleason.empty:
        return None
    patient_gleason.loc[:, 'time_diff'] = abs(patient_gleason['START_DATE'] - treatment_date)
    return patient_gleason.loc[patient_gleason['time_diff'].idxmin()].get('GLEASON_SCORE', None)


def get_breast_cancer_status(patient_info):
    """Get HR and HER2 status for breast cancer patients."""
    hr = "Positive" if patient_info.get('HR') == 'Yes' else ("Negative" if patient_info.get('HR') == 'No' else "Unknown")
    her2 = "Positive" if patient_info.get('HER2') == 'Yes' else ("Negative" if patient_info.get('HER2') == 'No' else "Unknown")
    return hr, her2


# ============================================================
# Drug classification
# ============================================================
def get_drug_response_score(patient_id, drug_combination, drug_response_df):
    """Get drug response score by matching patient + drug combination."""
    patient_responses = drug_response_df[drug_response_df['PATIENT_ID'] == patient_id]
    if patient_responses.empty:
        return -1.0

    sorted_combo = '+'.join(sorted(drug_combination.split('+')))

    # Exact match
    for _, row in patient_responses.iterrows():
        row_combo = '+'.join(sorted(row['DRUG_COMBINATION'].split('+')))
        if row_combo == sorted_combo:
            response = row['RECIST_RESPONSE']
            if response in RECIST_SCORE_MAP:
                return RECIST_SCORE_MAP[response]

    # Partial match (at least one drug in common)
    best_score = -1.0
    for _, row in patient_responses.iterrows():
        response = row['RECIST_RESPONSE']
        if response in RECIST_SCORE_MAP:
            drugs_in_combo = set(row['DRUG_COMBINATION'].split('+'))
            drugs_in_treatment = set(drug_combination.split('+'))
            if drugs_in_combo & drugs_in_treatment:
                best_score = max(best_score, RECIST_SCORE_MAP[response])

    return best_score


def get_drug_info(drug_name, drugbank_data):
    """Get drug information from DrugBank with substitution and manual overrides."""
    if drug_name in DRUG_SUBSTITUTIONS:
        drug_name = DRUG_SUBSTITUTIONS[drug_name]

    if drug_name in MANUAL_DESCRIPTIONS:
        return MANUAL_DESCRIPTIONS[drug_name]

    # Handle combination drugs
    if "+" in drug_name or "and" in drug_name.lower():
        parts = [d.strip() for d in drug_name.split("+")] if "+" in drug_name else [d.strip() for d in drug_name.lower().split("and")]
        descriptions, mechanisms = [], []
        for drug in parts:
            info = get_drug_info(drug, drugbank_data)
            if not info['description'].startswith("No information"):
                descriptions.append(f"{drug}: {info['description']}")
            if not info['mechanism_of_action'].startswith("No information"):
                mechanisms.append(f"{drug}: {info['mechanism_of_action']}")
        if descriptions or mechanisms:
            return {
                'description': " ".join(descriptions) if descriptions else f"No information available for {drug_name}.",
                'mechanism_of_action': " ".join(mechanisms) if mechanisms else f"No information available for {drug_name}.",
            }

    # DrugBank lookup
    normalized = drug_name.lower()
    for drug in drugbank_data:
        if drug['drug_name'].lower() == normalized:
            return {
                'description': extract_first_sentence(drug.get('description', '')),
                'mechanism_of_action': extract_first_sentence(drug.get('mechanism_of_action', '')) or f"No information available for {drug_name}.",
            }

    # Partial match
    for drug in drugbank_data:
        if normalized in drug['drug_name'].lower() or drug['drug_name'].lower() in normalized:
            return {
                'description': extract_first_sentence(drug.get('description', '')),
                'mechanism_of_action': extract_first_sentence(drug.get('mechanism_of_action', '')) or f"No information available for {drug_name}.",
            }

    return {
        'description': f"No information available for {drug_name}.",
        'mechanism_of_action': f"No information available for {drug_name}.",
    }


# ============================================================
# Check for meaningful data
# ============================================================
def has_meaningful_data(mutations_list, svs_list, lab_results_list, pdl1_list, mmr_list):
    """Check if the patient has meaningful data for analysis."""
    has_mutations = any(len(m.get('mutations', [])) > 0 for m in mutations_list)
    has_svs = any(len(sv.get('structural_variants', [])) > 0 for sv in svs_list)
    has_labs = any(len(lab_list) > 0 for lab_list in lab_results_list)
    has_status = len(pdl1_list) > 0 or len(mmr_list) > 0
    return has_mutations or has_svs or has_labs or has_status


# ============================================================
# Prompt generation (per patient-treatment plan)
# ============================================================
def generate_prompt_from_treatment_plan(args):
    """Generate a structured prompt for a single patient-treatment plan pair."""
    patient_id, treatment_plan, data_dict = args

    try:
        patient_df = data_dict['patient_df']
        sample_df = data_dict['sample_df']
        mutation_df = data_dict['mutation_df']
        sv_df = data_dict['sv_df']
        diagnosis_df = data_dict['diagnosis_df']
        treatment_df = data_dict['treatment_df']
        prior_meds_df = data_dict['prior_meds_df']
        drug_response_df = data_dict['drug_response_df']
        specimen_surgery_df = data_dict['specimen_surgery_df']
        pdl1_df = data_dict['pdl1_df']
        mmr_df = data_dict['mmr_df']
        gleason_df = data_dict['gleason_df']
        ca_15_3_df = data_dict['ca_15_3_df']
        ca_19_9_df = data_dict['ca_19_9_df']
        cea_df = data_dict['cea_df']
        drugbank_data = data_dict['drugbank_data']

        patient_info = patient_df[patient_df['PATIENT_ID'] == patient_id]
        if patient_info.empty:
            return None
        patient_info = patient_info.iloc[0]

        drugs = [drug["agent"] for drug in treatment_plan.get("drugs", [])]
        if not drugs:
            return None

        treatment_date = extract_days_from_string(treatment_plan.get("start_date", 0))

        # Pre-treatment samples and features
        pre_treatment_samples = find_pre_treatment_samples(patient_id, treatment_date, specimen_surgery_df)
        mutations_with_timing = get_mutations_with_timing(pre_treatment_samples, mutation_df)
        svs_with_timing = get_structural_variants_with_timing(pre_treatment_samples, sv_df)
        sample_info_with_timing = get_sample_info_with_timing(pre_treatment_samples, sample_df)

        # Lab results
        cea_results = find_pre_treatment_lab_results(patient_id, treatment_date, cea_df, "CEA")
        ca_15_3_results = find_pre_treatment_lab_results(patient_id, treatment_date, ca_15_3_df, "CA 15-3")
        ca_19_9_results = find_pre_treatment_lab_results(patient_id, treatment_date, ca_19_9_df, "CA 19-9")

        # Status records
        pdl1_results = find_pre_treatment_status(patient_id, treatment_date, pdl1_df, "PD-L1")
        mmr_results = find_pre_treatment_status(patient_id, treatment_date, mmr_df, "MMR")

        if not has_meaningful_data(mutations_with_timing, svs_with_timing,
                                   [cea_results, ca_15_3_results, ca_19_9_results],
                                   pdl1_results, mmr_results):
            return None

        closest_diagnosis = find_closest_diagnosis(patient_id, treatment_date, diagnosis_df)
        prior_to_msk, prior_msk_drugs = get_prior_medication_status(patient_id, treatment_date, prior_meds_df, treatment_df)

        # Cancer-specific features
        gleason_score = None
        hr_status, her2_status = "Unknown", "Unknown"
        if sample_info_with_timing:
            for si in sample_info_with_timing:
                if 'CANCER_TYPE' in si['sample_data']:
                    ct = str(si['sample_data']['CANCER_TYPE']).lower()
                    if 'prostate' in ct:
                        gleason_score = get_gleason_score(patient_id, treatment_date, gleason_df)
                    if 'breast' in ct:
                        hr_status, her2_status = get_breast_cancer_status(patient_info)

        # Get response score
        plan_id = treatment_plan.get("plan_id", "")
        patient_plan_responses = drug_response_df[
            (drug_response_df['PATIENT_ID'] == patient_id) & (drug_response_df['PLAN_ID'] == plan_id)
        ]
        if not patient_plan_responses.empty:
            recist = patient_plan_responses.iloc[0]['RECIST_RESPONSE']
            response_score = RECIST_SCORE_MAP.get(recist, -1.0)
        else:
            response_score = get_drug_response_score(patient_id, "+".join(drugs), drug_response_df)

        if response_score is None or response_score < 0:
            return None

        # ---- Build the prompt ----
        prompt = f"Patient ID: {patient_id}\n"
        prompt += f"Drug(s): {', '.join(drugs)}\n\n"

        # Treatment Plan Information
        prompt += "# Treatment Plan Information\n"
        prompt += f"Plan ID: {treatment_plan.get('plan_id', 'Unknown')}\n"
        prompt += f"Regimen Type: {treatment_plan.get('regimen_type', 'Unknown')}\n"
        if treatment_plan.get('notes'):
            prompt += f"Notes: {treatment_plan['notes']}\n"
        prompt += "\n"

        # Drug Information
        prompt += "# Drug Information\n"
        for drug in drugs:
            if drug in DRUGS_TO_SKIP_INFO:
                continue
            display_drug = DRUG_SUBSTITUTIONS.get(drug, drug)
            drug_info = get_drug_info(drug, drugbank_data)
            prompt += f"## {display_drug}\n"
            prompt += f"Description: {drug_info['description']}\n"
            prompt += f"Mechanism of Action: {drug_info['mechanism_of_action']}\n\n"

        # Clinical and Diagnosis Information
        prompt += "# Clinical and Diagnosis Information\n"
        prompt += f"Gender: {patient_info.get('GENDER', 'Unknown')}\n"
        prompt += f"Age: {patient_info.get('CURRENT_AGE_DEID', 'Unknown')}\n"
        prompt += f"Race: {patient_info.get('RACE', 'Unknown')}\n"
        prompt += f"Ethnicity: {patient_info.get('ETHNICITY', 'Unknown')}\n"
        prompt += f"Smoking History: {patient_info.get('SMOKING_PREDICTIONS_3_CLASSES', 'Unknown')}\n"

        # Stage from closest diagnosis (avoids temporal leakage)
        current_stage = "Unknown"
        if closest_diagnosis is not None:
            if 'STAGE_CDM_DERIVED' in closest_diagnosis and not pd.isna(closest_diagnosis['STAGE_CDM_DERIVED']):
                current_stage = closest_diagnosis['STAGE_CDM_DERIVED']
            elif 'CLINICAL_GROUP' in closest_diagnosis and not pd.isna(closest_diagnosis['CLINICAL_GROUP']):
                current_stage = f"Stage {closest_diagnosis['CLINICAL_GROUP']}"
            elif 'PATH_GROUP' in closest_diagnosis and not pd.isna(closest_diagnosis['PATH_GROUP']):
                current_stage = f"Stage {closest_diagnosis['PATH_GROUP']}"
        prompt += f"Stage (at diagnosis): {current_stage}\n"

        # Status information
        if pdl1_results:
            prompt += "\n## PD-L1 Status\n"
            for pdl1 in pdl1_results:
                prompt += f"PD-L1 Positive: {pdl1['positive']} ({pdl1['timing_desc']})\n"
        if mmr_results:
            prompt += "\n## MMR Status\n"
            for mmr in mmr_results:
                prompt += f"MMR Absent: {mmr['mmr_absent']} ({mmr['timing_desc']})\n"

        # Lab results
        all_lab_results = cea_results + ca_15_3_results + ca_19_9_results
        if all_lab_results:
            prompt += "\n## Laboratory Results\n"
            for lab in all_lab_results:
                if lab['result'] is not None:
                    prompt += f"{lab['test']}: {lab['result']} {lab['unit']} ({lab['timing_desc']})\n"

        # Prior medications
        prompt += f"\nPrior Medication Status: {prior_to_msk}\n"
        if prior_msk_drugs:
            prompt += f"Prior Treatments: {', '.join(prior_msk_drugs)}\n"

        # Cancer-specific features
        if gleason_score is not None:
            prompt += f"Gleason Score: {gleason_score}\n"
        if hr_status != "Unknown" or her2_status != "Unknown":
            prompt += f"HR Status: {hr_status}\n"
            prompt += f"HER2 Status: {her2_status}\n"

        # Diagnosis section
        if closest_diagnosis is not None:
            prompt += "\n# Diagnosis Information\n"
            prompt += f"Diagnosis Description: {closest_diagnosis.get('DX_DESCRIPTION', 'Unknown')}\n"
            if 'AJCC' in closest_diagnosis and not pd.isna(closest_diagnosis['AJCC']):
                prompt += f"AJCC Stage: {closest_diagnosis['AJCC']}\n"
            if 'CLINICAL_GROUP' in closest_diagnosis and not pd.isna(closest_diagnosis['CLINICAL_GROUP']):
                prompt += f"Clinical Group: {closest_diagnosis['CLINICAL_GROUP']}\n"
            if 'PATH_GROUP' in closest_diagnosis and not pd.isna(closest_diagnosis['PATH_GROUP']):
                prompt += f"Pathological Group: {closest_diagnosis['PATH_GROUP']}\n"
            if 'STAGE_CDM_DERIVED' in closest_diagnosis and not pd.isna(closest_diagnosis['STAGE_CDM_DERIVED']):
                prompt += f"Derived Stage: {closest_diagnosis['STAGE_CDM_DERIVED']}\n"
            if 'SUMMARY' in closest_diagnosis and not pd.isna(closest_diagnosis['SUMMARY']):
                prompt += f"Summary: {closest_diagnosis['SUMMARY']}\n"

        # Sample Information
        if sample_info_with_timing:
            prompt += "\n# Sample Information\n"
            for si in sample_info_with_timing:
                sd = si['sample_data']
                prompt += f"\n## Sample {si['sample_id']} ({si['timing_desc']})\n"
                prompt += f"Cancer Type: {sd.get('CANCER_TYPE', 'Unknown')}\n"
                prompt += f"Cancer Type Detailed: {sd.get('CANCER_TYPE_DETAILED', 'Unknown')}\n"
                prompt += f"Primary Site: {sd.get('PRIMARY_SITE', 'Unknown')}\n"
                if not pd.isna(sd.get('METASTATIC_SITE', 'Unknown')) and sd.get('METASTATIC_SITE') != 'Not Applicable':
                    prompt += f"Metastatic Site: {sd.get('METASTATIC_SITE', 'Unknown')}\n"
                prompt += f"Sample Type: {sd.get('SAMPLE_TYPE', 'Unknown')}\n"
                if 'MSI_COMMENT' in sd and not pd.isna(sd['MSI_COMMENT']):
                    prompt += f"MSI Comment: {sd['MSI_COMMENT']}\n"
                if 'MSI_SCORE' in sd and not pd.isna(sd['MSI_SCORE']):
                    prompt += f"MSI Score: {sd['MSI_SCORE']}\n"
                if 'MSI_TYPE' in sd and not pd.isna(sd['MSI_TYPE']):
                    prompt += f"MSI Type: {sd['MSI_TYPE']}\n"
                if 'TMB_NONSYNONYMOUS' in sd and not pd.isna(sd['TMB_NONSYNONYMOUS']):
                    prompt += f"TMB (nonsynonymous): {float(sd['TMB_NONSYNONYMOUS']):.2f}\n"

        # Genetic Profile
        prompt += "\n# Genetic Profile\n"
        if mutations_with_timing:
            prompt += "## Mutations\n"
            for mi in mutations_with_timing:
                prompt += f"Sample {mi['sample_id']} ({mi['timing_desc']}):\n"
                prompt += f"{', '.join(mi['mutations'])}\n\n"
        else:
            prompt += "## Mutations\nNo mutations detected\n\n"

        if svs_with_timing:
            prompt += "## Structural Variants\n"
            for sv in svs_with_timing:
                prompt += f"Sample {sv['sample_id']} ({sv['timing_desc']}):\n"
                prompt += f"{', '.join(sv['structural_variants'])}\n\n"
        else:
            prompt += "## Structural Variants\nNo structural variants detected\n\n"

        # Prediction Task
        prompt += "# Prediction Task\n"
        prompt += "Based on the patient's genetic profile (mutations and structural variants), clinical information, and the drug's mechanism of action, please provide:\n\n"
        prompt += "1. Output a score representing the likelihood of positive treatment response (a number between 0 and 1).\n"
        prompt += " - Scores closer to 1 indicate higher likelihood of complete response(positive)\n"
        prompt += " - Scores closer to 0 indicate higher likelihood of disease progression(negative)\n"
        prompt += "2. Key genetic factors influencing this prediction and why they matter for this specific drug response\n"
        prompt += "3. Any other clinical factors that might affect response to this treatment\n"

        # Standardize drug names for output
        standardized_drugs = [DRUG_SUBSTITUTIONS.get(d, d) for d in drugs]

        return {
            "patient_id": patient_id,
            "plan_id": treatment_plan.get("plan_id", "unknown"),
            "drug_name": ", ".join(standardized_drugs),
            "prompt": prompt,
            "score": f"{response_score:.1f}",
        }

    except Exception as e:
        print(f"Error generating prompt for patient {patient_id}: {e}")
        return None


def generate_prompts_from_treatment_regimens(data_dict, treatment_regimens, limit=None):
    """Generate prompts from pre-divided treatment regimens using parallel processing."""
    process_args = []
    for patient_id, regimen in treatment_regimens.items():
        if 'treatment_plans' in regimen:
            for plan in regimen['treatment_plans']:
                if plan.get('drugs', []):
                    process_args.append((patient_id, plan, data_dict))

    if limit and len(process_args) > limit:
        print(f"Limiting to {limit} prompts (from {len(process_args)} total)")
        import random
        random.seed(42)
        process_args = random.sample(process_args, limit)

    chunk_size = max(1, len(process_args) // (os.cpu_count() * 10))

    prompts = []
    with tqdm(total=len(process_args), desc="Generating prompts") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for i in range(0, len(process_args), chunk_size):
                batch = process_args[i:i+chunk_size]
                futures = [executor.submit(generate_prompt_from_treatment_plan, arg) for arg in batch]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            prompts.append(result)
                    except Exception as e:
                        print(f"Error in prompt generation: {e}")
                    pbar.update(1)

    return prompts


def main(debug=False):
    data_dict = load_data()
    treatment_regimens = load_treatment_regimens()

    limit = DEBUG_LIMIT if debug else None
    prompts = generate_prompts_from_treatment_regimens(data_dict, treatment_regimens, limit=limit)

    output_file = RAW_PROMPTS
    if debug:
        output_file = RAW_PROMPTS.replace('.jsonl', '_debug.jsonl')

    print(f"Writing {len(prompts)} prompts to {output_file}")
    from utils import write_jsonl
    write_jsonl(prompts, output_file)

    if prompts:
        sample_file = os.path.join(BASE_DIR, "sample_treatment_plan_prompt_2026.txt")
        with open(sample_file, "w") as f:
            f.write(prompts[0]["prompt"])
        print(f"Sample prompt saved to {sample_file}")

    print(f"Successfully generated {len(prompts)} prompts")


if __name__ == "__main__":
    import argparse
    os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() // 2))
    parser = argparse.ArgumentParser(description='Generate drug response prompts from treatment plans')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with limited prompts')
    args = parser.parse_args()
    main(debug=args.debug)
