"""
Step 0c: Treatment response assessment using adapted RECIST criteria.

Synthesizes tumor site documentation, NLP-extracted radiologic progression records,
and temporal treatment data to classify drug responses as CR/PR/SD/PD.
"""
import pandas as pd
import time
import os
import json
import glob
from collections import defaultdict

from config import (
    BASE_DIR, TREATMENT_FILE, TREATMENT_REGIMENS_DIR,
    TUMOR_SITES_FILE, PROGRESSION_FILE, RECIST_OUTPUT,
)
from utils import extract_days_from_string


def load_data():
    """Load and preprocess all required data."""
    print("Loading data...")
    t0 = time.time()

    df_treatment = pd.read_csv(
        TREATMENT_FILE, sep='\t',
        usecols=['PATIENT_ID', 'START_DATE', 'STOP_DATE', 'AGENT', 'RX_INVESTIGATIVE', 'SUBTYPE'],
    )
    df_tumor_sites = pd.read_csv(
        TUMOR_SITES_FILE, sep='\t',
        usecols=['PATIENT_ID', 'START_DATE', 'TUMOR_SITE'],
        low_memory=False,
    )
    df_progression = pd.read_csv(
        PROGRESSION_FILE, sep='\t',
        usecols=['PATIENT_ID', 'START_DATE', 'PROGRESSION', 'NLP_PROGRESSION_PROBABILITY'],
    )
    print(f"Data loaded in {time.time() - t0:.1f}s")

    # Filter out investigational drugs
    df_treatment = df_treatment[df_treatment['RX_INVESTIGATIVE'] == 'N']

    # Load API treatment plans
    print("Loading API treatment plan responses...")
    t0 = time.time()
    patient_plans = {}
    api_files = glob.glob(os.path.join(TREATMENT_REGIMENS_DIR, "*_api_response.json"))
    print(f"Found {len(api_files)} API response files")

    for fp in api_files:
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
            pid = data.get('patient_id')
            json_data = data.get('extracted_json', data)
            if 'treatment_plans' in json_data:
                patient_plans[pid] = json_data['treatment_plans']
        except Exception:
            pass

    print(f"Parsed {len(patient_plans)} patient plans in {time.time() - t0:.1f}s")

    # Build lookup dicts
    patient_tumor_sites = defaultdict(list)
    for _, row in df_tumor_sites.iterrows():
        patient_tumor_sites[row['PATIENT_ID']].append((row['START_DATE'], row['TUMOR_SITE']))
    for pid in patient_tumor_sites:
        patient_tumor_sites[pid].sort(key=lambda x: x[0])

    patient_progression = defaultdict(list)
    for _, row in df_progression.iterrows():
        patient_progression[row['PATIENT_ID']].append(
            (row['START_DATE'], row['PROGRESSION'], row.get('NLP_PROGRESSION_PROBABILITY', 0.5))
        )
    for pid in patient_progression:
        patient_progression[pid].sort(key=lambda x: x[0])

    patient_treatment_subtypes = defaultdict(set)
    for _, row in df_treatment.iterrows():
        if row['SUBTYPE']:
            patient_treatment_subtypes[row['PATIENT_ID']].add(row['SUBTYPE'])

    return patient_plans, patient_tumor_sites, patient_progression, patient_treatment_subtypes


def determine_adjusted_recist_response(patient_id, plan_start_date, treatment_subtypes,
                                       patient_tumor_sites, patient_progression):
    """
    Classify treatment response using adapted RECIST criteria.

    Treatment-class-specific evaluation windows account for differing
    response kinetics across drug categories.

    Response classification:
      PD: high-confidence progression (>0.95), or >=2 progressions with
          at least one >0.80 or >=28-day interval, or new tumor sites
      CR: complete disappearance of baseline tumor sites with no progression
      PR: <=70% reduction in tumor burden without progression
      SD: default (does not meet PD/CR/PR criteria)
    """
    # Determine evaluation window based on treatment type
    slow_types = {'Hormone', 'Immuno', 'Investigational'}
    if any(sub in slow_types for sub in treatment_subtypes):
        follow_up_end = plan_start_date + 84
    else:
        follow_up_end = plan_start_date + 56

    tumor_records = patient_tumor_sites.get(patient_id, [])
    prog_records = patient_progression.get(patient_id, [])
    if not tumor_records or not prog_records:
        return "Unknown"

    baseline_sites = set()
    during_sites = set()

    for date, site in tumor_records:
        if date < plan_start_date:
            baseline_sites.add(site)
        elif plan_start_date <= date <= follow_up_end:
            during_sites.add(site)

    # --- PD ---
    high_conf = [(d, s, p) for d, s, p in prog_records
                 if plan_start_date <= d <= follow_up_end and s == 'Y' and p > 0.7]

    if any(p > 0.95 for _, _, p in high_conf):
        return "PD"

    if len(high_conf) >= 2:
        has_high = any(p > 0.80 for _, _, p in high_conf)
        sorted_p = sorted(high_conf, key=lambda x: x[0])
        has_interval = any(sorted_p[i][0] - sorted_p[i-1][0] >= 28 for i in range(1, len(sorted_p)))
        if has_high or has_interval:
            return "PD"

    new_sites = during_sites - baseline_sites
    if new_sites:
        return "PD"

    # --- CR ---
    if baseline_sites and not during_sites:
        relevant = [(d, s, p) for d, s, p in prog_records if plan_start_date <= d <= follow_up_end]
        non_indet = [(d, s, p) for d, s, p in relevant if s != 'Indeterminate']
        prog_y = [(d, p) for d, s, p in non_indet if s == 'Y']

        if prog_y:
            return "SD"
        if len(non_indet) < 1:
            return "PR"
        if all(s == 'N' for _, s, _ in non_indet):
            return "CR"
        return "SD"

    # --- PR ---
    if baseline_sites and during_sites:
        if len(during_sites) <= len(baseline_sites) * 0.7:
            window_prog = any(
                s == 'Y' and p > 0.7
                for d, s, p in prog_records
                if plan_start_date <= d <= follow_up_end
            )
            if not window_prog:
                return "PR"

    return "SD"


def analyze_all_plans(patient_plans, patient_tumor_sites, patient_progression,
                      patient_treatment_subtypes):
    """Iterate over all patient treatment plans and classify responses."""
    print("Analyzing all treatment plans...")
    t0 = time.time()
    results = []
    plan_count = 0

    for patient_id, plans in patient_plans.items():
        for plan in plans:
            plan_count += 1
            plan_id = plan.get('plan_id')

            try:
                start = extract_days_from_string(plan.get('start_date'))
                end = extract_days_from_string(plan.get('end_date')) if plan.get('end_date') else None
            except Exception:
                continue

            drugs = plan.get('drugs', [])
            drug_agents = []
            drug_subtypes = set()
            for drug in drugs:
                agent = drug.get('agent')
                if agent and agent != 'INVESTIGATIONAL':
                    drug_agents.append(agent)
                subtype = drug.get('subtype')
                if subtype:
                    drug_subtypes.add(subtype)

            if not drug_subtypes and patient_id in patient_treatment_subtypes:
                drug_subtypes = patient_treatment_subtypes[patient_id]

            drug_combination = "+".join(sorted(drug_agents)) if drug_agents else "Unknown"
            drug_subtypes_str = "+".join(sorted(drug_subtypes)) if drug_subtypes else "Unknown"

            response = determine_adjusted_recist_response(
                patient_id, start, drug_subtypes,
                patient_tumor_sites, patient_progression,
            )

            slow_types = {'Hormone', 'Immuno', 'Investigational'}
            eval_window = 84 if any(s in slow_types for s in drug_subtypes) else 56

            results.append({
                'PATIENT_ID': patient_id,
                'PLAN_ID': plan_id,
                'REGIMEN_TYPE': plan.get('regimen_type', 'Unknown'),
                'DRUG_COMBINATION': drug_combination,
                'DRUG_SUBTYPES': drug_subtypes_str,
                'START_DATE': start,
                'END_DATE': end,
                'EVALUATION_WINDOW': eval_window,
                'RECIST_RESPONSE': response,
            })

            if plan_count % 5000 == 0:
                print(f"  Processed {plan_count} plans...")

    print(f"Done: {len(patient_plans)} patients, {plan_count} plans in {time.time() - t0:.1f}s")
    return pd.DataFrame(results)


def main():
    print("Step 0c: RECIST response assessment...")
    plans, tumor, prog, subtypes = load_data()
    df = analyze_all_plans(plans, tumor, prog, subtypes)

    known = df[~df['RECIST_RESPONSE'].str.startswith('Unknown')]
    print(f"\nTotal regimens: {len(df)}")
    print(f"With valid response: {len(known)}")
    print(f"Excluded (insufficient data): {len(df) - len(known)}")

    df.to_csv(RECIST_OUTPUT, index=False)
    print(f"\nSaved to: {RECIST_OUTPUT}")

    print("\nResponse distribution:")
    counts = df['RECIST_RESPONSE'].value_counts()
    for resp, cnt in counts.items():
        print(f"  {resp}: {cnt} ({cnt/len(df)*100:.1f}%)")

    print("\nEvaluation window distribution:")
    print(df['EVALUATION_WINDOW'].value_counts().to_string())


if __name__ == "__main__":
    main()
