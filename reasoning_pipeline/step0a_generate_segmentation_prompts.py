"""
Step 0a: Generate patient prompts for LLM-based treatment regimen segmentation.

Reads raw MSK diagnosis + treatment timeline data, creates structured prompts
that instruct an LLM to parse continuous medication histories into discrete
treatment plans with temporal boundaries.
"""
import pandas as pd
import os
from tqdm import tqdm
from config import DIAGNOSIS_FILE, TREATMENT_FILE, SEGMENTATION_PROMPTS_DIR

PROMPT_TEMPLATE = """# Cancer Treatment Plan Analysis Task

## Task Description
Analyze the given patient's diagnosis and medication records to organize them into one or more coherent treatment plans, and output in a structured format.

## Input Data

### Patient Diagnosis Record
```
{diagnosis_data}
```

### Patient Medication Records (chronologically ordered)
```
{treatment_data}
```

## Segmentation Rules
1. Group medication records based on temporal continuity and clinical significance
2. Medications used within similar timeframes (intervals not exceeding 30 days) are considered part of the same treatment plan
3. Consider medication type combinations (chemotherapy, immunotherapy, hormone therapy, etc.)
4. Consider the relationship between treatment and diagnosis (e.g., first-line treatment after initial diagnosis, salvage treatment after recurrence)

## Output Format Requirements
Please provide analysis results in JSON format, including the following:

```json
{{
  "patient_id": "PATIENT_ID",
  "diagnosis": {{
    "description": "Diagnosis description",
    "date": "Diagnosis date",
    "stage": "Stage",
    "summary": "Summary"
  }},
  "treatment_plans": [
    {{
      "plan_id": 1,
      "start_date": "Plan start date",
      "end_date": "Plan end date",
      "regimen_type": "Plan type (e.g., Chemotherapy, Combined Therapy)",
      "drugs": [
        {{
          "agent": "Drug name",
          "subtype": "Drug type",
          "start_date": "Start date",
          "stop_date": "End date"
        }}
      ],
      "notes": "Additional notes about the treatment plan"
    }}
  ],
  "summary": "Brief summary of the patient's overall treatment history"
}}
```

## Special Notes
1. Dates are represented as relative days, which may be negative (before diagnosis date) or positive (after diagnosis date)
2. Reasonably divide treatment plans based on time intervals, drug types, and clinical significance
3. In uncertain situations, make reasonable inferences based on clinical experience and common patterns of drug combinations
"""


def generate_treatment_plan_prompts(diagnosis_file, treatment_file, output_dir):
    """Generate one prompt file per patient for LLM treatment segmentation."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data files...")
    diagnosis_df = pd.read_csv(diagnosis_file, sep='\t')
    treatment_df = pd.read_csv(treatment_file, sep='\t')

    patient_ids = treatment_df['PATIENT_ID'].unique()
    print(f"Found {len(patient_ids)} unique patients")

    for patient_id in tqdm(patient_ids, desc="Generating prompts"):
        try:
            patient_diagnosis = diagnosis_df[diagnosis_df['PATIENT_ID'] == patient_id]
            patient_treatment = treatment_df[treatment_df['PATIENT_ID'] == patient_id]

            # Format diagnosis data
            diag_header = "PATIENT_ID | START_DATE | STOP_DATE | EVENT_TYPE | SUBTYPE | SOURCE | DX_DESCRIPTION | AJCC | CLINICAL_GROUP | PATH_GROUP | STAGE_CDM_DERIVED | SUMMARY"
            diagnosis_rows = [diag_header]
            if patient_diagnosis.empty:
                diagnosis_rows.append(
                    f"{patient_id} | N/A | N/A | Diagnosis | Primary | N/A | NO_DIAGNOSIS_AVAILABLE | N/A | N/A | N/A | N/A | N/A"
                )
            else:
                for _, row in patient_diagnosis.iterrows():
                    diagnosis_rows.append(" | ".join([
                        str(row.get('PATIENT_ID', '')),
                        str(row.get('START_DATE', '')),
                        str(row.get('STOP_DATE', '') if 'STOP_DATE' in row else ''),
                        str(row.get('EVENT_TYPE', '')),
                        str(row.get('SUBTYPE', '')),
                        str(row.get('SOURCE', '') if 'SOURCE' in row else ''),
                        str(row.get('DX_DESCRIPTION', '')),
                        str(row.get('AJCC', '') if 'AJCC' in row else ''),
                        str(row.get('CLINICAL_GROUP', '') if 'CLINICAL_GROUP' in row else ''),
                        str(row.get('PATH_GROUP', '') if 'PATH_GROUP' in row else ''),
                        str(row.get('STAGE_CDM_DERIVED', '')),
                        str(row.get('SUMMARY', '')),
                    ]))

            # Format treatment data
            treat_header = "PATIENT_ID | START_DATE | STOP_DATE | EVENT_TYPE | SUBTYPE | AGENT | RX_INVESTIGATIVE | FLAG_OROTOPICAL"
            treatment_rows = [treat_header]
            for _, row in patient_treatment.iterrows():
                treatment_rows.append(" | ".join([
                    str(row.get('PATIENT_ID', '')),
                    str(row.get('START_DATE', '')),
                    str(row.get('STOP_DATE', '')),
                    str(row.get('EVENT_TYPE', '')),
                    str(row.get('SUBTYPE', '')),
                    str(row.get('AGENT', '')),
                    str(row.get('RX_INVESTIGATIVE', '')),
                    str(row.get('FLAG_OROTOPICAL', '')),
                ]))

            patient_prompt = PROMPT_TEMPLATE.format(
                diagnosis_data="\n".join(diagnosis_rows),
                treatment_data="\n".join(treatment_rows),
            )

            output_file = os.path.join(output_dir, f"{patient_id}_prompt.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(patient_prompt)

        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")

    print(f"Done! Generated prompts for {len(patient_ids)} patients in {output_dir}")


def main():
    print("Step 0a: Generating treatment segmentation prompts...")
    print(f"Input:  {DIAGNOSIS_FILE}")
    print(f"        {TREATMENT_FILE}")
    print(f"Output: {SEGMENTATION_PROMPTS_DIR}/")
    generate_treatment_plan_prompts(DIAGNOSIS_FILE, TREATMENT_FILE, SEGMENTATION_PROMPTS_DIR)


if __name__ == "__main__":
    main()
