"""
Shared utilities for the OncoRAFT reasoning pipeline.

Consolidates duplicated code from multiple steps:
- RateLimiter (was in step0b, step5)
- extract_days_from_string (was in step0c, step1)
- load_jsonl / write_jsonl (was in step4, step5)
- Drug constants (was in step1)
"""
import re
import json
import time
import threading
import io
import pandas as pd


# ============================================================
# Rate Limiter (thread-safe)
# ============================================================
class RateLimiter:
    """Fixed-interval rate limiter for concurrent API calls."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.last_time = time.time() - min_interval

    def acquire(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_time = time.time()


# ============================================================
# Date parsing
# ============================================================
def extract_days_from_string(date_string):
    """Extract numeric days value from various string date formats.

    Handles: int, float, "70 days", "Day -111", "Relative day X",
             "X days after/before", "X (relative days)".
    Returns 0 if no pattern matches.
    """
    if date_string is None:
        return 0
    if isinstance(date_string, (int, float)):
        return int(date_string)
    try:
        return int(date_string)
    except (ValueError, TypeError):
        pass

    if not isinstance(date_string, str):
        return 0

    patterns = [
        (r'(-?\d+)\s*days?', None),                          # "70 days"
        (r'Day\s+(-?\d+)', None),                             # "Day -111"
        (r'Relative\s+day\s+(-?\d+)', None),                  # "Relative day X"
        (r'(-?\d+)\s*days?\s*(after|post|before)', 'dir'),    # "X days after"
        (r'(-?\d+)\s*\(', None),                              # "X (relative days)"
    ]

    for pattern, mode in patterns:
        match = re.search(pattern, date_string, re.IGNORECASE)
        if match:
            days = int(match.group(1))
            if mode == 'dir' and match.group(2).lower() == 'before':
                days = -days
            return days

    return 0


# ============================================================
# JSONL I/O
# ============================================================
def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(data, path):
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================
# TSV reader
# ============================================================
def read_tsv_file(file_path, comment_char='#'):
    """Read a TSV file, skipping comment lines.
    Uses manual comment-line skipping to avoid pandas comment='#' breaking
    columns that contain '#' in values (e.g., STYLE_COLOR='#fae9e9').
    """
    try:
        # Skip comment lines manually, don't pass comment= to pandas
        lines = []
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith(comment_char):
                    lines.append(line)
        if lines:
            return pd.read_csv(
                io.StringIO(''.join(lines)),
                sep='\t', low_memory=False
            )
        return pd.read_csv(file_path, sep='\t', low_memory=False)
    except Exception:
        lines = []
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith(comment_char):
                    lines.append(line)
        header_line = 0
        for i, line in enumerate(lines):
            if line.strip():
                header_line = i
                break
        return pd.read_csv(
            io.StringIO(''.join(lines[header_line:])),
            sep='\t', engine='python', low_memory=False
        )


# ============================================================
# Drug constants
# ============================================================
# Drug name normalization: MSK raw name -> DrugBank-compatible name
DRUG_SUBSTITUTIONS = {
    "PACLITAXEL PROTEIN-BOUND": "PACLITAXEL",
    "FAM-TRASTUZUMAB DERUXTECAN": "Trastuzumab deruxtecan",
    "TIPIRACIL-TRIFLURIDINE": "Trifluridine + Tipiracil",
    "DOXORUBICIN LIPOSOMAL": "DOXORUBICIN",
    "ADO-TRASTUZUMAB EMTANSINE": "TRASTUZUMAB EMTANSINE",
    "YTTRIUM Y-90 THERASPHERES": "Yttrium Y-90",
    "YTTRIUM Y-90 MICROSPHERES": "Yttrium Y-90",
    "YTTRIUM Y-90 CLIVATUZUMAB": "Yttrium Y-90",
    "LUTETIUM LU-177 DOTATATE": "Lutetium Lu-177 Dotatate",
    "LUTETIUM LU-177 DOTAEBTATE": "Lutetium Lu-177 Dotatate",
    "LUTETIUM LU-177 DOTA-JR11": "Lutetium Lu-177 DOTA-JR11",
    "LUTETIUM LU-177 5B1-MVT1075": "Lutetium Lu-177",
    "FLUOROURACIL TOPICAL": "FLUOROURACIL",
    "SODIUM IODIDE I-131": "Iodide I-131",
    "SODIUM IODIDE I-131 MIBG": "Iodide I-131",
    "SODIUM IODIDE I-131 MIP-1095": "Iodide I-131",
    "SODIUM IODIDE I-131 OMBURTAMAB 8H9": "Iodide I-131",
    "SODIUM IODIDE I-131 BC8 ANTICD45": "Iodide I-131",
    "AFLIBERCEPT OPHTHALMIC": "Aflibercept",
    "RADIUM RA-223 DICHLORIDE": "Radium Ra 223 dichloride",
    "LETROZOLE-RIBOCICLIB": "Ribociclib + letrozole",
    "IRINOTECAN LIPOSOMAL": "IRINOTECAN",
    "CYTARABINE LIPOSOMAL-DAUNORUBICIN LIPOSOMAL": "daunorubicin + cytarabine",
    "MITOMYCIN OPHTHALMIC": "MITOMYCIN",
    "THIOGUANINE": "Tioguanine",
    "CEDAZURIDINE-DECITABINE": "DECITABINE",
    "SAMARIUM SM-153 LEXIDRONAM": "Samarium Sm 153",
    "ABT-888 (VELIPARIB)": "VELIPARIB",
}

# Manual drug descriptions for radionuclides not in DrugBank
MANUAL_DESCRIPTIONS = {
    "Lutetium Lu-177 Dotatate": {
        "description": "Lutetium Lu-177 Dotatate (Lutathera) is a radiolabeled somatostatin analog used for targeted therapy of somatostatin receptor-positive neuroendocrine tumors (NETs).",
        "mechanism_of_action": "It binds with high affinity to somatostatin receptor subtype 2 (SSTR2) on tumor cells, delivering beta radiation from Lu-177 to induce DNA damage and cell death via receptor-mediated internalization.",
    },
    "Lutetium Lu-177 DOTA-JR11": {
        "description": "Lutetium Lu-177 DOTA-JR11 (OPS201) is an investigational radiolabeled somatostatin receptor antagonist for treating somatostatin receptor-positive tumors, such as neuroendocrine tumors.",
        "mechanism_of_action": "As an antagonist, it binds to multiple somatostatin receptor subtypes (primarily SSTR2 and SSTR5) on tumor cell surfaces, delivering Lu-177 beta radiation to cause DNA damage without receptor internalization.",
    },
    "Samarium Sm 153": {
        "description": "Samarium Sm 153 lexidronam is a radioactive medication used to treat pain caused by cancer that has spread to the bone.",
        "mechanism_of_action": "Samarium Sm 153 lexidronam targets the sites of new bone formation, concentrating in regions of the bone that have been invaded with metastatic tumor, irradiating the osteoblastic tumor sites resulting in relief of pain.",
    },
    "Yttrium Y-90": {
        "description": "Yttrium Y-90 is a radioisotope used in targeted radiation therapy, especially for liver tumors and certain types of lymphoma.",
        "mechanism_of_action": "Yttrium Y-90 emits high-energy beta particles that cause DNA damage and cell death in tumor tissue, with a limited penetration range that helps minimize damage to surrounding healthy tissues.",
    },
    "Iodide I-131": {
        "description": "Iodide I-131 is a radioactive form of iodine used to treat certain thyroid conditions including hyperthyroidism and thyroid cancer.",
        "mechanism_of_action": "Iodide I-131 is taken up by thyroid cells where it emits beta radiation, causing damage to DNA and cell death, particularly affecting thyroid cancer cells that concentrate the isotope.",
    },
}

# Drugs to skip in drug information display (but retain in analysis)
DRUGS_TO_SKIP_INFO = [
    "AC225 H11B6 MED 20-321",
    "CU64 NOTA-PSMAI-PEG-CY55-C' DOTS",
    "INVESTIGATIONAL",
]

# RECIST response -> numeric score mapping
RECIST_SCORE_MAP = {"CR": 1.0, "PR": 0.7, "SD": 0.3, "PD": 0.0}
