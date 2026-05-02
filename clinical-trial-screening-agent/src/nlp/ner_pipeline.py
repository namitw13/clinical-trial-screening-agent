import spacy
import re
import sys
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).parent))
from data_loader import load_and_clean

print("Loading NER model...")
nlp = spacy.load("en_core_web_sm")
print("✅ Model loaded.")

# Rule-based medical entity dictionaries
DISEASES = [
    "diabetes mellitus", "hypertension", "chronic kidney disease",
    "asthma", "rheumatoid arthritis", "lung cancer", "subarachnoid hemorrhage",
    "septic arthritis", "adenocarcinoma", "pulmonary nodule",
    "chest pain", "shortness of breath", "fatigue", "weight loss",
    "night sweats", "headache"
]

MEDICATIONS = [
    "metformin", "lisinopril", "atorvastatin", "methotrexate",
    "prednisone", "vancomycin", "albuterol", "ipratropium",
    "methylprednisolone", "fluticasone", "nimodipine"
]

LAB_PATTERN = re.compile(
    r"""(
        HbA1c\s*[\d.]+%?       |
        eGFR\s*[\d.]+          |
        BP\s*[\d/]+            |
        GCS\s*[\d]+            |
        ESR\s*[\d]+            |
        CRP\s*[\d]+            |
        WBC\s*[\d,]+           |
        O2\s*saturation\s*[\d.]+%? |
        peak\s*flow\s*[\d.]+%? |
        \d+\s*pack.year
    )""",
    re.IGNORECASE | re.VERBOSE
)


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract medical entities using rule-based matching.
    Combines dictionary lookup + regex for lab values.
    """
    text_lower = text.lower()

    # Disease extraction
    diseases_found = [d for d in DISEASES if d.lower() in text_lower]

    # Medication extraction
    meds_found = [m for m in MEDICATIONS if m.lower() in text_lower]

    # Lab value extraction via regex
    labs_found = [match.group().strip() for match in LAB_PATTERN.finditer(text)]

    # Also run spaCy for any additional named entities
    doc = nlp(text)
    spacy_entities = [{
        "text": ent.text.strip(),
        "label": ent.label_
    } for ent in doc.ents]

    entities = {
        "diseases": list(set(diseases_found)),
        "medications": list(set(meds_found)),
        "lab_values": list(set(labs_found)),
        "spacy_entities": spacy_entities
    }

    return entities


def process_all_notes(df) -> List[Dict]:
    """Run entity extraction on all notes."""
    results = []
    for _, row in df.iterrows():
        entities = extract_entities(row["text"])
        total = len(entities["diseases"]) + len(entities["medications"]) + len(entities["lab_values"])
        results.append({
            "note_id": row["note_id"],
            "patient_id": row["patient_id"],
            "text": row["text"],
            "entities": entities
        })
        print(f"✅ Processed {row['note_id']} — found {total} medical entities")
    return results


if __name__ == "__main__":
    df = load_and_clean()
    results = process_all_notes(df)

    print("\n--- Entity Extraction Sample (NOTE_001) ---")
    sample = results[0]["entities"]
    print(f"Diseases/Conditions : {sample['diseases']}")
    print(f"Medications         : {sample['medications']}")
    print(f"Lab Values          : {sample['lab_values']}")

    print("\n--- Entity Extraction Sample (NOTE_002) ---")
    sample2 = results[1]["entities"]
    print(f"Diseases/Conditions : {sample2['diseases']}")
    print(f"Medications         : {sample2['medications']}")
    print(f"Lab Values          : {sample2['lab_values']}")