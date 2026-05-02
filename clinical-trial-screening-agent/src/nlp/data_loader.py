import json
import pandas as pd
from pathlib import Path
from typing import List, Dict


DATA_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "sample_notes.json"


def load_notes(path: Path = DATA_PATH) -> List[Dict]:
    """Load clinical notes from JSON file."""
    with open(path, "r") as f:
        notes = json.load(f)
    print(f"✅ Loaded {len(notes)} clinical notes.")
    return notes


def notes_to_dataframe(notes: List[Dict]) -> pd.DataFrame:
    """Convert notes list to a pandas DataFrame."""
    df = pd.DataFrame(notes)
    df["text_length"] = df["text"].apply(len)
    print(f"✅ DataFrame created with shape: {df.shape}")
    return df


def clean_text(text: str) -> str:
    """Basic text cleaning for clinical notes."""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def load_and_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """Full pipeline: load → clean → return DataFrame."""
    notes = load_notes(path)
    for note in notes:
        note["text"] = clean_text(note["text"])
    df = notes_to_dataframe(notes)
    return df


if __name__ == "__main__":
    df = load_and_clean()
    print("\n--- Sample Note ---")
    print(df.iloc[0]["text"][:300])
    print("\n--- DataFrame Info ---")
    print(df.info())