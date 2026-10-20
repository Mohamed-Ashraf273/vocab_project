import json
import re
import unicodedata
from pathlib import Path
from typing import Any

INPUT_JSON = Path("medication_vocab.json")
OUTPUT_JSON = Path("medication_vocab_normalized_arabic.json")

ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)
NON_ALNUM = re.compile(r"[^\w\u0600-\u06FF]+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def normalize(text: Any) -> str:
    s = str(text or "").casefold().translate(
        str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    )
    s = unicodedata.normalize("NFKC", s)
    s = ARABIC_DIACRITICS.sub("", s)
    s = (
        s.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
    )
    s = NON_ALNUM.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip()


with INPUT_JSON.open("r", encoding="utf-8") as f:
    data = json.load(f)

medications = data["medications"] if isinstance(data, dict) else data

medications_with_arabic = 0
total_duplicates_removed = 0
total_names_normalized = 0

for med in medications:
    arabic_names = med.get("arabic_names")

    # Skip medications without Arabic names
    if not arabic_names:
        continue

    # Support both a list of names and a single string
    if isinstance(arabic_names, str):
        arabic_names = [arabic_names]

    if not isinstance(arabic_names, list):
        continue

    medications_with_arabic += 1

    # Normalize names and discard empty results
    normalized_names = [
        normalize(name)
        for name in arabic_names
    ]
    normalized_names = [
        name for name in normalized_names if name
    ]

    # Remove duplicates while preserving the original order
    unique_names = list(dict.fromkeys(normalized_names))

    total_names_normalized += len(normalized_names)
    total_duplicates_removed += len(normalized_names) - len(unique_names)

    # Update the medication
    med["arabic_names"] = unique_names

# Save to a separate file, preserving other JSON fields
with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Total medications: {len(medications)}")
print(f"Medications with Arabic names: {medications_with_arabic}")
print(f"Arabic names processed: {total_names_normalized}")
print(f"Duplicate names removed: {total_duplicates_removed}")
print(f"Saved to: {OUTPUT_JSON.resolve()}")