import json
import re
import unicodedata
from pathlib import Path
from typing import Any

GEMINI_JSON = Path("gemini_gt.json")
GEMMA_JSON = Path("gemma_pre_gt_bg.json")
OUTPUT_JSON = Path("matching_medications.json")

MAX_EDITS = 1000


def iter_entries(data: Any):
    """Find sample dictionaries recursively."""
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if "filename" in item or "file_name" in item:
                    yield item
                else:
                    yield from iter_entries(item)
    elif isinstance(data, dict):
        if "filename" in data or "file_name" in data:
            yield data
        else:
            for value in data.values():
                if isinstance(value, (list, dict)):
                    yield from iter_entries(value)


def get_filename(entry: dict) -> str:
    return str(entry.get("filename") or entry.get("file_name") or "").strip()


def normalize(text: Any) -> str:
    """Normalize medication text for comparison."""
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def edit_distance(a: str, b: str) -> int:
    """Calculate Levenshtein edit distance."""
    a, b = normalize(a), normalize(b)

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,          # insertion
                    previous[j] + 1,          # deletion
                    previous[j - 1] + (char_a != char_b),  # substitution
                )
            )

        previous = current

    return previous[-1]


def get_medications(entry: dict) -> list:
    """Support common medication field names."""
    for key in ("medications", "medication", "medication_names", "meds"):
        value = entry.get(key)

        if isinstance(value, list):
            return value

        if isinstance(value, dict):
            return [value]

    return []


def get_medication_name(medication: Any) -> str:
    """Extract the medication name while retaining the full original object."""
    if isinstance(medication, str):
        return medication.strip()

    if isinstance(medication, dict):
        for key in ("name", "medication_name", "medication", "text"):
            if medication.get(key):
                return str(medication[key]).strip()

    return ""


def main() -> None:
    for path in (GEMINI_JSON, GEMMA_JSON):
        if not path.is_file():
            raise FileNotFoundError(f"JSON file not found: {path.resolve()}")

    with GEMINI_JSON.open("r", encoding="utf-8") as f:
        gemini_data = json.load(f)

    with GEMMA_JSON.open("r", encoding="utf-8") as f:
        gemma_data = json.load(f)

    gemini_by_filename = {
        get_filename(entry): entry
        for entry in iter_entries(gemini_data)
        if get_filename(entry)
    }

    gemma_by_filename = {
        get_filename(entry): entry
        for entry in iter_entries(gemma_data)
        if get_filename(entry)
    }

    results = []
    compared = 0
    matched = 0

    for filename, gemini_entry in gemini_by_filename.items():
        gemma_entry = gemma_by_filename.get(filename)

        if gemma_entry is None:
            continue

        gemini_meds = get_medications(gemini_entry)
        gemma_meds = get_medications(gemma_entry)

        # Compare medications in order, pairing each Gemini medication
        # with the closest unused Gemma medication.
        unused_gemma = list(enumerate(gemma_meds))
        matched_pairs = []

        for gemini_med in gemini_meds:
            gemini_name = get_medication_name(gemini_med)

            if not gemini_name or not unused_gemma:
                continue

            candidates = [
                (
                    edit_distance(
                        gemini_name,
                        get_medication_name(gemma_med),
                    ),
                    index,
                    gemma_med,
                )
                for index, gemma_med in unused_gemma
                if get_medication_name(gemma_med)
            ]

            if not candidates:
                continue

            distance, index, gemma_med = min(
                candidates, key=lambda item: item[0]
            )
            compared += 1

            if distance <= MAX_EDITS:
                matched_pairs.append({
                    "gemini": gemini_med,
                    "gemma": gemma_med,
                    "edit_distance": distance,
                })
                unused_gemma = [
                    item for item in unused_gemma if item[0] != index
                ]
                matched += 1

        if matched_pairs:
            results.append({
                "filename": filename,
                "medications": matched_pairs,
            })

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Gemini samples: {len(gemini_by_filename)}")
    print(f"Gemma samples: {len(gemma_by_filename)}")
    print(f"Medication pairs compared: {compared}")
    print(f"Medication pairs within {MAX_EDITS} edits: {matched}")
    print(f"Samples with matching medications: {len(results)}")
    print(f"Output: {OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    main()