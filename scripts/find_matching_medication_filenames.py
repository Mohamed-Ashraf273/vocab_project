import json
import re
import unicodedata
from pathlib import Path
from typing import Any

GEMINI_FILE = Path("gemini_gt.json")
GEMMA_FILE = Path("gemma_pre_gt_bg.json")
OUTPUT_FILE = Path("matching_filenames.txt")
MAX_EDITS = 2


def normalize_name(value: Any) -> str:
    text = str(value or "").casefold()
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def levenshtein_distance(a: str, b: str, limit: int | None = None) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if limit is not None and abs(len(a) - len(b)) > limit:
        return limit + 1

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            value = min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if limit is not None and row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def iter_entries(data: Any):
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
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


def get_medications(entry: dict) -> list[dict]:
    containers = [entry]
    extracted = entry.get("extracted_data")
    if isinstance(extracted, dict):
        containers.append(extracted)
    for container in containers:
        medications = container.get("medications")
        if isinstance(medications, list):
            return [m for m in medications if isinstance(m, dict)]
    return []


def get_medication_names(entry: dict) -> list[str]:
    names = []
    for medication in get_medications(entry):
        name = normalize_name(medication.get("name"))
        if name:
            names.append(name)
    return names


def main() -> None:
    for path in (GEMINI_FILE, GEMMA_FILE):
        if not path.is_file():
            raise FileNotFoundError(
                f"Could not find {path.resolve()}. Put it beside this script "
                "or update GEMINI_FILE / GEMMA_FILE."
            )

    with GEMINI_FILE.open("r", encoding="utf-8") as f:
        gemini_data = json.load(f)
    with GEMMA_FILE.open("r", encoding="utf-8") as f:
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

    matching_filenames = []
    for filename, gemini_entry in gemini_by_filename.items():
        gemma_entry = gemma_by_filename.get(filename)
        if gemma_entry is None:
            continue

        gemini_names = get_medication_names(gemini_entry)
        gemma_names = get_medication_names(gemma_entry)
        if not gemini_names or not gemma_names:
            continue

        # Include this filename if any Gemma medication name is within
        # two Levenshtein edits of any Gemini medication name for this sample.
        matched = any(
            levenshtein_distance(gemma_name, gemini_name, MAX_EDITS) <= MAX_EDITS
            for gemma_name in gemma_names
            for gemini_name in gemini_names
        )
        if matched:
            matching_filenames.append(filename)

    OUTPUT_FILE.write_text(
        "".join(f"{filename}\n" for filename in matching_filenames),
        encoding="utf-8",
    )
    print(f"Gemini entries: {len(gemini_by_filename)}")
    print(f"Gemma entries: {len(gemma_by_filename)}")
    print(f"Matching filenames: {len(matching_filenames)}")
    print(f"Saved: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
