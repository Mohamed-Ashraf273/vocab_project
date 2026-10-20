#!/usr/bin/env python3
"""
Find duplicate medication vocabulary entries and export them to JSON.

Default input:
    medication_vocabulary_merged.json

Default output:
    medication_vocabulary_duplicates.json

This script checks exact duplicate names after conservative text normalization:
- Unicode NFKC
- case folding
- trimming/collapsing whitespace
- punctuation normalization
- Arabic diacritic/tatweel removal
- Arabic digit normalization

It does NOT automatically merge or delete entries. It reports duplicate-name groups
and keeps the full original entries in each group for manual inspection.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TATWEEL = "\u0640"
PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
WHITESPACE = re.compile(r"\s+")

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def normalize_name(value: Any) -> str:
    """Normalize a name for duplicate detection, without changing the source data."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.translate(ARABIC_DIGITS).translate(PERSIAN_DIGITS)
    text = text.replace(TATWEEL, "")
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.casefold()
    text = PUNCTUATION.sub(" ", text)
    text = WHITESPACE.sub(" ", text).strip()
    return text


def load_vocabulary(path: Path) -> tuple[list[dict], dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Supports a bare list, or a common object containing a vocabulary list.
    if isinstance(data, list):
        entries = data
        wrapper = {"kind": "list"}
    elif isinstance(data, dict):
        wrapper = {"kind": "dict", "original": data}
        entries = None
        for key in ("vocabulary", "entries", "medications", "items"):
            if isinstance(data.get(key), list):
                entries = data[key]
                wrapper["list_key"] = key
                break

        if entries is None:
            # Some generated vocabularies are keyed by a vocabulary/normalized key.
            # Convert the mapping to records while retaining the mapping key.
            if data and all(isinstance(v, dict) for v in data.values()):
                entries = []
                for key, value in data.items():
                    item = dict(value)
                    item.setdefault("_vocabulary_mapping_key", key)
                    entries.append(item)
                wrapper["kind"] = "mapping"
            else:
                raise ValueError(
                    "Could not locate the vocabulary list. Expected a JSON list or "
                    "an object containing one of: vocabulary, entries, medications, items."
                )
    else:
        raise ValueError("Unsupported JSON root. Expected a list or object.")

    normalized_entries = []
    for i, entry in enumerate(entries):
        if isinstance(entry, dict):
            normalized_entries.append(entry)
        else:
            normalized_entries.append({"_entry_value": entry, "_entry_index": i})
    return normalized_entries, wrapper


def find_name(entry: dict) -> tuple[str | None, str | None]:
    """Return (field name, value) for a likely medication-name field."""
    candidates = (
        "canonical_name",
        "medication_name",
        "name",
        "prescription_canonical_name",
        "display_name",
        "tree_med_canonical_name",
        "original_name",
        "raw_name",
        "Name",
        "ArabicName",
        "ScientificName",
    )
    for key in candidates:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return key, value

    # Fallback: use the entry's normalized key only if no display-name field exists.
    for key in ("normalized_key", "tree_med_base_key", "key"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return key, value
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find duplicate medication names in a vocabulary JSON."
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        default="medication_vocabulary_merged.json",
        help="Vocabulary JSON (default: medication_vocabulary_merged.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="medication_vocabulary_duplicates.json",
        help="Output JSON (default: medication_vocabulary_duplicates.json)",
    )
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Only detect exact duplicate strings (ignores normalization equivalence).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output)
    entries, _wrapper = load_vocabulary(input_path)

    groups: dict[str, list[dict]] = defaultdict(list)
    no_name_count = 0

    for index, entry in enumerate(entries):
        field, raw_name = find_name(entry)
        if not raw_name:
            no_name_count += 1
            continue
        key = raw_name.strip() if args.exact_only else normalize_name(raw_name)
        if not key:
            no_name_count += 1
            continue
        groups[key].append({
            "entry_index": index,
            "name_field": field,
            "name": raw_name,
            "normalized_name": normalize_name(raw_name),
            "entry": entry,
        })

    duplicate_groups = []
    duplicate_entry_count = 0
    for key, group in groups.items():
        if len(group) < 2:
            continue
        duplicate_entry_count += len(group)
        duplicate_groups.append({
            "duplicate_key": key,
            "entry_count": len(group),
            "entries": group,
        })

    duplicate_groups.sort(key=lambda x: (-x["entry_count"], x["duplicate_key"]))

    output = {
        "metadata": {
            "input_file": str(input_path),
            "duplicate_definition": (
                "Exact name strings after trimming whitespace."
                if args.exact_only
                else "Names that become equal after Unicode, case, whitespace, punctuation, "
                     "Arabic diacritic/tatweel, and digit normalization."
            ),
            "normalization_mode": "exact_only" if args.exact_only else "normalized",
            "input_entry_count": len(entries),
            "entries_with_no_detectable_name": no_name_count,
            "duplicate_group_count": len(duplicate_groups),
            "entries_in_duplicate_groups": duplicate_entry_count,
            "extra_duplicate_entries": sum(g["entry_count"] - 1 for g in duplicate_groups),
        },
        "duplicate_groups": duplicate_groups,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("Duplicate scan complete.")
    print(f"Vocabulary entries scanned:       {len(entries)}")
    print(f"Duplicate name groups:            {len(duplicate_groups)}")
    print(f"Entries in duplicate groups:      {duplicate_entry_count}")
    print(f"Extra duplicate entries:          {output['metadata']['extra_duplicate_entries']}")
    print(f"Entries without detectable name:  {no_name_count}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
