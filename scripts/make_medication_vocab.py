#!/usr/bin/env python3
"""Build a medication vocabulary from prescription extractions and TreeMed.

Usage:
    python make_medication_vocab.py input.json -o medication_vocabulary.json
    python make_medication_vocab.py input.json --fuzzy-report medication_match_review.json

Only conservative, normalized exact-name matches are merged automatically.
Near matches are reported for manual review because similar medication names can
refer to different products/ingredients.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

SOURCE_PRES = "pres_gemini"
SOURCE_TREE = "treemed"

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Remove common dosage strengths and package/form suffixes from TreeMed product
# labels to derive a product-family name. Original product labels are preserved.
STRENGTH_RE = re.compile(
    r"(?ix)\b\d+(?:[.,]\d+)?\s*"
    r"(?:mg|mcg|µg|ug|g|gm|gr|kg|ml|l|iu|i\.u\.?|meq|mmol|%|mg/ml|mcg/ml)\b"
    r"(?:\s*/\s*\d+(?:[.,]\d+)?\s*(?:ml|mg|mcg|g|iu|i\.u\.?))?"
)
PACK_RE = re.compile(
    r"(?ix)\b\d+\s*(?:f\.?\s*c\.?\s*)?"
    r"(?:tabs?\.?|tablets?|caps?\.?|capsules?|caplets?|sachets?|ampoules?|amps?\.?|"
    r"vials?|bottles?|pcs?\.?|pieces?|supp(?:ositories)?\.?|lozenges?)\b"
)
FORM_SUFFIX_RE = re.compile(
    r"(?ix)\b(?:f\.?\s*c\.?\s*tabs?\.?|tabs?\.?|tablets?|caps?\.?|capsules?|"
    r"caplets?|susp(?:ension)?\.?|syr(?:up)?\.?|vials?|sprays?|creams?|ointments?|oint\.?|"
    r"gels?|drops?|soap|solution|soln\.?|ampoules?|amps?\.?|injection|suppositories?)\b"
)
ANNOTATION_RE = re.compile(r"\s*\([^()]*\)\s*$")
NON_WORD_RE = re.compile(r"[^\w\u0600-\u06FF]+", flags=re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+")
ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


def clean_scalar(value: Any) -> Any:
    """Return JSON-safe values and convert blank strings to None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


def normalize_name(value: Any) -> str:
    """Normalize casing, Unicode, Arabic marks/digits, and punctuation."""
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value)).translate(ARABIC_DIGITS)
    s = s.replace("ـ", "")
    s = ARABIC_DIACRITICS.sub("", s).casefold()
    s = NON_WORD_RE.sub(" ", s)
    return MULTISPACE_RE.sub(" ", s).strip()


def tree_product_base(full_name: Any) -> str:
    """Approximate a TreeMed product-family name from its full product label."""
    if full_name is None:
        return ""
    s = unicodedata.normalize("NFKC", str(full_name)).strip()
    # Drop terminal catalog annotations such as (CANCELLED) or (ILLEGAL IMPORT).
    while True:
        new_s = ANNOTATION_RE.sub("", s).strip()
        if new_s == s:
            break
        s = new_s
    s = STRENGTH_RE.sub(" ", s)
    s = PACK_RE.sub(" ", s)
    s = FORM_SUFFIX_RE.sub(" ", s)
    s = re.sub(r"\b(?:i\.?v\.?|i\.?m\.?|i\.?m\.?|oral)\b\s*$", "", s, flags=re.I)
    s = s.strip(" -_/.,;:")
    s = MULTISPACE_RE.sub(" ", s)
    return s or str(full_name).strip()


def source_type(record: dict[str, Any]) -> str:
    source = str(record.get("source", "")).strip().casefold().replace("-", "_").replace(" ", "_")
    if source in {"pres_gemini", "presgemini", "prescription", "prescriptions"}:
        return SOURCE_PRES
    if source in {"treemed", "tree_med", "tree_medication", "tree_meds"}:
        return SOURCE_TREE
    return source or "unknown"


def load_input(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load JSON or JSONL and locate the list of records."""
    if path.suffix.casefold() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
                if isinstance(obj, dict):
                    records.append(obj)
        return records, {}

    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        summary = {}
        records = []

        for item in payload:
            if not isinstance(item, dict):
                continue

            if isinstance(item.get("data_summary"), dict):
                summary = item["data_summary"]
            else:
                records.append(item)

        return records, summary
    
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be a list or an object containing a records list.")

    summary = payload.get("data_summary", {})
    if not isinstance(summary, dict):
        summary = {}

    preferred_keys = ("records", "data", "samples", "items", "entries", "medications", "results")
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, list) and (not value or any(isinstance(x, dict) and "source" in x for x in value)):
            return [x for x in value if isinstance(x, dict)], summary

    # Fall back to any top-level list containing source-tagged records.
    for key, value in payload.items():
        if key == "data_summary" or not isinstance(value, list):
            continue
        if any(isinstance(x, dict) and "source" in x for x in value):
            return [x for x in value if isinstance(x, dict)], summary

    raise ValueError(
        "Could not locate the record list. Expected a top-level list or a key such as "
        "'data', 'records', 'samples', or 'items' containing objects with a 'source' field."
    )


def prescription_medications(record: dict[str, Any]) -> list[dict[str, Any]]:
    meds = record.get("medications")
    if meds is None and isinstance(record.get("extraction"), dict):
        meds = record["extraction"].get("medications")
    if not isinstance(meds, list):
        return []
    return [m if isinstance(m, dict) else {"name": str(m)} for m in meds]


def add_variant(group: dict[str, Any], value: Any, source: str) -> None:
    value = clean_scalar(value)
    if value is None:
        return
    key = str(value).strip()
    if not key:
        return
    item = group["_variant_counts"].setdefault(key, {"count": 0, "sources": Counter()})
    item["count"] += 1
    item["sources"][source] += 1


def add_unique(group: dict[str, Any], field: str, value: Any) -> None:
    value = clean_scalar(value)
    if value is None:
        return
    values = group[field]
    if value not in values:
        values.append(value)


def make_group(key: str, initial_name: str = "") -> dict[str, Any]:
    return {
        "normalized_key": key,
        "canonical_name": initial_name or None,
        "_variant_counts": {},
        "arabic_names": [],
        "scientific_names": [],
        "drug_classes": [],
        "forms": [],
        "strengths": [],
        "routes": [],
        "doses": [],
        "frequencies": [],
        "durations": [],
        "instructions_examples": [],
        "tree_med_match_methods": [],
        "tree_med_products": [],
        "occurrences": [],
        "_sample_ids": set(),
        "_prescription_name_counts": Counter(),
    }


def build_tree_med_alias_index(
    tree_groups_by_key: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    """Map normalized TreeMed labels to product-family keys.

    Aliases are only used for exact normalized matching. Ambiguous aliases are
    retained as multiple candidates and are not auto-matched by the caller.
    """
    aliases: dict[str, set[str]] = defaultdict(set)

    for tree_key, group in tree_groups_by_key.items():
        if tree_key:
            aliases[tree_key].add(tree_key)

        for product in group.get("tree_med_products", []):
            for field in ("Name", "ArabicName", "ScientificName"):
                value = clean_scalar(product.get(field))
                alias = normalize_name(value)
                if alias:
                    aliases[alias].add(tree_key)

            # Some prescription names include the same dosage/form suffixes
            # that TreeMed product labels contain, so index the stripped label too.
            full_name = clean_scalar(product.get("Name"))
            stripped_name = normalize_name(tree_product_base(full_name))
            if stripped_name:
                aliases[stripped_name].add(tree_key)

    return aliases


def match_tree_medication_name(
    medication_name: Any,
    tree_groups_by_key: dict[str, dict[str, Any]],
    tree_alias_index: dict[str, set[str]],
) -> tuple[str | None, str | None]:
    """Return a uniquely matched TreeMed family key and match method.

    Matching order:
      1. Exact normalized TreeMed family key.
      2. Exact match after removing common strength/form/package suffixes.
      3. Exact normalized full/Arabic/scientific TreeMed alias, only when unique.

    No fuzzy matching is performed here; fuzzy candidates stay review-only.
    """
    normalized = normalize_name(medication_name)
    if not normalized:
        return None, None

    if normalized in tree_groups_by_key:
        return normalized, "exact_normalized_family_name"

    stripped = normalize_name(tree_product_base(medication_name))
    if stripped in tree_groups_by_key:
        return stripped, "exact_after_removing_form_strength_or_packaging"

    candidates = tree_alias_index.get(normalized, set())
    if len(candidates) == 1:
        return next(iter(candidates)), "unique_exact_treemed_alias"

    return None, None


def expected_summary_value(summary: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = summary.get(key)
        try:
            if value is not None:
                return int(value)
        except (ValueError, TypeError):
            pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a medication vocabulary from prescription and TreeMed data.")
    parser.add_argument("input", nargs="?", default="medications_from_prescriptions.json", help="Input .json or .jsonl file")
    parser.add_argument("-o", "--output", default="medication_vocabulary.json", help="Vocabulary JSON output path")
    parser.add_argument("--fuzzy-report", default="medication_match_review.json", help="Possible cross-source matches for manual review")
    parser.add_argument("--no-fuzzy-review", action="store_true", help="Skip near-match suggestions")
    parser.add_argument("--threshold", type=float, default=0.78, help="Similarity threshold for review suggestions (not auto-merged)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2

    try:
        records, declared_summary = load_input(input_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    actual_source_counts = Counter(source_type(r) for r in records)
    groups: dict[str, dict[str, Any]] = {}
    total_pres_samples = 0
    pres_samples_with_meds = 0
    pres_samples_without_meds = 0
    observed_pres_occurrences = 0
    occurrence_id_set: set[str] = set()
    record_occurrence_total = 0
    unknown_source_records: list[dict[str, Any]] = []

    # Group product lines by normalized product-family base while
    # preserving every original product record.
    tree_groups_by_key: dict[str, dict[str, Any]] = {}
    tree_product_record_count = 0

    for record_index, record in enumerate(records):
        record_source = source_type(record)

        if record_source != SOURCE_TREE:
            if record_source != SOURCE_PRES:
                unknown_source_records.append({
                    "record_index": record_index,
                    "source": record.get("source"),
                    "item_id": record.get("item_id"),
                    "source_rel_path": record.get("source_rel_path"),
                })
            continue

        tree_product_record_count += 1

        full_name = clean_scalar(
            record.get("Name") or record.get("name")
        )
        base_name = tree_product_base(full_name)
        key = normalize_name(base_name or full_name)

        if not key:
            key = f"unknown_treemed_{record_index}"
            base_name = str(
                full_name or f"Unknown TreeMed product {record_index + 1}"
            )

        group = groups.setdefault(
            key,
            make_group(key, base_name)
        )
        tree_groups_by_key[key] = group

        add_variant(group, base_name, SOURCE_TREE)
        add_variant(group, full_name, SOURCE_TREE)

        for field, dest in (
            ("ArabicName", "arabic_names"),
            ("ScientificName", "scientific_names"),
            ("DrugClass", "drug_classes"),
            ("Route", "routes"),
        ):
            value = clean_scalar(record.get(field))
            add_unique(group, dest, value)

            if field == "ArabicName":
                add_variant(group, value, SOURCE_TREE)

        product = {
            "Name": clean_scalar(record.get("Name")),
            "ArabicName": clean_scalar(record.get("ArabicName")),
            "ScientificName": clean_scalar(
                record.get("ScientificName")
            ),
            "DrugClass": clean_scalar(record.get("DrugClass")),
            "Route": clean_scalar(record.get("Route")),
            "source_record_index": record_index,
        }

        group["tree_med_products"].append(product)

    # Exact aliases let prescription names inherit Arabic/scientific metadata
    # from TreeMed even when the extracted name contains dosage/form text.
    tree_alias_index = build_tree_med_alias_index(tree_groups_by_key)

    # unique occurrence_id, so the final accounting can prove none were dropped
    # or accidentally duplicated.
    prescription_record_indices: list[int] = []
    for record_index, record in enumerate(records):
        if source_type(record) != SOURCE_PRES:
            continue
        total_pres_samples += 1
        prescription_record_indices.append(record_index)
        meds = prescription_medications(record)
        if meds:
            pres_samples_with_meds += 1
        else:
            pres_samples_without_meds += 1
        record_occurrence_total += len(meds)
        for med_index, med in enumerate(meds):
            observed_pres_occurrences += 1
            occurrence_id = f"record_{record_index}:med_{med_index}"
            if occurrence_id in occurrence_id_set:
                raise AssertionError(f"Duplicate internal occurrence id: {occurrence_id}")
            occurrence_id_set.add(occurrence_id)

            name = clean_scalar(med.get("name") or med.get("Name") or med.get("medication_name"))
            raw_text = clean_scalar(med.get("raw_text"))
            match_name = str(name) if name is not None else ""
            original_key = normalize_name(match_name)
            tree_match_key, tree_match_method = match_tree_medication_name(
                match_name, tree_groups_by_key, tree_alias_index
            )
            key = tree_match_key or original_key
            if not key:
                # Unknown names remain separate to avoid merging unrelated unreadable drugs.
                key = f"unknown_pres_{record_index}_{med_index}"
                match_name = f"[unknown medication — {occurrence_id}]"

            group = groups.setdefault(key, make_group(key, match_name))
            add_variant(group, name or match_name, SOURCE_PRES)
            if tree_match_method:
                add_unique(group, "tree_med_match_methods", tree_match_method)
            if name:
                group["_prescription_name_counts"][str(name)] += 1

            form = clean_scalar(med.get("form"))
            strength = clean_scalar(med.get("strength"))
            route = clean_scalar(med.get("route"))
            dose = clean_scalar(med.get("dose"))
            frequency = clean_scalar(med.get("frequency"))
            duration = clean_scalar(med.get("duration"))
            instructions = clean_scalar(med.get("instructions"))
            add_unique(group, "forms", form)
            add_unique(group, "strengths", strength)
            add_unique(group, "routes", route)
            add_unique(group, "doses", dose)
            add_unique(group, "frequencies", frequency)
            add_unique(group, "durations", duration)
            if instructions and instructions not in group["instructions_examples"]:
                group["instructions_examples"].append(instructions)
                group["instructions_examples"] = group["instructions_examples"][:10]

            sample_id = record.get("item_id") or record.get("source_rel_path") or f"record_{record_index}"
            group["_sample_ids"].add(str(sample_id))
            occurrence = {
                "occurrence_id": occurrence_id,
                "source": SOURCE_PRES,
                "source_rel_path": clean_scalar(record.get("source_rel_path")),
                "item_id": clean_scalar(record.get("item_id")),
                "folder": clean_scalar(record.get("folder")),
                "kind": clean_scalar(record.get("kind")),
                "page": clean_scalar(record.get("page")),
                "medication_index_in_sample": med_index,
                "name": name,
                "tree_med_match_method": tree_match_method,
                "raw_text": raw_text,
                "form": form,
                "strength": strength,
                "dose": dose,
                "route": route,
                "frequency": frequency,
                "duration": duration,
                "instructions": instructions,
            }
            group["occurrences"].append(occurrence)

    # Assign a readable canonical label and turn internal counters/sets into JSON.
    vocabulary: list[dict[str, Any]] = []
    for group in groups.values():
        pres_counts: Counter = group["_prescription_name_counts"]
        if pres_counts:
            canonical_name = sorted(pres_counts.items(), key=lambda pair: (-pair[1], pair[0].casefold()))[0][0]
        elif group["canonical_name"]:
            canonical_name = group["canonical_name"]
        elif group["_variant_counts"]:
            canonical_name = max(group["_variant_counts"].items(), key=lambda item: item[1]["count"])[0]
        else:
            canonical_name = group["normalized_key"]

        variants = []
        for value, info in sorted(group["_variant_counts"].items(), key=lambda item: (-item[1]["count"], item[0].casefold())):
            variants.append({"value": value, "count": info["count"], "sources": dict(info["sources"])})

        occurrences = group["occurrences"]
        tree_products = group["tree_med_products"]
        final_group = {
            "medication_id": f"med_{len(vocabulary) + 1:06d}",
            "canonical_name": canonical_name,
            "normalized_key": group["normalized_key"],
            "name_variants": variants,
            "arabic_names": group["arabic_names"],
            "scientific_names": group["scientific_names"],
            "drug_classes": group["drug_classes"],
            "forms": group["forms"],
            "strengths": group["strengths"],
            "routes": group["routes"],
            "doses_observed": group["doses"],
            "frequencies_observed": group["frequencies"],
            "durations_observed": group["durations"],
            "instruction_examples": group["instructions_examples"],
            "tree_med_match_methods": group["tree_med_match_methods"],
            "prescription_occurrence_count": len(occurrences),
            "unique_prescription_sample_count": len(group["_sample_ids"]),
            "tree_med_product_count": len(tree_products),
            "tree_med_products": tree_products,
            "occurrences": occurrences,
        }
        vocabulary.append(final_group)

    # Fuzzy suggestions are review-only; they are never merged automatically.
    review_suggestions: list[dict[str, Any]] = []
    if not args.no_fuzzy_review:
        tree_items = [
            (key, group)
            for key, group in tree_groups_by_key.items()
            if key and not key.startswith("unknown_treemed_")
        ]
        by_first_char: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for key, group in tree_items:
            by_first_char[key[0]].append((key, group))

        for group in vocabulary:
            if not group["prescription_occurrence_count"] or group["tree_med_product_count"]:
                continue
            key = group["normalized_key"]
            if not key or key.startswith("unknown_pres_"):
                continue
            candidates = []
            first = key[0]
            possible = list(by_first_char.get(first, []))
            # A second pool allows one-character prefix differences, but restricts
            # by length to avoid comparing obviously unrelated names.
            for c, items in by_first_char.items():
                if c != first and abs(len(key) - len(c)) <= 1:
                    possible.extend(items[:250])
            for cand_key, cand_group in possible:
                if abs(len(key) - len(cand_key)) > max(4, int(max(len(key), len(cand_key)) * 0.35)):
                    continue
                score = SequenceMatcher(None, key, cand_key).ratio()
                if score >= args.threshold:
                    candidates.append({
                        "tree_med_canonical_name": cand_group.get("canonical_name"),
                        "tree_med_base_key": cand_key,
                        "similarity": round(score, 4),
                        "tree_med_products": [p.get("Name") for p in cand_group.get("tree_med_products", [])[:5]],
                    })
            candidates.sort(key=lambda x: (-x["similarity"], str(x["tree_med_canonical_name"] or "")))
            if candidates:
                review_suggestions.append({
                    "prescription_canonical_name": group["canonical_name"],
                    "normalized_key": key,
                    "occurrence_count": group["prescription_occurrence_count"],
                    "possible_tree_med_matches": candidates[:5],
                    "note": "Suggestions only; not automatically merged. Verify brand/ingredient before combining.",
                })

    # Stable ordering: most frequently observed medicines first.
    vocabulary.sort(key=lambda x: (-x["prescription_occurrence_count"], x["canonical_name"].casefold()))
    # Re-number after sorting.
    for i, item in enumerate(vocabulary, 1):
        item["medication_id"] = f"med_{i:06d}"

    occurrences_in_vocab = [occ["occurrence_id"] for item in vocabulary for occ in item["occurrences"]]
    assigned_occurrences = len(occurrences_in_vocab)
    unique_assigned_occurrences = len(set(occurrences_in_vocab))

    expected_total = expected_summary_value(declared_summary, ("total", "records", "total_records"))
    expected_pres = expected_summary_value(declared_summary, ("pres_gemini", "prescription_records"))
    expected_tree = expected_summary_value(declared_summary, ("treemed", "tree_med", "tree_med_records"))

    source_counts_match_summary = None
    if expected_pres is not None and expected_tree is not None:
        source_counts_match_summary = (
            actual_source_counts[SOURCE_PRES] == expected_pres
            and actual_source_counts[SOURCE_TREE] == expected_tree
        )
    total_matches_summary = None if expected_total is None else len(records) == expected_total
    source_sum_matches_total = None if expected_total is None or (expected_pres is None or expected_tree is None) else expected_pres + expected_tree == expected_total
    occurrence_conservation = assigned_occurrences == observed_pres_occurrences == record_occurrence_total
    every_occurrence_assigned_once = assigned_occurrences == unique_assigned_occurrences == observed_pres_occurrences
    all_sources_recognized = not unknown_source_records and actual_source_counts["unknown"] == 0

    validation = {
        "input_record_count": len(records),
        "declared_total_records": expected_total,
        "total_records_match_declared_summary": total_matches_summary,
        "actual_pres_gemini_records": actual_source_counts[SOURCE_PRES],
        "declared_pres_gemini_records": expected_pres,
        "actual_treemed_records": actual_source_counts[SOURCE_TREE],
        "declared_treemed_records": expected_tree,
        "source_counts_match_declared_summary": source_counts_match_summary,
        "declared_source_counts_sum_to_declared_total": source_sum_matches_total,
        "all_source_labels_recognized": all_sources_recognized,
        "unknown_or_unexpected_source_records": unknown_source_records[:50],
        "prescription_samples": total_pres_samples,
        "prescription_samples_with_at_least_one_medication": pres_samples_with_meds,
        "prescription_samples_with_no_medications": pres_samples_without_meds,
        "raw_medication_occurrences_in_prescription_records": observed_pres_occurrences,
        "medication_occurrences_assigned_to_vocabulary": assigned_occurrences,
        "unique_occurrence_ids_assigned": unique_assigned_occurrences,
        "all_medication_occurrences_preserved": occurrence_conservation,
        "every_occurrence_assigned_exactly_once": every_occurrence_assigned_once,
        "vocabulary_entry_count": len(vocabulary),
        "tree_med_product_records": tree_product_record_count,
        "prescription_occurrences_matched_to_treemed": sum(
            1 for item in vocabulary for occurrence in item["occurrences"]
            if occurrence.get("tree_med_match_method")
        ),
        "vocabulary_entries_with_arabic_names": sum(1 for item in vocabulary if item["arabic_names"]),
        "vocabulary_entries_with_scientific_names": sum(1 for item in vocabulary if item["scientific_names"]),
        "fuzzy_suggestions_are_review_only": True,
    }

    # These checks should always hold for a successful build, regardless of whether
    # the declared data summary was stale.
    if not occurrence_conservation or not every_occurrence_assigned_once:
        print("ERROR: medication occurrence accounting failed; output not written.", file=sys.stderr)
        print(json.dumps(validation, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3

    output = {
        "metadata": {
            "input_file": str(input_path),
            "matching_policy": "Prescription names inherit TreeMed Arabic/scientific metadata via exact normalized family-name matches, names after conservative form/strength/package stripping, or unique exact TreeMed aliases. Ambiguous and fuzzy matches are review suggestions only.",
            "normalization": "Unicode NFKC, casefold, Arabic diacritic/tatweel removal, Arabic digit normalization, punctuation normalization.",
            "warning": "Vocabulary entry count and medication occurrence count are not expected to equal total input records. Prescription samples may contain zero, one, or multiple medications; TreeMed records are reference products, not prescription occurrences.",
        },
        "data_summary": validation,
        "medications": vocabulary,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if not args.no_fuzzy_review:
        review_path = Path(args.fuzzy_report)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with review_path.open("w", encoding="utf-8") as f:
            json.dump({"threshold": args.threshold, "suggestions": review_suggestions}, f, ensure_ascii=False, indent=2)

    print("\nMedication vocabulary created successfully.")
    print(f"Input records:                       {len(records):,}")
    print(f"Prescription sample records:         {total_pres_samples:,}")
    print(f"  Samples with medications:          {pres_samples_with_meds:,}")
    print(f"  Samples without medications:       {pres_samples_without_meds:,}")
    print(f"TreeMed product records:             {tree_product_record_count:,}")
    print(f"Medication occurrences:              {observed_pres_occurrences:,}")
    print(f"Vocabulary entries:                  {len(vocabulary):,}")
    print(f"Occurrences assigned exactly once:   {assigned_occurrences:,}")
    print(f"Output: {output_path.resolve()}")
    if not args.no_fuzzy_review:
        print(f"Manual match review: {Path(args.fuzzy_report).resolve()} ({len(review_suggestions):,} unmatched names with suggestions)")

    print("\nValidation checks:")
    for name in (
        "total_records_match_declared_summary",
        "source_counts_match_declared_summary",
        "declared_source_counts_sum_to_declared_total",
        "all_source_labels_recognized",
        "all_medication_occurrences_preserved",
        "every_occurrence_assigned_exactly_once",
    ):
        print(f"  {name}: {validation[name]}")

    # Return non-zero if data-summary counts disagree, but keep output for diagnosis.
    summary_checks = [total_matches_summary, source_counts_match_summary, source_sum_matches_total]
    if any(result is False for result in summary_checks) or not all_sources_recognized:
        print("\nWARNING: One or more dataset-level checks failed. Inspect data_summary in the output JSON.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
