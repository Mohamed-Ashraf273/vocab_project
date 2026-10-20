#!/usr/bin/env python3
"""
Read-only auditor for medication_vocab.json.

Usage:
    python audit_medication_vocab.py
    python audit_medication_vocab.py --input "D:\\projects\\medical_ocr\\reports\\med_vocab\\medication_vocab.json"
    python audit_medication_vocab.py --input medication_vocab.json --out-dir vocab_audit

Outputs:
    vocab_audit_report.json
    vocab_audit_report.txt
    vocab_audit_records.csv

The script never modifies the source JSON or any database.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path(
    r"D:\projects\medical_ocr\reports\med_vocab\medication_vocab.json"
)

# Likely field names seen in medication vocabulary schemas. The script also
# searches recursively for relevant keys, so unexpected schemas are still audited.
ID_KEYS = {"id", "medication_id", "medicine_id", "drug_id", "uuid"}
CANONICAL_KEYS = {
    "canonical_name", "name", "medication_name", "medicine_name",
    "drug_name", "generic_name", "brand_name",
}
ALIAS_KEYS = {
    "name_variants", "variants", "aliases", "alias", "name_variants_en",
    "english_names", "name_variants_ar", "arabic_names", "scientific_names",
    "synonyms", "alternate_names", "alternative_names",
}
STRENGTH_KEYS = {"strength", "strengths", "dose_strength", "concentration"}
FORM_KEYS = {"form", "forms", "dosage_form", "dosage_forms", "presentation"}
TEXT_KEYS = {
    "canonical_name", "name", "medication_name", "medicine_name", "drug_name",
    "generic_name", "brand_name", "name_variants", "variants", "aliases",
    "alias", "name_variants_en", "english_names", "name_variants_ar",
    "arabic_names", "scientific_names", "synonyms", "alternate_names",
    "alternative_names", "strength", "strengths", "dose_strength",
    "concentration", "form", "forms", "dosage_form", "dosage_forms",
    "presentation",
}

def norm_key(value: str) -> str:
    return re.sub(r"[\s\-]+", "_", value.strip().casefold())

def normalize_name(value: str) -> str:
    """Conservative comparison key only; never used to rewrite source values."""
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    value = value.replace("ى", "ي")
    value = re.sub(r"[\u064B-\u065F\u0670]", "", value)
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())

def as_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                # Keep complex objects visible in schema audit, not as aliases.
                continue
        return out
    return []

def walk_dicts(obj: Any, path: str = "$"):
    if isinstance(obj, dict):
        yield path, obj
        for key, value in obj.items():
            yield from walk_dicts(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from walk_dicts(value, f"{path}[{i}]")

def find_values_recursive(record: dict, accepted_keys: set[str]) -> list[tuple[str, Any]]:
    found = []
    for path, d in walk_dicts(record):
        for key, value in d.items():
            if norm_key(str(key)) in accepted_keys:
                found.append((f"{path}.{key}", value))
    return found

def first_value(record: dict, accepted_keys: set[str]) -> Any:
    for key, value in record.items():
        if norm_key(str(key)) in accepted_keys:
            return value
    return None

def extract_records(data: Any) -> tuple[list[dict], str]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)], "$"
    if isinstance(data, dict):
        # Prefer common collection containers if one is present.
        for key in ("medications", "medication_vocab", "items", "records", "data", "vocabulary"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)], f"$.{key}"
        # If the root itself looks like a single medication record.
        root_keys = {norm_key(str(k)) for k in data}
        if root_keys & (CANONICAL_KEYS | ID_KEYS):
            return [data], "$"
        # Otherwise try to find a dict-of-records.
        dict_values = [(k, v) for k, v in data.items() if isinstance(v, dict)]
        if dict_values:
            records = []
            for key, value in dict_values:
                record = dict(value)
                record.setdefault("_outer_key", str(key))
                records.append(record)
            return records, "$<dict-values>"
    return [], "$"

def safe_repr(value: Any, limit: int = 180) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit - 3] + "..."

def main():
    parser = argparse.ArgumentParser(description="Audit medication_vocab.json without modifying it.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to medication_vocab.json")
    parser.add_argument("--out-dir", type=Path, default=Path("vocab_audit"), help="Output directory")
    parser.add_argument("--max-examples", type=int, default=100, help="Maximum examples per issue type")
    args = parser.parse_args()

    input_path = args.input.expanduser()
    out_dir = args.out_dir.expanduser()

    if not input_path.is_file():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        print('Run with --input "FULL_PATH_TO\\medication_vocab.json"', file=sys.stderr)
        sys.exit(2)

    try:
        raw_bytes = input_path.read_bytes()
        # utf-8-sig tolerates a UTF-8 BOM if present.
        data = json.loads(raw_bytes.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        print(f"ERROR: File is not valid UTF-8: {exc}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}", file=sys.stderr)
        sys.exit(2)

    records, records_path = extract_records(data)
    issues: list[dict[str, Any]] = []
    record_summaries: list[dict[str, Any]] = []
    key_counts = Counter()
    type_counts = Counter()
    all_alias_occurrences = defaultdict(list)
    canonical_occurrences = defaultdict(list)
    id_occurrences = defaultdict(list)
    alias_field_shapes = Counter()
    key_examples = defaultdict(list)
    record_alias_count = []
    missing_canonical = 0
    missing_id = 0
    records_with_no_aliases = 0
    nested_records = 0

    def add_issue(severity: str, code: str, message: str, record_index=None, path=None, value=None):
        issues.append({
            "severity": severity,
            "code": code,
            "record_index": record_index,
            "path": path,
            "message": message,
            "value": safe_repr(value) if value is not None else None,
        })

    for i, record in enumerate(records):
        record_path = f"{records_path}[{i}]"
        for key, value in record.items():
            nk = norm_key(str(key))
            key_counts[nk] += 1
            type_counts[f"{nk}:{type(value).__name__}"] += 1
            if len(key_examples[nk]) < 3:
                key_examples[nk].append(safe_repr(value, 100))
            if isinstance(value, (dict, list)):
                nested_records += 1

        raw_id = first_value(record, ID_KEYS)
        if raw_id is None and record.get("_outer_key") is not None:
            raw_id = record["_outer_key"]
        raw_canonical = first_value(record, CANONICAL_KEYS)
        if raw_id is None or (isinstance(raw_id, str) and not raw_id.strip()):
            missing_id += 1
            add_issue("warning", "MISSING_ID", "No non-empty ID-like field found.", i, record_path)
        else:
            id_key = str(raw_id).strip().casefold()
            id_occurrences[id_key].append({"record_index": i, "path": record_path, "canonical": raw_canonical})

        if not isinstance(raw_canonical, str) or not raw_canonical.strip():
            missing_canonical += 1
            add_issue("error", "MISSING_CANONICAL_NAME", "No non-empty top-level canonical/name field found.", i, record_path, raw_canonical)
            canonical = ""
        else:
            canonical = raw_canonical.strip()
            canonical_occurrences[normalize_name(canonical)].append({
                "record_index": i, "path": record_path, "original": canonical, "id": raw_id
            })
            if canonical != raw_canonical:
                add_issue("info", "CANONICAL_OUTER_WHITESPACE", "Canonical name has leading/trailing whitespace.", i, record_path, raw_canonical)
            if "\ufffd" in canonical:
                add_issue("error", "REPLACEMENT_CHARACTER", "Canonical name contains Unicode replacement character.", i, record_path, canonical)
            if any(unicodedata.category(ch) == "Cc" and ch not in "\t\n\r" for ch in canonical):
                add_issue("warning", "CONTROL_CHARACTER", "Canonical name contains a control character.", i, record_path, canonical)

        alias_items: list[tuple[str, str]] = []
        alias_fields = find_values_recursive(record, ALIAS_KEYS)
        for field_path, value in alias_fields:
            alias_field_shapes[f"{field_path.rsplit('.', 1)[-1].split('[')[0]}:{type(value).__name__}"] += 1
            if isinstance(value, str):
                if not value.strip():
                    add_issue("warning", "EMPTY_ALIAS_FIELD", "Alias field is an empty string.", i, field_path, value)
                else:
                    # Deliberately flag, do not auto-split: commas may be part of source conventions.
                    if re.search(r"[,;،]", value):
                        add_issue("review", "POSSIBLE_UNSPLIT_ALIASES", "Alias field is a string containing comma/semicolon delimiters; verify builder splits it.", i, field_path, value)
                    alias_items.append((field_path, value))
            elif isinstance(value, list):
                for j, alias in enumerate(value):
                    if isinstance(alias, str):
                        if not alias.strip():
                            add_issue("warning", "EMPTY_ALIAS_ITEM", "Alias list contains an empty string.", i, f"{field_path}[{j}]", alias)
                        else:
                            alias_items.append((f"{field_path}[{j}]", alias))
                    elif isinstance(alias, (dict, list)):
                        add_issue("review", "COMPLEX_ALIAS_ITEM", "Alias list contains a non-string item; inspect schema/builder.", i, f"{field_path}[{j}]", alias)
                    else:
                        add_issue("warning", "NON_STRING_ALIAS_ITEM", "Alias list contains a non-string item.", i, f"{field_path}[{j}]", alias)
            elif value is not None:
                add_issue("warning", "NON_LIST_ALIAS_FIELD", "Alias field is neither a string nor list.", i, field_path, value)

        if not alias_items:
            records_with_no_aliases += 1

        # Canonical names also count as searchable aliases for duplicate/collision checks.
        if canonical:
            alias_items.append((f"{record_path}.<canonical>", canonical))

        dedup_local = Counter(normalize_name(alias) for _, alias in alias_items if normalize_name(alias))
        for norm, count in dedup_local.items():
            if count > 1:
                vals = [a for _, a in alias_items if normalize_name(a) == norm]
                add_issue("info", "DUPLICATE_ALIAS_WITHIN_RECORD", "Repeated alias/canonical name within the same record after normalization.", i, record_path, vals)

        for field_path, alias in alias_items:
            normalized = normalize_name(alias)
            if not normalized:
                add_issue("warning", "ALIAS_NORMALIZES_EMPTY", "Alias becomes empty after normalization.", i, field_path, alias)
                continue
            all_alias_occurrences[normalized].append({
                "record_index": i, "path": field_path, "original": alias,
                "canonical": canonical, "id": raw_id
            })
        record_alias_count.append(max(0, len(alias_items) - (1 if canonical else 0)))

        # Inspect likely metadata fields without assuming every schema has them.
        for path, value in walk_dicts(record):
            for key, val in value.items():
                nk = norm_key(str(key))
                if nk in STRENGTH_KEYS and val is not None and not isinstance(val, (str, int, float, list)):
                    add_issue("warning", "UNEXPECTED_STRENGTH_TYPE", "Strength field has an unexpected type.", i, f"{path}.{key}", val)
                if nk in FORM_KEYS and val is not None and not isinstance(val, (str, list)):
                    add_issue("warning", "UNEXPECTED_FORM_TYPE", "Form field has an unexpected type.", i, f"{path}.{key}", val)

        record_summaries.append({
            "record_index": i,
            "id": raw_id,
            "canonical_name": canonical or None,
            "alias_count_excluding_canonical": record_alias_count[-1],
            "top_level_keys": list(record.keys()),
            "record_json": safe_repr(record, 1200),
        })

    duplicate_ids = [
        {"id": key, "records": vals}
        for key, vals in id_occurrences.items() if len(vals) > 1
    ]
    duplicate_canonical_names = [
        {"normalized_name": key, "records": vals}
        for key, vals in canonical_occurrences.items() if key and len(vals) > 1
    ]
    alias_collisions = []
    for key, vals in all_alias_occurrences.items():
        unique_ids = {str(v["id"]) for v in vals}
        unique_canonicals = {normalize_name(str(v["canonical"] or "")) for v in vals}
        if len(unique_ids) > 1 or len(unique_canonicals) > 1:
            alias_collisions.append({"normalized_alias": key, "occurrences": vals})

    # Report likely suspiciously short aliases, but do not call them errors automatically.
    short_aliases = []
    for norm, vals in all_alias_occurrences.items():
        if len(norm.replace(" ", "")) <= 2:
            short_aliases.append({"normalized_alias": norm, "occurrences": vals})

    issue_counts = Counter(x["code"] for x in issues)
    severity_counts = Counter(x["severity"] for x in issues)

    report = {
        "input_file": str(input_path.resolve()),
        "file_size_bytes": len(raw_bytes),
        "json_root_type": type(data).__name__,
        "record_container_path": records_path,
        "record_count": len(records),
        "missing_id_count": missing_id,
        "missing_canonical_name_count": missing_canonical,
        "records_with_no_aliases_count": records_with_no_aliases,
        "average_aliases_per_record_excluding_canonical": (
            round(sum(record_alias_count) / len(record_alias_count), 3) if record_alias_count else 0
        ),
        "top_level_key_counts": dict(key_counts.most_common()),
        "top_level_key_type_counts": dict(type_counts.most_common()),
        "alias_field_shapes": dict(alias_field_shapes.most_common()),
        "duplicate_id_groups": duplicate_ids,
        "duplicate_canonical_name_groups": duplicate_canonical_names,
        "cross_medication_alias_collisions": alias_collisions,
        "short_aliases_review": short_aliases,
        "issue_counts_by_code": dict(issue_counts.most_common()),
        "issue_counts_by_severity": dict(severity_counts),
        "issues": issues,
        "record_summaries": record_summaries,
        "notes": [
            "This is a structural/data-quality audit, not proof that a medication name is medically correct.",
            "Similar-looking names are not automatically merged or labeled as duplicates.",
            "Cross-medication alias collisions may be legitimate but should be reviewed.",
            "If the source data has no strength/form fields, this report cannot validate them.",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "vocab_audit_report.json"
    txt_path = out_dir / "vocab_audit_report.txt"
    csv_path = out_dir / "vocab_audit_records.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "MEDICATION VOCABULARY AUDIT",
        "=" * 32,
        f"Input: {input_path.resolve()}",
        f"Size: {len(raw_bytes):,} bytes",
        f"Root type: {type(data).__name__}",
        f"Record container: {records_path}",
        f"Medication records: {len(records):,}",
        f"Missing IDs: {missing_id:,}",
        f"Missing canonical names: {missing_canonical:,}",
        f"Records with no aliases: {records_with_no_aliases:,}",
        f"Average aliases per record (excluding canonical): {report['average_aliases_per_record_excluding_canonical']}",
        "",
        "ISSUE COUNTS",
        "-" * 32,
    ]
    if issue_counts:
        for code, count in issue_counts.most_common():
            lines.append(f"{code}: {count}")
    else:
        lines.append("No structural issues detected by these checks.")
    lines += [
        "",
        f"Duplicate ID groups: {len(duplicate_ids)}",
        f"Duplicate canonical-name groups: {len(duplicate_canonical_names)}",
        f"Cross-medication alias collisions: {len(alias_collisions)}",
        f"Short aliases for review: {len(short_aliases)}",
        "",
        "TOP-LEVEL KEYS",
        "-" * 32,
    ]
    for key, count in key_counts.most_common():
        lines.append(f"{key}: {count}")
    lines += ["", "ISSUE EXAMPLES", "-" * 32]
    for issue in issues[: max(0, args.max_examples)]:
        lines.append(
            f"[{issue['severity'].upper()}] {issue['code']} | "
            f"record={issue['record_index']} | path={issue['path']} | "
            f"{issue['message']} | value={issue['value']}"
        )
    if len(issues) > args.max_examples:
        lines.append(f"... {len(issues) - args.max_examples} more issues in JSON report.")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["record_index", "id", "canonical_name", "alias_count_excluding_canonical", "top_level_keys", "record_json"],
        )
        writer.writeheader()
        for row in record_summaries:
            writer.writerow({
                **row,
                "top_level_keys": json.dumps(row["top_level_keys"], ensure_ascii=False),
            })

    print("Audit complete. Source JSON was not modified.")
    print(f"Records: {len(records):,}")
    print(f"Duplicate ID groups: {len(duplicate_ids):,}")
    print(f"Duplicate canonical-name groups: {len(duplicate_canonical_names):,}")
    print(f"Cross-medication alias collisions: {len(alias_collisions):,}")
    print(f"Flagged issues: {len(issues):,}")
    print(f"Reports written to: {out_dir.resolve()}")
    print(f" - {json_path.name}")
    print(f" - {txt_path.name}")
    print(f" - {csv_path.name}")

if __name__ == "__main__":
    main()
