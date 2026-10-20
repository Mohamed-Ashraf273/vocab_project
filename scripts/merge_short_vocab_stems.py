#!/usr/bin/env python3
"""Merge prescription-only short stubs into a unique TreeMed brand family.

A source row is merged only when:
  - it has prescription occurrences and no TreeMed products
  - its brand token uniquely prefixes exactly one TreeMed first-token family
  - the token is long enough that ``asp`` / ``dexa`` stay out (ambiguous)

Example: ``babeton`` -> ``babetone``. ``Asp`` is skipped (Aspocard vs Aspocid).

The target canonical name is kept (catalog). The stub becomes a name_variant.
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GENERIC_STUBS = {
    "saline",
    "glycerin",
    "glycerine",
    "insulin",
    "vitamin",
    "calcium",
    "iron",
    "zinc",
    "water",
    "cream",
    "drops",
    "syrup",
    "tablet",
    "capsule",
}

from scripts.apply_medication_merges import (  # noqa: E402
    medication_list,
    merge_medication_entries,
    normalize_name,
    read_json,
    recompute_counts,
    write_json,
)


def brand_token(med: dict) -> str:
    key = normalize_name(med.get("normalized_key") or med.get("canonical_name") or "")
    parts = key.split()
    return parts[0] if parts else ""


def is_tree_backed(med: dict) -> bool:
    return int(med.get("tree_med_product_count", 0) or 0) > 0


def is_prescription_stub(med: dict) -> bool:
    return int(med.get("prescription_occurrence_count", 0) or 0) > 0 and not is_tree_backed(med)


def pick_family_target(family: list[dict]) -> dict:
    exact = [m for m in family if brand_token(m) == normalize_name(m.get("normalized_key") or "")]
    pool = exact or family
    return min(
        pool,
        key=lambda m: (
            len(str(m.get("normalized_key") or "")),
            -int(m.get("tree_med_product_count", 0) or 0),
            str(m.get("canonical_name") or "").casefold(),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge unique short OCR stubs into TreeMed brand families.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=ROOT / "reports" / "med_vocab" / "medication_vocab.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Default: overwrite input",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "med_vocab" / "short_stem_merge_report.json",
    )
    parser.add_argument("--min-token-length", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output or args.input

    payload = read_json(args.input)
    meds = medication_list(payload)
    working = copy.deepcopy(meds)

    by_brand: dict[str, list[dict]] = defaultdict(list)
    tree_brands: set[str] = set()
    for med in working:
        token = brand_token(med)
        if token and is_tree_backed(med):
            by_brand[token].append(med)
            tree_brands.add(token)

    merged: list[dict] = []
    skipped: list[dict] = []
    removed_ids: set[str] = set()

    for source in working:
        if not is_prescription_stub(source):
            continue
        src_id = str(source.get("medication_id") or "")
        if src_id in removed_ids:
            continue
        token = brand_token(source)
        source_key = normalize_name(source.get("normalized_key") or "")
        if len(source_key.split()) != 1:
            skipped.append({
                "source_id": src_id,
                "source_canonical": source.get("canonical_name"),
                "decision": "skip",
                "reason": "not_a_single_token_stub",
                "token": token,
            })
            continue
        if token in GENERIC_STUBS:
            skipped.append({
                "source_id": src_id,
                "source_canonical": source.get("canonical_name"),
                "decision": "skip",
                "reason": "generic_stub",
                "token": token,
            })
            continue
        if len(token) < args.min_token_length:
            skipped.append({
                "source_id": src_id,
                "source_canonical": source.get("canonical_name"),
                "decision": "skip",
                "reason": "token_too_short",
                "token": token,
            })
            continue

        prefix_families = sorted(b for b in tree_brands if b.startswith(token) and b != token)
        families = prefix_families
        if len(prefix_families) != 1:
            skipped.append({
                "source_id": src_id,
                "source_canonical": source.get("canonical_name"),
                "decision": "skip",
                "reason": "ambiguous_or_no_tree_brand_family",
                "token": token,
                "candidate_brands": families[:20],
                "candidate_brand_count": len(families),
            })
            continue

        brand = families[0]
        if (len(token) / max(len(brand), 1)) < 0.75:
            skipped.append({
                "source_id": src_id,
                "source_canonical": source.get("canonical_name"),
                "decision": "skip",
                "reason": "stub_too_short_relative_to_brand",
                "token": token,
                "brand_family": brand,
            })
            continue
        family = by_brand[brand]
        if not family:
            skipped.append({
                "source_id": src_id,
                "source_canonical": source.get("canonical_name"),
                "decision": "skip",
                "reason": "empty_family",
                "token": token,
            })
            continue

        target = pick_family_target(family)
        if target is source:
            continue
        merge_medication_entries(target, source)
        methods = target.setdefault("tree_med_match_methods", [])
        if "unique_short_stem_merge" not in methods:
            methods.append("unique_short_stem_merge")
        removed_ids.add(src_id)
        merged.append({
            "source_id": src_id,
            "source_canonical": source.get("canonical_name"),
            "source_key": source.get("normalized_key"),
            "target_id": target.get("medication_id"),
            "target_canonical": target.get("canonical_name"),
            "target_key": target.get("normalized_key"),
            "brand_family": brand,
            "decision": "merged",
        })

    kept = [m for m in working if str(m.get("medication_id") or "") not in removed_ids]
    for med in kept:
        recompute_counts(med)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "output": str(output.resolve()),
        "dry_run": args.dry_run,
        "min_token_length": args.min_token_length,
        "summary": {
            "records_before": len(working),
            "records_after": len(kept),
            "merges_applied": len(merged),
            "skipped": len(skipped),
        },
        "merged": merged,
        "skipped": skipped,
        "note": (
            "Target catalog canonical is preserved; stub names are name_variants. "
            "Rebuild Chroma after applying so removed short IDs are not retrieved."
        ),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    print(f"Merges: {len(merged):,}  skipped: {len(skipped):,}  before: {len(working):,} after: {len(kept):,}")
    print(f"Report: {args.report}")
    if args.dry_run:
        print("Dry run: vocabulary JSON not written.")
        return 0

    if isinstance(payload, dict):
        payload["medications"] = kept
        meta = payload.setdefault("metadata", {})
        meta["short_stem_merge"] = {
            "script": "scripts/merge_short_vocab_stems.py",
            "merges_applied": len(merged),
            "min_token_length": args.min_token_length,
        }
        summary = payload.setdefault("data_summary", {})
        summary["vocabulary_entry_count"] = len(kept)
        summary["short_stem_merges_applied"] = len(merged)
    else:
        payload = kept

    tmp = output.with_suffix(output.suffix + ".tmp")
    write_json(tmp, payload)
    tmp.replace(output)
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
