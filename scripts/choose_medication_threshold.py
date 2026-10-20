#!/usr/bin/env python3
"""
Explore medication-name similarity thresholds using the ORIGINAL vocabulary.

This is a read-only analysis tool:
- does not merge or modify vocabulary entries;
- counts candidate pairs at each threshold;
- highlights sudden increases in candidate volume ("elbow"/jump);
- exports candidates in score bands for manual review.

IMPORTANT:
Counts alone cannot prove which threshold is medically correct. The script can
recommend a PROVISIONAL review cutoff based on candidate-volume changes, but a
validated production threshold requires human-labeled correct/incorrect pairs.

Usage:
    python analyze_medication_thresholds.py medication_vocabulary.json

Optional:
    python analyze_medication_thresholds.py medication_vocabulary.json \
        --min-threshold 0.60 --max-threshold 0.90 --step 0.01
    python analyze_medication_thresholds.py medication_vocabulary.json --no-prefilter
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
WHITESPACE = re.compile(r"\s+")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
TATWEEL = "\u0640"

try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except ImportError:
    fuzz = None
    HAVE_RAPIDFUZZ = False


def normalize(value: Any) -> str:
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value))
    s = s.translate(ARABIC_DIGITS).translate(PERSIAN_DIGITS)
    s = s.replace(TATWEEL, "")
    s = ARABIC_DIACRITICS.sub("", s)
    s = s.casefold()
    s = PUNCTUATION.sub(" ", s)
    return WHITESPACE.sub(" ", s).strip()


def load_entries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = next(
            (data[k] for k in ("vocabulary", "entries", "medications", "items")
             if isinstance(data.get(k), list)),
            None,
        )
        if raw is None and data and all(isinstance(v, dict) for v in data.values()):
            raw = []
            for key, value in data.items():
                item = dict(value)
                item.setdefault("_mapping_key", key)
                raw.append(item)
        if raw is None:
            raise ValueError(
                "Cannot find entries. Expected a JSON list, a mapping of entries, "
                "or a dict containing vocabulary/entries/medications/items."
            )
    else:
        raise ValueError("JSON root must be a list or object.")

    entries = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            item = dict(item)
            item.setdefault("_entry_index", i)
            entries.append(item)
    return entries


def first_text(entry: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def has_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def is_tree_med_backed(entry: dict) -> bool:
    if has_list(entry.get("tree_med_products")):
        return True
    for field in (
        "tree_med_base_key", "tree_med_match_method", "tree_med_match_method_used",
        "matched_tree_med_key", "tree_med_key",
    ):
        value = entry.get(field)
        if isinstance(value, str) and value.strip():
            return True
    return any(
        has_list(entry.get(field))
        for field in ("treemed_products", "tree_med_matches", "matched_tree_med_products")
    )


def source_name(entry: dict) -> str:
    return first_text(entry, (
        "normalized_key", "prescription_normalized_key", "canonical_name",
        "prescription_canonical_name", "medication_name", "name", "display_name",
        "original_name", "raw_name", "ArabicName", "ScientificName",
    ))


def tree_name(entry: dict) -> str:
    return first_text(entry, (
        "tree_med_base_key", "normalized_key", "canonical_name",
        "tree_med_canonical_name", "name", "display_name", "Name",
        "ArabicName", "ScientificName",
    ))


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if HAVE_RAPIDFUZZ:
        return fuzz.ratio(a, b) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def eligible_by_prefilter(a: str, b: str) -> bool:
    # Fast heuristic, not exhaustive. Disable with --no-prefilter for all pairs.
    if not a or not b or a[0] != b[0]:
        return False
    return min(len(a), len(b)) / max(len(a), len(b)) >= 0.40


def make_thresholds(low: float, high: float, step: float) -> list[float]:
    # Generate descending thresholds with stable 2-decimal rounding by default.
    vals = []
    n = round((high - low) / step)
    for i in range(n + 1):
        value = round(high - i * step, 4)
        if value >= low - 1e-9:
            vals.append(value)
    if not vals or vals[-1] > low:
        vals.append(round(low, 4))
    return sorted(set(vals), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Choose a medication similarity threshold using candidate diagnostics and, when available, human labels."
    )
    parser.add_argument("input_json", nargs="?", default="medication_vocabulary.json")
    parser.add_argument("--min-threshold", type=float, default=0.60)
    parser.add_argument("--max-threshold", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--no-prefilter", action="store_true",
                        help="Compare all source-target pairs; can be much slower.")
    parser.add_argument("--output-prefix", default="medication_threshold_analysis")
    parser.add_argument("--labels-csv", default=None,
                        help="CSV containing manually reviewed candidate pairs; see generated review CSV.")
    parser.add_argument("--min-precision", type=float, default=0.95,
                        help="Target minimum precision for a labeled evaluation (default 0.95).")
    parser.add_argument("--volume-weight", type=float, default=0.15,
                        help="Weight of candidate-volume penalty in unlabeled provisional ranking (0-1).")
    parser.add_argument("--review-per-band", type=int, default=100,
                        help="Maximum candidate pairs exported per 0.01 score band for manual labeling.")
    args = parser.parse_args()

    if not (0 <= args.min_threshold < args.max_threshold <= 1):
        raise SystemExit("Require 0 <= min-threshold < max-threshold <= 1.")
    if args.step <= 0 or args.review_per_band < 1:
        raise SystemExit("--step and --review-per-band must be positive.")
    if not 0 <= args.min_precision <= 1 or not 0 <= args.volume_weight <= 1:
        raise SystemExit("--min-precision and --volume-weight must be between 0 and 1.")

    path = Path(args.input_json)
    entries = load_entries(path)
    sources, targets = [], []
    for entry in entries:
        raw = source_name(entry)
        key = normalize(raw)
        if not key:
            continue
        if is_tree_med_backed(entry):
            target_raw = tree_name(entry)
            target_key = normalize(target_raw)
            if target_key:
                targets.append({"index": entry.get("_entry_index"), "name": target_raw, "key": target_key})
        else:
            sources.append({
                "index": entry.get("_entry_index"), "name": raw, "key": key,
                "occurrence_count": len(entry.get("occurrences", []))
                    if isinstance(entry.get("occurrences"), list) else None,
            })

    target_map = {}
    for t in targets:
        target_map.setdefault(t["key"], t)
    targets = list(target_map.values())
    if not sources or not targets:
        raise SystemExit(
            f"Could not identify both source and TreeMed-backed entries "
            f"(sources={len(sources)}, targets={len(targets)}). "
            "Check the vocabulary schema and TreeMed linkage fields."
        )

    # Best score per unique source-target pair. No automatic merging is done.
    pairs = []
    skipped = 0
    for si, src in enumerate(sources, 1):
        per_source = []
        for tgt in targets:
            if src["key"] == tgt["key"]:
                continue
            if not args.no_prefilter and not eligible_by_prefilter(src["key"], tgt["key"]):
                skipped += 1
                continue
            score = similarity(src["key"], tgt["key"])
            if score >= args.min_threshold:
                per_source.append({
                    "source_index": src["index"], "source_name": src["name"],
                    "source_key": src["key"], "occurrence_count": src["occurrence_count"],
                    "tree_med_entry_index": tgt["index"], "tree_med_name": tgt["name"],
                    "tree_med_key": tgt["key"], "similarity": round(score, 4),
                })
        per_source.sort(key=lambda p: p["similarity"], reverse=True)
        # Candidate ambiguity: compare each pair to the strongest alternative for this source.
        for j, pair in enumerate(per_source):
            next_score = per_source[j + 1]["similarity"] if j + 1 < len(per_source) else None
            pair["margin_to_next_candidate"] = (
                round(pair["similarity"] - next_score, 4) if next_score is not None else None
            )
            pair["review_label"] = ""  # reviewer fills: correct / incorrect / uncertain
        pairs.extend(per_source)
        if si % 250 == 0:
            print(f"Scored source entries: {si:,}/{len(sources):,}")

    thresholds = make_thresholds(args.min_threshold, args.max_threshold, args.step)
    rows = []
    prev_count = None
    previous_band_added = None
    for threshold in thresholds:
        qualifying = [p for p in pairs if p["similarity"] >= threshold]
        count = len(qualifying)
        added = None if prev_count is None else count - prev_count
        # Diagnostics that are not truth labels:
        ambiguous = sum(
            1 for p in qualifying
            if p["margin_to_next_candidate"] is not None and p["margin_to_next_candidate"] < 0.05
        )
        low_margin_rate = ambiguous / count if count else 0.0
        band_ratio = None
        if added is not None and previous_band_added not in (None, 0):
            band_ratio = added / previous_band_added
        rows.append({
            "threshold": threshold,
            "candidate_pairs": count,
            "additional_vs_previous_higher_threshold": added,
            "low_margin_pairs": ambiguous,
            "low_margin_rate": round(low_margin_rate, 4),
            "band_growth_ratio": round(band_ratio, 3) if band_ratio is not None else None,
        })
        prev_count = count
        if added is not None:
            previous_band_added = added

    prefix = Path(args.output_prefix)
    curve_path = prefix.with_name(prefix.name + "_curve.csv")
    review_path = prefix.with_name(prefix.name + "_review.csv")
    report_path = prefix.with_name(prefix.name + "_report.json")

    with curve_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Export a stratified review sample from every score band. This is what enables
    # an evidence-based threshold decision instead of relying on candidate volume alone.
    review_rows = []
    for lower in thresholds:
        upper = round(lower + args.step, 6)
        band = [p for p in pairs if lower <= p["similarity"] < upper]
        band.sort(key=lambda p: (p["margin_to_next_candidate"] is None, p["margin_to_next_candidate"] or 0))
        if len(band) > args.review_per_band:
            # Deterministic spread across the band, including low-margin examples.
            if args.review_per_band == 1:
                band = band[:1]
            else:
                indices = sorted(set(round(i * (len(band) - 1) / (args.review_per_band - 1))
                                     for i in range(args.review_per_band)))
                band = [band[i] for i in indices]
        review_rows.extend(band)

    review_fields = [
        "source_index", "source_name", "source_key", "occurrence_count",
        "tree_med_entry_index", "tree_med_name", "tree_med_key", "similarity",
        "margin_to_next_candidate", "review_label",
    ]
    with review_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=review_fields)
        w.writeheader()
        for p in review_rows:
            w.writerow({k: p.get(k, "") for k in review_fields})

    report = {
        "input_file": str(path),
        "input_entries": len(entries),
        "likely_prescription_only_entries": len(sources),
        "likely_tree_med_backed_entries_before_key_deduplication": sum(1 for e in entries if is_tree_med_backed(e)),
        "unique_tree_med_keys": len(targets),
        "similarity_metric": "RapidFuzz fuzz.ratio" if HAVE_RAPIDFUZZ else "difflib SequenceMatcher.ratio",
        "prefilter_enabled": not args.no_prefilter,
        "pairs_at_or_above_minimum_threshold": len(pairs),
        "pairs_skipped_by_prefilter": skipped,
        "threshold_curve": rows,
        "methodology": {
            "candidate_volume": "diagnostic factor only; not treated as ground truth",
            "ambiguity": "fraction of qualifying pairs whose score is within 0.05 of the next-best candidate for the same source",
            "human_labels": "needed to estimate correctness and select a defensible production threshold",
            "provisional_only": True,
        },
        "safety": {"review_only": True, "automatic_merges_performed": 0},
    }

    if args.labels_csv:
        label_path = Path(args.labels_csv)
        labels = {}
        with label_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            needed = {"source_key", "tree_med_key", "review_label"}
            if not reader.fieldnames or not needed.issubset(set(reader.fieldnames)):
                raise SystemExit(
                    f"Labels CSV must contain columns: {', '.join(sorted(needed))}. "
                    "Fill review_label with correct, incorrect, or uncertain."
                )
            for row in reader:
                label = normalize(row.get("review_label", ""))
                if label in {"correct", "yes", "true", "1", "match"}:
                    labels[(normalize(row["source_key"]), normalize(row["tree_med_key"]))] = 1
                elif label in {"incorrect", "no", "false", "0", "nonmatch", "not match"}:
                    labels[(normalize(row["source_key"]), normalize(row["tree_med_key"]))] = 0

        labeled = []
        for p in pairs:
            k = (p["source_key"], p["tree_med_key"])
            if k in labels:
                labeled.append((p, labels[k]))
        if not labeled:
            raise SystemExit("No labeled pairs matched the generated source_key/tree_med_key pairs.")
        total_positive = sum(y for _, y in labeled)
        eval_rows = []
        for threshold in thresholds:
            selected = [(p, y) for p, y in labeled if p["similarity"] >= threshold]
            tp = sum(y for _, y in selected)
            fp = len(selected) - tp
            fn = total_positive - tp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / total_positive if total_positive else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            eval_rows.append({
                "threshold": threshold, "labeled_pairs_accepted": len(selected),
                "true_positives": tp, "false_positives": fp, "false_negatives": fn,
                "precision": round(precision, 4), "recall": round(recall, 4),
                "f1": round(f1, 4), "meets_min_precision": precision >= args.min_precision,
            })
        eligible = [r for r in eval_rows if r["meets_min_precision"] and r["labeled_pairs_accepted"] > 0]
        # Under the precision constraint, choose maximum recall; tie-break with F1, then stricter threshold.
        best = max(eligible, key=lambda r: (r["recall"], r["f1"], r["threshold"])) if eligible else None
        eval_path = prefix.with_name(prefix.name + "_labeled_evaluation.csv")
        with eval_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
            w.writeheader()
            w.writerows(eval_rows)
        report["labeled_evaluation"] = {
            "labels_file": str(label_path),
            "matched_labeled_pairs": len(labeled),
            "positive_labels": total_positive,
            "negative_labels": len(labeled) - total_positive,
            "minimum_precision_target": args.min_precision,
            "evaluation_rows": eval_rows,
            "recommended_threshold": best["threshold"] if best else None,
            "recommendation_note": (
                "Selected among tested thresholds to maximize recall while meeting the minimum precision target, "
                "with F1 and stricter threshold as tie-breakers. This is only as reliable as the labeled sample; "
                "use a representative, stratified sample and keep uncertain pairs out of the metrics."
            ),
            "evaluation_csv": str(eval_path),
        }
    else:
        # Multi-factor provisional ranking: avoid thresholds with high ambiguity and
        # extreme growth, but do not pretend these unsupervised signals establish correctness.
        max_count = max((r["candidate_pairs"] for r in rows), default=1) or 1
        max_added = max((r["additional_vs_previous_higher_threshold"] or 0 for r in rows), default=1) or 1
        for r in rows:
            volume_norm = r["candidate_pairs"] / max_count
            growth_norm = min((r["additional_vs_previous_higher_threshold"] or 0) / max_added, 1.0)
            # Penalize low-margin ambiguity and explosive growth; retain candidate count as a smaller factor.
            r["unlabeled_diagnostic_score"] = round(
                1.0 - (0.45 * r["low_margin_rate"] + 0.35 * growth_norm + args.volume_weight * volume_norm), 4
            )
        # Only an exploratory ranking, never call this "best" without labels.
        ranked = sorted(rows, key=lambda r: (r["unlabeled_diagnostic_score"], r["threshold"]), reverse=True)
        report["unlabeled_diagnostic_ranking"] = ranked
        report["provisional_threshold_for_manual_review"] = ranked[0]["threshold"] if ranked else None
        report["recommendation_note"] = (
            "This ranking combines candidate volume, sudden growth, and score-margin ambiguity, but these are "
            "unlabeled proxies, not correctness. It is useful only to prioritize thresholds for manual review. "
            "Do NOT replace 0.78 in production solely from this ranking. Fill the review CSV labels, then rerun "
            "with --labels-csv to choose a threshold against measured precision/recall."
        )

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nThreshold analysis complete — NO MERGES PERFORMED.")
    print(f"Source entries: {len(sources):,} | unique TreeMed keys: {len(targets):,}")
    print(f"Candidate pairs scoring >= {args.min_threshold:.2f}: {len(pairs):,}")
    print(f"Curve: {curve_path}")
    print(f"Review sample: {review_path}")
    print(f"Report: {report_path}")
    if args.labels_csv:
        ev = report["labeled_evaluation"]
        print(f"Matched labeled pairs: {ev['matched_labeled_pairs']:,}")
        print(f"Minimum precision target: {args.min_precision:.1%}")
        print(f"Recommended threshold: {ev['recommended_threshold']}")
        print(f"Evaluation: {ev['evaluation_csv']}")
    else:
        print("No human labels supplied: any threshold suggestion is provisional only.")
        print("Review the CSV and label review_label as correct / incorrect / uncertain.")
        print("Then rerun with --labels-csv <review_csv> to evaluate precision and recall.")
    if not args.no_prefilter:
        print("NOTE: prefilter is enabled; use --no-prefilter for exhaustive comparison (slower).")


if __name__ == "__main__":
    main()
