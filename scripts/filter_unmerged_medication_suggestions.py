#!/usr/bin/env python3
"""
Create a review JSON containing only merge suggestions that were NOT applied.

Inputs:
  medication_match_review.json
  medication_merge_decisions.json

Output:
  medication_match_review_unmerged.json

The script keeps the original review JSON structure and filters its
"suggestions" list using the merge decisions report. It does not modify inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def get_suggestions(data: Any) -> list[dict]:
    if isinstance(data, dict) and isinstance(data.get("suggestions"), list):
        return data["suggestions"]
    raise ValueError(
        "The review JSON must be an object containing a 'suggestions' list."
    )


def get_decision_records(data: Any) -> list[dict]:
    """
    Supports common decision-report layouts:
      - {"decisions": [...]}
      - {"results": [...]}
      - {"items": [...]}
      - a top-level list
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ("decisions", "results", "items", "merge_decisions"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

    raise ValueError(
        "Could not find a decision list in medication_merge_decisions.json. "
        "Expected a top-level list or one of: decisions, results, items, merge_decisions."
    )


def first_value(record: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def norm_key(value: Any) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).casefold().split())


def is_merged(decision: dict) -> bool | None:
    """
    Infer applied status from common fields. Returns None when no clear status
    can be inferred, to avoid silently treating unknown decisions as unmerged.
    """
    status = first_value(
        decision,
        ("status", "decision", "action", "result", "outcome", "merge_status"),
    )
    if isinstance(status, str):
        s = status.strip().casefold().replace("-", "_").replace(" ", "_")
        if s in {
            "merged", "merge_applied", "applied", "accepted", "success",
            "merged_successfully", "auto_merged", "true",
        }:
            return True
        if s in {
            "not_merged", "left_for_review", "review", "needs_review",
            "rejected", "skipped", "not_applied", "ambiguous", "false",
        }:
            return False

    for key in ("merged", "merge_applied", "applied", "was_merged", "success"):
        value = decision.get(key)
        if isinstance(value, bool):
            return value

    return None


def suggestion_key(suggestion: dict) -> str | None:
    value = first_value(
        suggestion,
        ("normalized_key", "prescription_canonical_name", "canonical_name", "name"),
    )
    return norm_key(value)


def decision_key(decision: dict) -> str | None:
    value = first_value(
        decision,
        (
            "normalized_key",
            "source_normalized_key",
            "prescription_normalized_key",
            "prescription_canonical_name",
            "source_canonical_name",
            "source_name",
            "prescription_name",
            "source_key",
            "canonical_name",
        ),
    )
    return norm_key(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter medication_match_review.json to only unmerged suggestions."
    )
    parser.add_argument(
        "review_json",
        nargs="?",
        default="medication_match_review.json",
        help="Original review JSON (default: medication_match_review.json)",
    )
    parser.add_argument(
        "decisions_json",
        nargs="?",
        default="medication_merge_decisions.json",
        help="Merge decisions JSON (default: medication_merge_decisions.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="medication_match_review_unmerged.json",
        help="Output path (default: medication_match_review_unmerged.json)",
    )
    args = parser.parse_args()

    review_path = Path(args.review_json)
    decisions_path = Path(args.decisions_json)
    output_path = Path(args.output)

    review_data = load_json(review_path)
    decisions_data = load_json(decisions_path)
    suggestions = get_suggestions(review_data)
    decisions = get_decision_records(decisions_data)

    # Index decisions by normalized source identity.
    decisions_by_key: dict[str, list[dict]] = {}
    for record in decisions:
        key = decision_key(record)
        if key:
            decisions_by_key.setdefault(key, []).append(record)

    kept: list[dict] = []
    merged_count = 0
    unmatched_count = 0
    unknown_status_count = 0

    for suggestion in suggestions:
        key = suggestion_key(suggestion)
        matching_decisions = decisions_by_key.get(key, []) if key else []

        if not matching_decisions:
            # No decision found: keep the suggestion, but report it as unmatched.
            kept.append(suggestion)
            unmatched_count += 1
            continue

        statuses = [is_merged(record) for record in matching_decisions]
        known_statuses = [s for s in statuses if s is not None]

        if any(status is True for status in known_statuses):
            merged_count += 1
            continue

        if known_statuses and all(status is False for status in known_statuses):
            kept.append(suggestion)
            continue

        # Unknown status: keep it rather than dropping it, and surface the issue.
        kept.append(suggestion)
        unknown_status_count += 1

    output_data = dict(review_data)
    output_data["suggestions"] = kept
    output_data["filter_metadata"] = {
        "source_review_json": str(review_path),
        "source_decisions_json": str(decisions_path),
        "original_suggestion_count": len(suggestions),
        "removed_as_merged_count": merged_count,
        "unmerged_or_unmatched_kept_count": len(kept),
        "unmatched_decision_count": unmatched_count,
        "unknown_decision_status_count": unknown_status_count,
        "note": (
            "Suggestions without a matching decision or with an unrecognized "
            "decision status are retained for safety; inspect them before use."
        ),
    }
    dump_json(output_path, output_data)

    print("Filtering complete.")
    print(f"Original suggestions:              {len(suggestions)}")
    print(f"Removed as already merged:         {merged_count}")
    print(f"Kept in output:                    {len(kept)}")
    print(f"No matching decision found:        {unmatched_count}")
    print(f"Unrecognized decision status:      {unknown_status_count}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
