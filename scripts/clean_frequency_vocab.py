#!/usr/bin/env python3
"""
Clean malformed frequency variants in frequency_vocab.json.

Default behavior is SAFE:
- Removes clearly corrupted/placeholder variants (e.g. "1 x", "//", "/", "1 x[?]", anything containing "?").
- Keeps a backup of the original file.
- Writes removed entries to frequency_vocab_removed_variants.json for review.
- Supports common JSON layouts where variants are strings in lists/dicts.

Usage:
    python clean_frequency_vocab.py
    python clean_frequency_vocab.py frequency_vocab.json
    python clean_frequency_vocab.py frequency_vocab.json --dry-run
"""

import argparse
import json
import re
import shutil
from pathlib import Path

PLACEHOLDER_TOKENS = {
    "", "?", "[?]", "[illegible]", "illegible", "unknown", "n/a", "na", "none", "null"
}

# Clearly malformed standalone values. Be conservative with short valid notation
# such as "1x1", "1x2", "x2", Arabic numerals, and legitimate Arabic text.
BAD_EXACT = {
    "/", "//", "\\", "\\\\", "|", "||", "-", "--", "...", "…",
    "1 x", "1x", "x", "x ", " x", "1 ×", "×", "× ",
    "[?]", "[illegible]", "illegible", "unknown", "n/a", "na", "none", "null"
}

def normalize_for_check(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    # Normalize common multiplication symbols for malformed-pattern checks.
    value = value.replace("＊", "*").replace("✕", "x").replace("×", "x")
    return value

def is_bad_variant(value) -> tuple[bool, str]:
    if not isinstance(value, str):
        return False, ""
    raw = value.strip()
    norm = normalize_for_check(raw)

    if not raw:
        return True, "empty/whitespace"
    if "?" in raw:
        return True, "contains '?' placeholder"
    if norm in PLACEHOLDER_TOKENS or norm in BAD_EXACT:
        return True, "placeholder or malformed standalone token"
    # Repeated punctuation / slash-only noise.
    if re.fullmatch(r"[/\\|]+", norm):
        return True, "punctuation-only noise"
    # A numeric dose followed only by x/×/*, e.g. "1 x", "2×", "1*x".
    if re.fullmatch(r"\d+\s*[x*]\s*", norm):
        return True, "incomplete multiplication notation"
    # Any variant made solely from punctuation and whitespace.
    if re.fullmatch(r"[\W_]+", norm, flags=re.UNICODE):
        return True, "punctuation-only/no text"
    return False, ""

def clean_node(node, path="$", removed=None, stats=None):
    if removed is None:
        removed = []
    if stats is None:
        stats = {"strings_seen": 0, "removed": 0}

    if isinstance(node, list):
        cleaned = []
        for i, item in enumerate(node):
            item_path = f"{path}[{i}]"
            if isinstance(item, str):
                stats["strings_seen"] += 1
                bad, reason = is_bad_variant(item)
                if bad:
                    removed.append({"path": item_path, "value": item, "reason": reason})
                    stats["removed"] += 1
                    continue
            cleaned.append(clean_node(item, item_path, removed, stats))
        return cleaned

    if isinstance(node, dict):
        cleaned = {}
        for key, value in node.items():
            # Don't remove dictionary keys automatically; keys may be IDs or group names.
            if isinstance(value, str):
                stats["strings_seen"] += 1
                bad, reason = is_bad_variant(value)
                if bad:
                    removed.append({"path": f"{path}.{key}", "value": value, "reason": reason})
                    stats["removed"] += 1
                    continue
            cleaned[key] = clean_node(value, f"{path}.{key}", removed, stats)
        return cleaned

    return node

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="frequency_vocab.json")
    parser.add_argument("--dry-run", action="store_true", help="report removals without changing the input")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"File not found: {input_path.resolve()}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    removed = []
    stats = {"strings_seen": 0, "removed": 0}
    cleaned = clean_node(data, removed=removed, stats=stats)

    print(f"Input: {input_path.resolve()}")
    print(f"String values checked: {stats['strings_seen']}")
    print(f"Malformed/placeholder values found: {stats['removed']}")

    report_path = input_path.with_name(input_path.stem + "_removed_variants.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(removed, f, ensure_ascii=False, indent=2)
    print(f"Removal report: {report_path.resolve()}")

    if args.dry_run:
        print("Dry run only: input file was not modified.")
        return

    backup_path = input_path.with_name(input_path.stem + ".backup.json")
    if not backup_path.exists():
        shutil.copy2(input_path, backup_path)
        print(f"Backup: {backup_path.resolve()}")

    temp_path = input_path.with_suffix(input_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temp_path.replace(input_path)
    print(f"Cleaned file written: {input_path.resolve()}")

if __name__ == "__main__":
    main()
