import json
import string
import unicodedata
from pathlib import Path

INPUT_JSON = Path("medication_vocab.json")
OUTPUT_JSON = Path("medications_with_weird_chars.json")

# All standard ASCII punctuation, including / and \
WEIRD_CHARS = set(string.punctuation)

# Additional Unicode punctuation and symbols
WEIRD_CHARS.update(
    "،؛؟ـ"                  # Arabic punctuation
    "…–—−‐‒"                # Ellipses and dashes
    "“”‘’„‟‹›«»"            # Quotation marks
    "•·∙⋅‧※⁎⁕"              # Bullets and dots
    "×÷±≈≠≤≥≪≫∞√∑∏∫∂∆∇"    # Mathematical symbols
    "°©®™℠§¶†‡"             # Miscellaneous symbols
    "€£¥₹¢₽₩₪₫₱₦₴₡₲₵"      # Currency symbols
    "→←↑↓↔↕⇒⇐⇔"             # Arrows
    "★☆♥♡✓✔✗✘"             # Decorative symbols
    "\u00A0\u200B\u200C\u200D\u200E\u200F\uFEFF"
    # Non-breaking space, zero-width characters, BOM
)


def find_weird_chars(value):
    """Return suspicious characters found in a string or list of strings."""
    if isinstance(value, list):
        text = " ".join(str(item) for item in value)
    elif value is None:
        text = ""
    else:
        text = str(value)

    found = {
        char
        for char in text
        if unicodedata.category(char).startswith(("P", "S", "C"))
        or char in WEIRD_CHARS
    }

    return sorted(found, key=ord)


with INPUT_JSON.open("r", encoding="utf-8") as f:
    data = json.load(f)

medications = data["medications"] if isinstance(data, dict) else data

results = []

for med in medications:
    normalized_name = med.get("normalized_key") or ""
    arabic_names = med.get("arabic_names") or []

    normalized_weird = find_weird_chars(normalized_name)
    arabic_weird = find_weird_chars(arabic_names)

    # Include the medication if either field contains suspicious characters.
    if normalized_weird or arabic_weird:
        results.append({
            "medication_id": med.get("medication_id"),
            "canonical_name": med.get("canonical_name"),
            "normalized_key": normalized_name,
            "normalized_key_weird_chars": normalized_weird,
            "normalized_key_weird_chars_repr": [
                repr(char) for char in normalized_weird
            ],
            "arabic_names": arabic_names,
            "arabic_names_weird_chars": arabic_weird,
            "arabic_names_weird_chars_repr": [
                repr(char) for char in arabic_weird
            ],
        })

with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Medications checked: {len(medications)}")
print(f"Medications with suspicious characters: {len(results)}")
print(f"Saved to: {OUTPUT_JSON.resolve()}")

for med in results:
    print(
        f"\nID: {med['medication_id']}"
        f"\nCanonical: {med['canonical_name']}"
        f"\nNormalized key: {med['normalized_key']}"
        f"\nNormalized suspicious chars: "
        f"{med['normalized_key_weird_chars_repr']}"
        f"\nArabic names: {med['arabic_names']}"
        f"\nArabic suspicious chars: "
        f"{med['arabic_names_weird_chars_repr']}"
    )