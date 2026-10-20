
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

INPUT_FILE = Path("forms_vocab.json")
OUTPUT_FILE = Path("forms_vocab_filtered.json")

ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)
NON_ALNUM = re.compile(r"[^\w\u0600-\u06FF]+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def detect_language(text: str) -> str:
    """Classify text by its Arabic and English letter counts."""
    arabic_count = len(
        re.findall(r"[\u0621-\u064A\u066E-\u06D3\u0750-\u077F\u08A0-\u08FF]", text)
    )
    english_count = len(re.findall(r"[a-zA-Z]", text))

    if arabic_count == 0 and english_count == 0:
        return "unknown"

    return "ar" if arabic_count > english_count else "en"


def normalize(text: Any) -> str:
    """Normalize Arabic/English form aliases."""
    s = str(text or "").casefold()

    # Convert Arabic-Indic digits to Western digits.
    s = s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))

    s = unicodedata.normalize("NFKC", s)
    s = ARABIC_DIACRITICS.sub("", s)

    # Normalize common Arabic letter variants.
    s = (
        s.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
    )

    s = NON_ALNUM.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip()


def main():
    # Load source vocabulary.
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise ValueError(
            "Expected forms_vocab.json to contain an 'entries' list."
        )

    # Collect and normalize retained form aliases.
    forms = set()

    for entry in data["entries"]:
        if not isinstance(entry, dict):
            continue

        if entry.get("status") != "retain_as_form_alias":
            continue

        raw_form = entry.get("raw")

        if raw_form is None:
            continue

        form = normalize(raw_form)

        if form:
            forms.add(form)

    # Classify forms by language.
    forms_json = {
        "ar": [],
        "en": [],
        "unknown": [],
    }

    for form in sorted(forms):
        language = detect_language(form)
        forms_json[language].append(form)

    # Save classified vocabulary.
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(forms_json, f, ensure_ascii=False, indent=2)

    # Print summary.
    print(f"Input entries: {len(data['entries'])}")
    print(f"Unique retained forms: {len(forms)}")
    print(f"Arabic forms: {len(forms_json['ar'])}")
    print(f"English forms: {len(forms_json['en'])}")
    print(f"Unknown forms: {len(forms_json['unknown'])}")
    print(f"Total classified: {sum(len(v) for v in forms_json.values())}")
    print(f"Output saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
