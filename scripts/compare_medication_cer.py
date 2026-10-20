
import json
import re
import unicodedata
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_FILE = Path(r"D:\projects\medical_ocr\gemini_filtered_samples.json")
GEMINI_CLEANED_FILE = Path(
    r"D:\projects\medical_ocr\gemini_filtered_samples_cleaned.json"
)
GEMMA_FILE = Path(r"D:\projects\medical_ocr\gemma_pre_gt_bg_filtered_samples.json")
GEMMA_CLEANED_FILE = Path(
    r"D:\projects\medical_ocr\gemma_pre_gt_bg_filtered_samples_cleaned.json"
)

# Reports will be saved beside this script.
OUTPUT_DIR = Path(__file__).resolve().parent
JSON_OUTPUT = OUTPUT_DIR / "prescriptions_got_worse.json"
TXT_OUTPUT = OUTPUT_DIR / "prescriptions_got_worse.txt"

FIELDS = [
    "name",
    "strength",
    "dose",
    "frequency",
    "duration",
    "instructions",
]

ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


# ============================================================
# NORMALIZATION AND CER
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.translate(ARABIC_DIGITS).casefold()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def levenshtein_distance(reference, hypothesis):
    """Calculate character-level edit distance."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))

    for i, ref_char in enumerate(reference, start=1):
        current = [i]

        for j, hyp_char in enumerate(hypothesis, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (
                ref_char != hyp_char
            )

            current.append(
                min(insertion, deletion, substitution)
            )

        previous = current

    return previous[-1]


def compare_field(reference, hypothesis):
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    distance = levenshtein_distance(ref, hyp)

    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "reference_chars": len(ref),
        "edit_distance": distance,
        "cer": distance / len(ref) if ref else None,
        "exact_match": ref == hyp,
        "reference_empty": not ref,
        "hypothesis_empty": not hyp,
    }


def compare_medication(reference_med, hypothesis_med):
    field_results = {}
    total_edits = 0
    total_reference_chars = 0

    for field in FIELDS:
        result = compare_field(
            reference_med.get(field),
            hypothesis_med.get(field),
        )

        field_results[field] = result
        total_edits += result["edit_distance"]
        total_reference_chars += result["reference_chars"]

    overall_cer = (
        total_edits / total_reference_chars
        if total_reference_chars
        else None
    )

    return {
        "fields": field_results,
        "total_edit_distance": total_edits,
        "total_reference_chars": total_reference_chars,
        "overall_cer": overall_cer,
        "exact_match": all(
            result["exact_match"]
            for result in field_results.values()
        ),
    }


# ============================================================
# LOAD AND MATCH PRESCRIPTIONS
# ============================================================

def load_prescriptions(path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = next(
            (
                data[key]
                for key in ("prescriptions", "data", "results")
                if isinstance(data.get(key), list)
            ),
            None,
        )

        if records is None:
            records = [data]
    else:
        raise ValueError(f"Unsupported JSON structure: {path}")

    return records


def record_key(record, index):
    filename = record.get("filename")

    if filename:
        return str(filename)

    return f"__record_index_{index}"


def index_records(records, dataset_name):
    indexed = {}

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            continue

        key = record_key(record, i)

        if key in indexed:
            raise ValueError(
                f"Duplicate prescription filename {key!r} "
                f"in {dataset_name}"
            )

        indexed[key] = record

    return indexed


def get_medications(record):
    if not isinstance(record, dict):
        return []

    medications = record.get("medications", [])
    return medications if isinstance(medications, list) else []


def rounded(value):
    return round(value, 4) if value is not None else None


# ============================================================
# MAIN COMPARISON
# ============================================================

def main():
    input_files = {
        "gemini": GEMINI_FILE,
        "gemini_cleaned": GEMINI_CLEANED_FILE,
        "gemma": GEMMA_FILE,
        "gemma_cleaned": GEMMA_CLEANED_FILE,
    }

    datasets = {
        name: index_records(
            load_prescriptions(path),
            name,
        )
        for name, path in input_files.items()
    }

    all_keys = set().union(
        *(set(dataset.keys()) for dataset in datasets.values())
    )

    worsened_prescriptions = []
    missing_prescriptions = []
    missing_medications = []

    for filename in sorted(all_keys):
        records = {
            name: dataset.get(filename)
            for name, dataset in datasets.items()
        }

        absent_records = [
            name for name, record in records.items()
            if record is None
        ]

        if absent_records:
            missing_prescriptions.append({
                "filename": filename,
                "missing_from": absent_records,
            })
            continue

        medications = {
            name: get_medications(record)
            for name, record in records.items()
        }

        max_count = max(
            len(items) for items in medications.values()
        )

        prescription_worsened_meds = []

        for i in range(max_count):
            meds = {
                name: (
                    items[i]
                    if i < len(items) and isinstance(items[i], dict)
                    else None
                )
                for name, items in medications.items()
            }

            absent_meds = [
                name for name, med in meds.items()
                if med is None
            ]

            if absent_meds:
                missing_medications.append({
                    "filename": filename,
                    "medication_number": i + 1,
                    "missing_from": absent_meds,
                    "available_names": {
                        name: med.get("name")
                        for name, med in meds.items()
                        if med is not None
                    },
                })
                continue

            # BEFORE: Gemini vs Gemma.
            before = compare_medication(
                meds["gemini"],
                meds["gemma"],
            )

            # AFTER: cleaned Gemini vs cleaned Gemma.
            after = compare_medication(
                meds["gemini_cleaned"],
                meds["gemma_cleaned"],
            )

            before_cer = before["overall_cer"]
            after_cer = after["overall_cer"]

            if before_cer is None or after_cer is None:
                continue

            cer_delta = after_cer - before_cer

            # Only keep medications whose CER increased.
            if cer_delta <= 0:
                continue

            worsened_fields = []

            for field in FIELDS:
                before_field = before["fields"][field]
                after_field = after["fields"][field]

                before_field_cer = before_field["cer"]
                after_field_cer = after_field["cer"]

                if (
                    before_field_cer is not None
                    and after_field_cer is not None
                    and after_field_cer > before_field_cer
                ):
                    worsened_fields.append({
                        "field": field,
                        "cer_before": rounded(before_field_cer),
                        "cer_after": rounded(after_field_cer),
                        "before": {
                            "gemini": meds["gemini"].get(field),
                            "gemma": meds["gemma"].get(field),
                        },
                        "after": {
                            "gemini": meds["gemini_cleaned"].get(field),
                            "gemma": meds["gemma_cleaned"].get(field),
                        },
                    })

            prescription_worsened_meds.append({
                "medication_number": i + 1,
                "cer_before": rounded(before_cer),
                "cer_after": rounded(after_cer),
                "cer_increase": rounded(cer_delta),
                "worsened_fields": worsened_fields,

                # Complete medication values before and after cleaning.
                "medication_before": {
                    "gemini": meds["gemini"],
                    "gemma": meds["gemma"],
                },
                "medication_after": {
                    "gemini": meds["gemini_cleaned"],
                    "gemma": meds["gemma_cleaned"],
                },
            })

        # Include a prescription only if at least one medication worsened.
        if prescription_worsened_meds:
            worsened_prescriptions.append({
                "filename": filename,
                "worsened_medications": prescription_worsened_meds,
            })

    # ========================================================
    # WRITE SIMPLE JSON REPORT
    # ========================================================

    json_report = {
        "description": (
            "Prescriptions where the cleaned Gemini-vs-Gemma "
            "CER increased compared with the original pair."
        ),
        "cer_before": "Original Gemini compared with original Gemma",
        "cer_after": "Cleaned Gemini compared with cleaned Gemma",
        "prescription_count": len(worsened_prescriptions),
        "medication_count": sum(
            len(item["worsened_medications"])
            for item in worsened_prescriptions
        ),
        "prescriptions": worsened_prescriptions,
        "missing_prescriptions": missing_prescriptions,
        "unmatched_medications": missing_medications,
    }

    with JSON_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)

    # ========================================================
    # WRITE HUMAN-READABLE TEXT REPORT
    # ========================================================

    with TXT_OUTPUT.open("w", encoding="utf-8") as f:
        f.write("PRESCRIPTIONS THAT GOT WORSE AFTER CLEANING\n")
        f.write("=" * 60 + "\n")
        f.write(
            f"Prescriptions affected: {len(worsened_prescriptions)}\n"
        )
        f.write(
            "CER before = original Gemini vs original Gemma\n"
        )
        f.write(
            "CER after  = cleaned Gemini vs cleaned Gemma\n\n"
        )

        if not worsened_prescriptions:
            f.write("No prescriptions got worse.\n")

        for prescription in worsened_prescriptions:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Prescription: {prescription['filename']}\n")

            for med in prescription["worsened_medications"]:
                f.write("\n" + "-" * 50 + "\n")
                f.write(
                    f"Medication number: {med['medication_number']}\n"
                )
                f.write(f"CER before: {med['cer_before']}\n")
                f.write(f"CER after:  {med['cer_after']}\n")
                f.write(f"CER increase: {med['cer_increase']}\n")

                f.write("\nWorsened fields:\n")

                if not med["worsened_fields"]:
                    f.write(
                        "  No individual field CER increased; "
                        "the combined CER increased.\n"
                    )

                for item in med["worsened_fields"]:
                    f.write(f"\n  Field: {item['field']}\n")
                    f.write(
                        f"    CER: {item['cer_before']} "
                        f"-> {item['cer_after']}\n"
                    )
                    f.write(
                        f"    Before - Gemini: "
                        f"{item['before']['gemini']}\n"
                    )
                    f.write(
                        f"    Before - Gemma:  "
                        f"{item['before']['gemma']}\n"
                    )
                    f.write(
                        f"    After  - Gemini: "
                        f"{item['after']['gemini']}\n"
                    )
                    f.write(
                        f"    After  - Gemma:  "
                        f"{item['after']['gemma']}\n"
                    )

    print("\nComparison complete!")
    print(f"Prescriptions that got worse: {len(worsened_prescriptions)}")
    print(
        "Medications that got worse: "
        f"{json_report['medication_count']}"
    )
    print(f"JSON report: {JSON_OUTPUT}")
    print(f"Text report: {TXT_OUTPUT}")


if __name__ == "__main__":
    main()
