import json
import re
import unicodedata
from pathlib import Path
from collections import Counter
from typing import Any

BASE_DIR = Path(r"D:\projects\vocab_project")
BEFORE_FILE = BASE_DIR / "matching_medications.json"
AFTER_FILE = BASE_DIR / "matching_medications_cleaned.json"
REPORT_JSON = BASE_DIR / "matching_medications_eval_report.json"
REPORT_TXT = BASE_DIR / "matching_medications_eval_report.txt"
FIELDS = ("name", "strength", "dose", "frequency", "form", "duration", "instructions")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    return re.sub(r"\s+", " ", value).strip()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer_stats(references, predictions):
    edits = chars = empty_refs = 0
    for ref, pred in zip(references, predictions):
        ref, pred = normalize_text(ref), normalize_text(pred)
        edits += levenshtein(ref, pred)
        chars += len(ref)
        empty_refs += not bool(ref)
    cer = edits / chars if chars else None
    return {"pairs": len(references), "total_edit_distance": edits,
            "total_reference_characters": chars, "cer": cer,
            "cer_percent": cer * 100 if cer is not None else None,
            "empty_reference_count": empty_refs}


def get_filename(sample):
    return str(sample.get("filename") or sample.get("file_name") or "").strip()


def get_samples(data):
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list at the top level.")
    result = {}
    for sample in data:
        if isinstance(sample, dict) and get_filename(sample):
            result[get_filename(sample)] = sample
    return result


def get_pairs(sample):
    value = sample.get("medications", [])
    return value if isinstance(value, list) else []


def get_med(pair, model):
    value = pair.get(model, {}) if isinstance(pair, dict) else {}
    return value if isinstance(value, dict) else {}


def same(a, b):
    return normalize_text(a) == normalize_text(b)


def compare_stage(samples, stage_name):
    counts = {field: Counter() for field in FIELDS}
    refs, preds, different_names = [], [], []
    pair_count = 0
    for filename, sample in sorted(samples.items()):
        for index, pair in enumerate(get_pairs(sample)):
            gemini, gemma = get_med(pair, "gemini"), get_med(pair, "gemma")
            if not gemini and not gemma:
                continue
            pair_count += 1
            for field in FIELDS:
                a, b = gemini.get(field), gemma.get(field)
                counts[field]["identical" if same(a, b) else "different"] += 1
                counts[field]["compared"] += 1
                if a is None and b is None:
                    counts[field]["both_missing"] += 1
                elif a is None or b is None:
                    counts[field]["one_missing"] += 1
            ref, pred = gemini.get("name") or "", gemma.get("name") or ""
            refs.append(ref)
            preds.append(pred)
            if not same(ref, pred):
                different_names.append({"filename": filename, "medication_index": index,
                                        "gemini_name": ref, "gemma_name": pred,
                                        "name_edit_distance": levenshtein(normalize_text(ref), normalize_text(pred))})
    field_stats = {}
    for field, count in counts.items():
        total, identical = count["compared"], count["identical"]
        field_stats[field] = {"compared": total, "identical": identical,
                              "different": count["different"],
                              "identical_percent": identical * 100 / total if total else None,
                              "both_missing": count["both_missing"], "one_missing": count["one_missing"]}
    return {"stage": stage_name, "sample_count": len(samples), "medication_pair_count": pair_count,
            "fields": field_stats, "name_cer_gemini_reference_gemma_prediction": cer_stats(refs, preds),
            "different_name_pairs": different_names}


def compare_changes(before, after):
    changes = {model: {field: Counter() for field in FIELDS} for model in ("gemini", "gemma")}
    examples, missing_samples, mismatches = [], [], []
    matched_pairs = 0
    for filename, old_sample in sorted(before.items()):
        new_sample = after.get(filename)
        if new_sample is None:
            missing_samples.append(filename)
            continue
        old_pairs, new_pairs = get_pairs(old_sample), get_pairs(new_sample)
        if len(old_pairs) != len(new_pairs):
            mismatches.append({"filename": filename, "before_pair_count": len(old_pairs), "after_pair_count": len(new_pairs)})
        for index, old_pair in enumerate(old_pairs):
            if index >= len(new_pairs):
                continue
            matched_pairs += 1
            for model in ("gemini", "gemma"):
                old_med, new_med = get_med(old_pair, model), get_med(new_pairs[index], model)
                for field in FIELDS:
                    old, new = old_med.get(field), new_med.get(field)
                    changed = not same(old, new)
                    changes[model][field]["changed" if changed else "unchanged"] += 1
                    if changed:
                        examples.append({"filename": filename, "medication_index": index,
                                         "model": model, "field": field, "before": old, "after": new})
    summary = {}
    for model, fields in changes.items():
        summary[model] = {}
        for field, count in fields.items():
            total = count["changed"] + count["unchanged"]
            summary[model][field] = {"compared": total, "changed": count["changed"],
                                     "unchanged": count["unchanged"],
                                     "changed_percent": count["changed"] * 100 / total if total else None}
    return {"matched_sample_count": len(set(before) & set(after)), "matched_medication_pair_count": matched_pairs,
            "missing_after_samples": missing_samples, "pair_count_mismatches": mismatches,
            "field_changes": summary, "changed_examples": examples}


def pct(value):
    return "N/A" if value is None else f"{value:.2f}%"


def write_text_report(report):
    before, after = report["before"], report["after"]
    changes = report["before_after_changes"]
    lines = ["MEDICATION CLEANUP EVALUATION", "=" * 72,
             f"Before file: {BEFORE_FILE}", f"After file:  {AFTER_FILE}", "",
             "DATA ALIGNMENT", "-" * 72,
             f"Samples before: {before['sample_count']}", f"Samples after:  {after['sample_count']}",
             f"Medication pairs before: {before['medication_pair_count']}",
             f"Medication pairs after:  {after['medication_pair_count']}",
             f"Matched medication pairs for before/after comparison: {changes['matched_medication_pair_count']}",
             f"Samples missing after: {len(changes['missing_after_samples'])}",
             f"Samples with pair-count mismatch: {len(changes['pair_count_mismatches'])}", ""]

    def append_stage(stage, heading):
        lines.extend([heading, "-" * 72, f"Medication pairs: {stage['medication_pair_count']}"])
        cer = stage["name_cer_gemini_reference_gemma_prediction"]
        lines.append(f"Name CER (Gemini reference -> Gemma prediction): {pct(cer['cer_percent'])} "
                     f"({cer['total_edit_distance']} edits / {cer['total_reference_characters']} reference characters)")
        lines.append("Field equality:")
        for field, stats in stage["fields"].items():
            lines.append(f"  {field:14} identical={stats['identical']:5}/{stats['compared']:5} "
                         f"({pct(stats['identical_percent'])}), different={stats['different']}, "
                         f"both_missing={stats['both_missing']}, one_missing={stats['one_missing']}")
        lines.append("")

    append_stage(before, "GEMINI VS GEMMA: BEFORE CLEANUP")
    append_stage(after, "GEMINI VS GEMMA: AFTER CLEANUP")
    bcer = before["name_cer_gemini_reference_gemma_prediction"]["cer_percent"]
    acer = after["name_cer_gemini_reference_gemma_prediction"]["cer_percent"]
    lines.extend(["CER CHANGE", "-" * 72, f"Before CER: {pct(bcer)}", f"After CER:  {pct(acer)}"])
    if bcer is not None and acer is not None:
        lines.append(f"Delta (after - before): {acer - bcer:+.2f} percentage points")
        lines.append("Negative delta means better Gemini/Gemma name agreement; positive means worse.")
    lines.extend(["", "FIELDS CHANGED BY CLEANUP", "-" * 72])
    for model, fields in changes["field_changes"].items():
        lines.append(model.upper() + ":")
        for field, stats in fields.items():
            lines.append(f"  {field:14} changed={stats['changed']:5}/{stats['compared']:5} "
                         f"({pct(stats['changed_percent'])}), unchanged={stats['unchanged']}")
    lines.extend(["", "DIFFERENT NAME PAIRS BEFORE CLEANUP", "-" * 72])
    for item in before["different_name_pairs"]:
        lines.append(f"{item['filename']} | medication #{item['medication_index']} | Gemini={item['gemini_name']!r} | "
                     f"Gemma={item['gemma_name']!r} | edit_distance={item['name_edit_distance']}")
    lines.extend(["", "DIFFERENT NAME PAIRS AFTER CLEANUP", "-" * 72])
    for item in after["different_name_pairs"]:
        lines.append(f"{item['filename']} | medication #{item['medication_index']} | Gemini={item['gemini_name']!r} | "
                     f"Gemma={item['gemma_name']!r} | edit_distance={item['name_edit_distance']}")
    lines.extend(["", "ALL FIELD CHANGES CAUSED BY CLEANUP", "-" * 72])
    for item in changes["changed_examples"]:
        lines.append(f"{item['filename']} | medication #{item['medication_index']} | {item['model']}.{item['field']}: "
                     f"{item['before']!r} -> {item['after']!r}")
    lines.extend(["", "INTERPRETATION NOTES", "-" * 72,
                  "CER is agreement CER: Gemini medication names are treated as references and Gemma names as predictions; it is not ground-truth OCR CER.",
                  "The input file is already filtered to pairs whose medication names were close, so the results apply only to this subset.",
                  "Equality uses Unicode NFKC, case-folding, Arabic-Indic digit conversion, and whitespace normalization."])
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    for path in (BEFORE_FILE, AFTER_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
    with BEFORE_FILE.open("r", encoding="utf-8") as f:
        before_samples = get_samples(json.load(f))
    with AFTER_FILE.open("r", encoding="utf-8") as f:
        after_samples = get_samples(json.load(f))

    before_stage = compare_stage(before_samples, "before_cleanup")
    after_stage = compare_stage(after_samples, "after_cleanup")
    changes = compare_changes(before_samples, after_samples)
    report = {
        "inputs": {"before": str(BEFORE_FILE), "after": str(AFTER_FILE)},
        "metric_definition": {
            "cer": "Micro CER = total character-level Levenshtein edits / total normalized Gemini reference-name characters.",
            "cer_comparison": "Gemini medication name is the reference; Gemma medication name is the prediction.",
            "exact_equality": "Unicode NFKC + casefold + Arabic-Indic digit conversion + whitespace normalization.",
            "warning": "Agreement CER between Gemini and Gemma, not ground-truth OCR CER."
        },
        "before": before_stage, "after": after_stage, "before_after_changes": changes
    }
    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_text_report(report)
    bcer = before_stage["name_cer_gemini_reference_gemma_prediction"]["cer_percent"]
    acer = after_stage["name_cer_gemini_reference_gemma_prediction"]["cer_percent"]
    print(f"Before samples: {before_stage['sample_count']}")
    print(f"After samples:  {after_stage['sample_count']}")
    print(f"Before medication pairs: {before_stage['medication_pair_count']}")
    print(f"After medication pairs:  {after_stage['medication_pair_count']}")
    print(f"Name CER before (Gemini ref, Gemma pred): {pct(bcer)}")
    print(f"Name CER after  (Gemini ref, Gemma pred): {pct(acer)}")
    if bcer is not None and acer is not None:
        print(f"CER delta (after - before): {acer - bcer:+.2f} percentage points")
    print(f"Detailed JSON report: {REPORT_JSON}")
    print(f"Readable text report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
