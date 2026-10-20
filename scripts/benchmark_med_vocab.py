
import chromadb
import re
import unicodedata
import json
from pathlib import Path
from copy import deepcopy

from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, List

INPUT_FILE = Path(
    r"D:\projects\vocab_project\matching_medications.json"
)

OUTPUT_FILE = INPUT_FILE.with_name(
    f"{INPUT_FILE.stem}_cleaned.json"
)

DEFAULT_INPUT = Path(
    r"D:\projects\vocab_project\data\med_vocab\medication_vocab.json"
)
DEFAULT_DB_DIR = Path(
    r"D:\projects\vocab_project\data\med_vocab\chroma_medications"
)
DEFAULT_REPORT = Path(
    r"D:\projects\vocab_project\data\med_vocab\real_prescription_retrieval_test.json"
)

DEFAULT_FREQ_DB_DIR = Path(
    r"D:\projects\vocab_project\data\freq_vocab\chroma_frequencies"
)

db_path = str(DEFAULT_DB_DIR)

forms_path = Path(
    r"D:\projects\vocab_project\data\forms_vocab\forms_vocab.json"
)
doses_path = Path(
    r"D:\projects\vocab_project\data\dose_vocab\doses_vocab.json"
)
durations_path = Path(
    r"D:\projects\vocab_project\data\durations_vocab\durations_vocab.json"
)

with open(forms_path, "r", encoding="utf-8") as f:
    forms = json.load(f)

with open(doses_path, "r", encoding="utf-8") as f:
    doses = json.load(f)

with open(durations_path, "r", encoding="utf-8") as f:
    durations = json.load(f)


FREQUENCY_COLLECTION_NAME = "frequency_vocabulary"
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "medication_vocabulary"
RANDOM_SEED = 42
STRENGTH_RE = re.compile(
    r"(?ix)(?<![\w])"
    r"(?:\d+(?:[.,]\d+)?(?:\s*/\s*\d+(?:[.,]\d+)?)+\s*"
    r"(?:mcg|µg|μg|ug|mg|gm|gr|kg|g|ml|l|iu|i\.u\.?|meq|mmol|%))"
    r"|"
    r"(?:\d+(?:[.,]\d+)?\s*"
    r"(?:mcg|µg|μg|ug|mg|gm|gr|kg|g|ml|l|iu|i\.u\.?|meq|mmol|%)"
    r"(?:\s*/\s*\d+(?:[.,]\d+)?\s*"
    r"(?:mcg|µg|μg|ug|mg|gm|gr|kg|g|ml|l|iu|i\.u\.?|meq|mmol|%)?)?)"
    r"(?![\w])"
)

form_terms = forms.get("ar", []) + forms.get("en", [])
form_patterns = forms.get("re", [])
form_terms_pattern = [
    re.escape(term)
    for term in form_terms
    if term.strip()
]

FORM_RE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        sorted(
            form_terms_pattern + form_patterns,
            key=len,
            reverse=True,
        )
    )
    + r")(?!\w)",
    flags=re.IGNORECASE,
)


ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)
NON_ALNUM = re.compile(r"[^\w\u0600-\u06FF]+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")

model = SentenceTransformer(DEFAULT_MODEL)
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name=COLLECTION_NAME)
freq_client = chromadb.PersistentClient(
    path=str(DEFAULT_FREQ_DB_DIR)
)
freq_collection = freq_client.get_collection(
    name=FREQUENCY_COLLECTION_NAME
)


def detect_language(text: str) -> str:
    arabic_count = len(
        re.findall(
            r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]",
            text,
        )
    )
    english_count = len(re.findall(r"[a-zA-Z]", text))

    if arabic_count == 0 and english_count == 0:
        return "unknown"

    return "ar" if arabic_count > english_count else "en"


def contain_numbers(a: str) -> bool:
    return bool(re.search(r"\d", a))


def normalize(text: Any) -> str:
    s = str(text or "").casefold().translate(
        str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    )
    s = unicodedata.normalize("NFKC", s)
    s = ARABIC_DIACRITICS.sub("", s)
    s = (
        s.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
    )
    s = NON_ALNUM.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip()


def fix_character_spacing(text: str) -> str:
    """Join OCR text made of individually spaced letters, without joining normal words."""
    text = str(text or "").strip()
    return re.sub(
        r"(?<!\S)(?:[^\W\d_]\s+){2,}[^\W\d_](?!\S)",
        lambda match: re.sub(r"\s+", "", match.group(0)),
        text,
        flags=re.UNICODE,
    )


def parse_medication_doc(doc: str) -> dict:
    fields = [
        "Name variants",
        "Arabic names",
    ]

    result = {}

    for field in fields:
        pattern = (
            rf"(?:^|\n){re.escape(field)}:\s*(.*?)"
            rf"(?=\n(?:{'|'.join(map(re.escape, fields))}):|$)"
        )
        match = re.search(pattern, doc, flags=re.DOTALL)

        if match:
            value = match.group(1).strip()

            if field == "Medication name":
                result[field] = value
            else:
                result[field] = [
                    item.strip()
                    for item in value.split(",")
                    if item.strip()
                ]
        else:
            result[field] = [] if field != "Medication name" else ""

    return result


def _get_best_ratio_of(query: str, candidates: List[str]):
    if not candidates or not query:
        return None, -1, 0.0

    normalized_query = normalize(query)
    best_cand = None
    best_index = -1
    best_ratio = 0.0

    for i, candidate in enumerate(candidates):
        normalized_candidate = normalize(candidate)
        if not normalized_candidate:
            continue

        ratio = SequenceMatcher(
            None, normalized_query, normalized_candidate
        ).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_cand = candidate
            best_index = i

    return best_cand, best_index, best_ratio

def _get_form(query: str, min_acc_thresh=0.84):
    normalized_query = normalize(query)
    if not normalized_query:
        return normalized_query, 0.0

    lang = detect_language(normalized_query)
    forms_lang = forms["en"] if lang == "en" else forms["ar"]
    best_cand, _, best_ratio = _get_best_ratio_of(normalized_query, forms_lang)

    form = best_cand if best_cand is not None and best_ratio >= min_acc_thresh else normalized_query
    return form, best_ratio

def _get_freq(query: str, min_acc_thresh=0.84):
    raw_query = unicodedata.normalize(
        "NFKC", str(query or "").casefold()
    ).translate(
        str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    ).strip()

    normalized_query = normalize(raw_query)

    if not normalized_query:
        return normalized_query, None, None

    number = r"[0-9IlOo]+(?:[.,][0-9IlOo]+)?"
    pattern = (
        rf"\s*({number})\s*[xX×*]\s*"
        rf"({number})\s*[xX×*]\s*"
        rf"({number})\s*"
    )
    match = re.fullmatch(pattern, raw_query)

    if match:
        def fix_number(value: str) -> str:
            return value.translate(
                str.maketrans({
                    "I": "1",
                    "l": "1",
                    "O": "0",
                    "o": "0",
                })
            )

        x, y, z = map(fix_number, match.groups())
        return y, x, z

    if freq_collection.count() > 0:
        query_embedding = model.encode(
            [f"query: {normalized_query}"],
            normalize_embeddings=True,
        ).tolist()

        results = freq_collection.query(
            query_embeddings=query_embedding,
            n_results=min(10, freq_collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        candidates = []

        for doc, metadata in zip(
            results["documents"][0],
            results["metadatas"][0],
        ):
            metadata = metadata or {}

            candidate = (
                metadata.get("normalized_key")
                or metadata.get("canonical_name")
                or doc
            )

            if candidate:
                candidates.append(str(candidate).strip())

        candidates = list(dict.fromkeys(candidates))

        if candidates:
            best, _, best_score = _get_best_ratio_of(
                normalized_query,
                candidates,
            )

            if best is not None and best_score >= min_acc_thresh:
                return normalize(best), None, None

    return normalized_query, None, None

def _get_dose(query: str, min_acc_thresh=0.84):
    normalized_query = normalize(query)
    if not normalized_query:
        return normalized_query, 0.0

    number_pattern = r"(?:\d+\s*/\s*\d+|\d+(?:[.,٫]\d+)?)"
    match = re.match(rf"^\s*({number_pattern})\s*(.*?)\s*$", normalized_query)
    number = match.group(1).replace(" ", "") if match else ""
    remaining = match.group(2).strip() if match else normalized_query

    lang = detect_language(remaining)
    doses_lang = doses["en"] if lang == "en" else doses["ar"]
    best_cand, _, best_ratio = _get_best_ratio_of(remaining, doses_lang)

    if (best_cand is not None and best_ratio >= min_acc_thresh
            and detect_language(normalize(best_cand)) == lang):
        return f"{number} {best_cand}".strip() if number else best_cand, best_ratio

    return normalized_query, best_ratio

def _get_duration(query: str):
    normalized_query = normalize(query)
    return normalized_query, 0.0


def _medication_processing(input, min_acc_thresh=0.84):
    raw_name = input.get("name") or ""
    original_name = fix_character_spacing(normalize(raw_name))

    raw_strength = input.get("strength")
    raw_dose = input.get("dose")
    raw_frequency = input.get("frequency")
    raw_duration = input.get("duration")
    raw_form = input.get("form")
    raw_instructions = input.get("instructions")

    strength = normalize(raw_strength)
    form = normalize(raw_form)
    dose_text = normalize(raw_dose)
    frequency_text = normalize(raw_frequency)
    duration_text = normalize(raw_duration)
    instructions_text = normalize(raw_instructions)

    name_core = original_name
    strength_match = STRENGTH_RE.search(name_core)
    if strength_match:
        if not strength:
            strength = normalize(strength_match.group(0))

        name_core = (
            name_core[:strength_match.start()]
            + " "
            + name_core[strength_match.end():]
        )

    form_match = FORM_RE.search(name_core)
    if form_match:
        if not form:
            form, _ = _get_form(form_match.group(0))

        name_core = (
            name_core[:form_match.start()]
            + " "
            + name_core[form_match.end():]
        )

    name_core = SPACE_RE.sub(" ", name_core).strip()
    cleaned_name = name_core or original_name

    if cleaned_name and collection.count() > 0:
        query_embedding = model.encode(
            [f"query: {cleaned_name}"],
            normalize_embeddings=True,
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(20, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        candidates = []
        language = detect_language(cleaned_name)

        for doc, metadata in zip(
            results["documents"][0],
            results["metadatas"][0],
        ):
            metadata = metadata or {}
            parsed = parse_medication_doc(doc or "")

            if language == "ar":
                values = parsed.get("Arabic names", [])
            else:
                values = [
                    metadata.get("normalized_key"),
                    metadata.get("canonical_name"),
                ]
                values.extend(parsed.get("Name variants", []))

            for value in values:
                if isinstance(value, str) and normalize(value):
                    candidates.append(value)

        candidates = list(dict.fromkeys(candidates))

        if candidates:
            best, _, best_score = _get_best_ratio_of(
                cleaned_name,
                candidates,
            )

            if best is not None:
                normalized_query = normalize(cleaned_name)
                normalized_best = normalize(best)

                scores = sorted(
                    {
                        normalize(candidate): SequenceMatcher(
                            None,
                            normalized_query,
                            normalize(candidate),
                        ).ratio()
                        for candidate in candidates
                        if normalize(candidate)
                    }.values(),
                    reverse=True,
                )

                second_score = scores[1] if len(scores) > 1 else 0.0

                if (
                    best_score >= min_acc_thresh
                    and (
                        normalized_query == normalized_best
                        or best_score - second_score >= 0.04
                    )
                    and detect_language(normalized_best) == language
                ):
                    cleaned_name = normalized_best

    frequency, parsed_dose, parsed_duration = _get_freq(
        raw_frequency or ""
    )

    if parsed_dose is not None:
        dose_text = parsed_dose

    if parsed_duration is not None:
        duration_text = parsed_duration

    if not dose_text:
        dose_text, _ = _get_dose(dose_text)

    if not duration_text:
        duration_text, _ = _get_duration(duration_text)

    return {
        "name": cleaned_name,
        "strength": strength,
        "dose": dose_text,
        "frequency": frequency,
        "form": form,
        "duration": duration_text,
        "instructions": instructions_text,
    }



def process_prescriptions(data):
    processed_data = deepcopy(data)

    # matching_medications.json stores each sample as:
    # {"filename": ..., "medications": [{"gemini": {...}, "gemma": {...}}]}
    # Process both model outputs while preserving the pair/sample structure.
    if isinstance(processed_data, list) and all(
        isinstance(sample, dict) and "medications" in sample
        for sample in processed_data
    ):
        total = processed = skipped = 0
        errors = []

        for sample_index, sample in enumerate(processed_data):
            medication_pairs = sample.get("medications", [])
            if not isinstance(medication_pairs, list):
                continue

            for pair_index, pair in enumerate(medication_pairs):
                if not isinstance(pair, dict):
                    continue

                for model_key in ("gemini", "gemma"):
                    medication = pair.get(model_key)
                    if not isinstance(medication, dict):
                        continue

                    total += 1
                    try:
                        cleaned = _medication_processing(medication)
                        if not isinstance(cleaned, dict):
                            raise TypeError(
                                "_medication_processing() must return a dict"
                            )
                        medication.update(cleaned)
                        processed += 1
                    except Exception as exc:
                        errors.append({
                            "sample_index": sample_index,
                            "pair_index": pair_index,
                            "model": model_key,
                            "name": medication.get("name"),
                            "error": str(exc),
                        })

        return processed_data, total, processed, skipped, errors



    if isinstance(processed_data, list):

        prescriptions = processed_data

    elif isinstance(processed_data, dict):
        if isinstance(processed_data.get("medications"), list):
            prescriptions = [processed_data]

        else:

            prescriptions = next(
                (

                    value
                    for value in processed_data.values()
                    if isinstance(value, list)
                    and any(
                        isinstance(item, dict)
                        and "medications" in item
                        for item in value
                    )
                ),
                None,
            )

            if prescriptions is None:
                raise ValueError(

                    "Could not find a list of prescriptions."

                )

    else:
        raise ValueError("Expected a JSON object or list.")

    total = 0
    processed = 0
    skipped = 0
    errors = []

    for prescription_index, prescription in enumerate(prescriptions):
        if not isinstance(prescription, dict):
            continue

        medications = prescription.get("medications", [])
        if not isinstance(medications, list):
            continue

        for medication_index, medication in enumerate(medications):
            if not isinstance(medication, dict):
                continue

            total += 1
            name = medication.get("name")
            try:
                cleaned = _medication_processing(medication)
                if not isinstance(cleaned, dict):
                    raise TypeError(
                        "_medication_processing() must return a dict"

                    )
                medication.update(cleaned)
                processed += 1

            except Exception as exc:
                errors.append({
                    "prescription_index": prescription_index,
                    "medication_index": medication_index,
                    "name": name,
                    "error": str(exc),
                })

    return processed_data, total, processed, skipped, errors

def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Input JSON not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    result, total, processed, skipped, errors = (
        process_prescriptions(data)
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total medications: {total}")
    print(f"Processed: {processed}")
    print(f"Skipped (missing name): {skipped}")
    print(f"Failed: {len(errors)}")

    if errors:
        error_file = OUTPUT_FILE.with_name(
            f"{OUTPUT_FILE.stem}_errors.json"
        )

        with error_file.open("w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

        print(f"Errors saved to: {error_file}")


if __name__ == "__main__":
    main()


# sample = {
#     "name": "SR lone or",
#     "strength": None,
#     "dose": "قرص",
#     "frequency": "الساعة ١٢ مساء",
#     "duration": "لمدة ٢٥ يوم",
#     "instructions": None
# }

# processed_input = _medication_processing(sample)
# with open("res.json", "w", encoding="utf-8") as f:
#     json.dump(processed_input, f, ensure_ascii=False, indent=2)