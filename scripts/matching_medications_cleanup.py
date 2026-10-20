

import chromadb

import re

import unicodedata

import json

import json

from pathlib import Path

from copy import deepcopy



from sentence_transformers import SentenceTransformer

from difflib import SequenceMatcher

from pathlib import Path

from typing import Any, List



INPUT_FILE = Path(

    r"D:\projects\medical_ocr\matching_medications.json"

)



OUTPUT_FILE = INPUT_FILE.with_name(

    f"{INPUT_FILE.stem}_cleaned.json"

)



DEFAULT_INPUT = Path(

    r"D:\projects\medical_ocr\reports\med_vocab\medication_vocab.json"

)

DEFAULT_DB_DIR = Path(

    r"D:\projects\medical_ocr\reports\med_vocab\chroma_medications"

)

DEFAULT_REPORT = Path(

    r"D:\projects\medical_ocr\reports\med_vocab\real_prescription_retrieval_test.json"

)



DEFAULT_FREQ_DB_DIR = Path(

    r"D:\projects\medical_ocr\reports\freq_vocab\chroma_frequencies"

)



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



FORM_RE = re.compile(
    r"(?i)\b(?:tablets?|tabs?\.?|f\.?\s*c\.?\s*tabs?\.?|"
    r"capsules?|caps?\.?|caplets?|syrup|syr\.?|syp\.?|susp(?:ension)?|"
    r"solution|soln\.?|cream|cr\.?|ointment|oint\.?|gel|drops?|"
    r"spray|supp(?:ositories)?\.?|vials?|ampoules?|amps?\.?|"
    r"injection|inj\.?|sachets?|powder|paint|lotion|patch|"
    r"قرص|اقراص|كبسوله|كبسولات|شراب|معلق|كريم|مرهم|جل|نقط|امبول|حقن)\b"
)

PACK_RE = re.compile(
    r"(?i)\b\d+\s*(?:tabs?\.?|tablets?|caps?\.?|capsules?|"
    r"sachets?|vials?|ampoules?|supp(?:ositories)?\.?)\b"
)


ARABIC_DIACRITICS = re.compile(

    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"

)

NON_ALNUM = re.compile(r"[^\w\u0600-\u06FF]+", re.UNICODE)

SPACE_RE = re.compile(r"\s+")



db_path = str(DEFAULT_DB_DIR)



forms_path = Path(

    r"D:\projects\medical_ocr\reports\forms_vocab\forms_vocab.json"

)

doses_path = Path(

    r"D:\projects\medical_ocr\reports\dose_vocab\doses_vocab.json"

)

durations_path = Path(

    r"D:\projects\medical_ocr\reports\durations_vocab\durations_vocab.json"

)



with open(forms_path, "r", encoding="utf-8") as f:

    forms = json.load(f)

FORM_ALIASES = {
    "syr": "syrup", "syp": "syrup", "syrup": "syrup",
    "susp": "suspension", "sol": "solution", "soln": "solution",
    "inj": "injection", "oint": "ointment", "lotn": "lotion",
    "supp": "suppositories", "tab": "tablet", "tabs": "tablet",
    "cap": "capsules", "caps": "capsules",
}
forms["en"] = list(dict.fromkeys([*forms.get("en", []), *FORM_ALIASES.keys()]))



with open(doses_path, "r", encoding="utf-8") as f:

    doses = json.load(f)



with open(durations_path, "r", encoding="utf-8") as f:

    durations = json.load(f)





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

                    for item in re.split(r"[,;،]", value)

                    if item.strip()

                ]

        else:

            result[field] = [] if field != "Medication name" else ""



    return result





def _get_best_ratio_of(query: str, candidates: List[str]):

    """Return the actual SequenceMatcher ratio (not an inflated score)."""

    if not candidates or not query:

        return None, -1, 0.0



    normalized_query = normalize(query)

    scored = []

    for i, candidate in enumerate(candidates):

        normalized_candidate = normalize(candidate)

        if normalized_candidate:

            score = SequenceMatcher(

                None, normalized_query, normalized_candidate

            ).ratio()

            scored.append((score, i, candidate))



    if not scored:

        return None, -1, 0.0



    scored.sort(reverse=True, key=lambda item: item[0])

    score, index, candidate = scored[0]

    return candidate, index, score





def _get_form(query: str, min_acc_thresh=0.80):

    lang = detect_language(normalize(query))

    forms_lang = forms["en"] if lang == "en" else forms["ar"]



    best_cand, _, best_ratio = _get_best_ratio_of(query, forms_lang)



    form = (

        best_cand

        if best_cand is not None and best_ratio >= min_acc_thresh

        else normalize(query)

    )



    return form, best_ratio



def _get_freq(query: str, min_acc_thresh=0.80):

    normalized_query = normalize(query)

    number = r"[0-9IlOo]+(?:[.,][0-9IlOo]+)?"

    pattern = (

        rf"\s*({number})\s*[xX×*]\s*"

        rf"({number})\s*[xX×*]\s*"

        rf"({number})\s*"

    )



    match = re.fullmatch(pattern, normalized_query)



    if match:

        x, y, z = match.groups()



        def fix_number(value: str) -> str:

            return value.translate(str.maketrans({

                "I": "1",

                "l": "1",

                "O": "0",

                "o": "0",

            }))



        x, y, z = map(fix_number, (x, y, z))



        freq = f"{y}"

        dose = x

        duration = z

        return freq, dose, duration



    query_embedding = model.encode(

        [f"query: {normalized_query}"],

        normalize_embeddings=True,

    ).tolist()



    results = freq_collection.query(

        query_embeddings=query_embedding,

        n_results=min(5, freq_collection.count()),

        include=["documents", "metadatas", "distances"],

    )



    candidates = []

    ids = []



    for metadata in results["metadatas"][0]:

        candidates.append(metadata["variant"])

        ids.append(metadata["group_id"])



    best_cand, _, best_ratio = _get_best_ratio_of(normalized_query, candidates)

    freq = best_cand if best_ratio > min_acc_thresh else normalized_query

    return freq, None, None





def _get_dose(query: str, min_acc_thresh=0.80):

    normalized_query = normalize(query)

    lang = detect_language(normalized_query)

    doses_lang = doses["en"] if lang == "en" else doses["ar"]



    number_pattern = r"(?:\d+\s*/\s*\d+|\d+(?:[.,٫]\d+)?)"

    match = re.match(

        rf"^\s*({number_pattern})\s*(.*?)\s*$",

        normalized_query,

    )



    if match:

        number = match.group(1).replace(" ", "")

        remaining = match.group(2).strip()

    else:

        number = ""

        remaining = normalized_query



    best_cand, _, best_ratio = _get_best_ratio_of(

        remaining,

        doses_lang,

    )



    if best_cand is not None and best_ratio >= min_acc_thresh:

        dose = f"{number} {best_cand}".strip() if number else best_cand

        return dose, best_ratio



    return normalized_query, best_ratio



def _get_duration(query: str, min_acc_thresh=0.80):

    normalized_query = normalize(query)

    lang = detect_language(normalized_query)

    durations_lang = durations["en"] if lang == "en" else durations["ar"]



    number_pattern = r"(?:\d+\s*/\s*\d+|\d+(?:[.,٫]\d+)?)"



    # Find a number anywhere in the duration.

    match = re.search(number_pattern, normalized_query)



    number = ""

    prefix = ""

    remaining = normalized_query



    if match:

        number = match.group(0).replace(" ", "")

        prefix = normalized_query[:match.start()].strip()

        remaining = normalized_query[match.end():].strip()



    best_cand, _, best_ratio = _get_best_ratio_of(

        remaining,

        durations_lang,

    )



    if best_cand is not None and best_ratio >= min_acc_thresh:

        duration = " ".join(

            part for part in (prefix, number, best_cand) if part

        )

        return duration, best_ratio



    return normalized_query, best_ratio



def _canonical_form(value: str):
    """Canonicalize common handwritten/typed dosage-form abbreviations."""
    key = normalize(value).strip(" .")
    aliases = {
        "syr": "syrup", "syr.": "syrup", "syp": "syrup", "syrup": "syrup",
        "susp": "suspension", "sol": "solution", "soln": "solution",
        "inj": "injection", "oint": "ointment", "lotn": "lotion",
        "supp": "suppositories", "tab": "tablet", "tabs": "tablet",
        "caps": "capsules", "cap": "capsules",
        "شراب": "syrup", "معلق": "suspension",
    }
    return aliases.get(key, value)


def _clean_candidate(value: str) -> str:
    """Avoid importing package counts such as '30 TABS' from vocabulary names."""
    value = str(value or "").strip()
    value = PACK_RE.sub(" ", value)
    value = SPACE_RE.sub(" ", value).strip(" .,-")
    return value


def _medication_processing(input, min_acc_thresh=0.80):

    """

    Minimal conservative cleanup:

    - correct the medicine core only when the actual spelling match is strong;

    - remove recognized dosage-form/package suffixes from `name` and store form separately;

    - never alter instructions;

    - preserve original clinical metadata when no confident correction exists.

    """

    raw_name = str(input.get("name") or "").strip()

    raw_strength = input.get("strength")

    raw_dose = input.get("dose")

    raw_frequency = input.get("frequency")

    raw_duration = input.get("duration")

    raw_form = input.get("form")

    raw_instructions = input.get("instructions")



    # Extract suffixes from the RAW name first so decimals such as 2.5 mg

    # are not damaged by punctuation normalization.

    raw_strength_match = STRENGTH_RE.search(raw_name)

    strength_from_name = raw_strength_match.group(0).strip() if raw_strength_match else ""

    if not raw_strength and strength_from_name:

        raw_strength = strength_from_name



    # Detect a dosage form only as a trailing suffix, not in the middle of a name.
    raw_form_match = next(
        (match for match in FORM_RE.finditer(raw_name)
         if not raw_name[match.end():].strip(" .,-;")),
        None,
    )
    form_from_name = raw_form_match.group(0).strip(" .,-;") if raw_form_match else ""
    if not raw_form and form_from_name:
        raw_form = form_from_name

    # Remove package counts before candidate lookup: "EUCARBON 30 TABS"
    # should be matched as "EUCARBON", not as a name containing "30".
    lookup_name = SPACE_RE.sub(" ", PACK_RE.sub(" ", raw_name)).strip(" .,-")
    name_for_matching = normalize(lookup_name)
    raw_core = lookup_name

    if raw_strength_match:
        strength_text = raw_strength_match.group(0)
        strength_pos = raw_core.find(strength_text)
        if strength_pos >= 0:
            raw_core = raw_core[:strength_pos] + " " + raw_core[strength_pos + len(strength_text):]

    current_form_match = next(
        (match for match in FORM_RE.finditer(raw_core)
         if not raw_core[match.end():].strip(" .,-;")),
        None,
    )
    if current_form_match:
        raw_core = raw_core[:current_form_match.start()] + " " + raw_core[current_form_match.end():]
    core = SPACE_RE.sub(" ", normalize(raw_core)).strip() or name_for_matching

    cleaned_core = core

    language = detect_language(core)

    if core and collection.count() > 0:

        query_embedding = model.encode(

            [f"query: {core}"],

            normalize_embeddings=True,

        ).tolist()

        results = collection.query(

            query_embeddings=query_embedding,

            n_results=min(20, collection.count()),

            include=["documents", "metadatas", "distances"],

        )



        # The rebuilt database has ONE document per medication. Read all four
        # fields from that medication entry; do not create one candidate per alias.
        candidates = []
        seen_candidates = set()

        for metadata in results["metadatas"][0]:
            metadata = metadata or {}
            if language == "ar":
                raw_values = metadata.get("arabic_names_json", "[]")
            else:
                raw_values = metadata.get("variants_json", "[]")

            try:
                parsed_values = json.loads(raw_values) if isinstance(raw_values, str) else []
            except (TypeError, json.JSONDecodeError):
                parsed_values = []

            values = [
                metadata.get("canonical_name", ""),
                metadata.get("normalized_name", ""),
                *parsed_values,
            ]
            # Include Arabic names too as a fallback for mixed-script OCR.
            try:
                arabic_values = json.loads(metadata.get("arabic_names_json", "[]"))
            except (TypeError, json.JSONDecodeError):
                arabic_values = []
            values.extend(arabic_values)

            for value in values:
                if not isinstance(value, str):
                    continue
                value = _clean_candidate(value)
                if not normalize(value):
                    continue
                key = normalize(value)
                if key not in seen_candidates:
                    candidates.append(value)
                    seen_candidates.add(key)

        best, best_index, best_score = _get_best_ratio_of(core, candidates)

        # Try to correct likely OCR spelling errors; don't discard the medication
        # or force a correction when the evidence is weak/ambiguous.
        if best is not None and best_score >= 0.84:
            score_by_name = {}
            for candidate in candidates:
                candidate_key = normalize(candidate)
                if not candidate_key:
                    continue
                score = SequenceMatcher(None, normalize(core), candidate_key).ratio()
                score_by_name[candidate_key] = max(score_by_name.get(candidate_key, 0.0), score)
            ranked_scores = sorted(score_by_name.values(), reverse=True)
            second_score = ranked_scores[1] if len(ranked_scores) > 1 else 0.0

            # Exact normalized matches are safe; fuzzy changes require a margin.
            if normalize(core) == normalize(best) or best_score - second_score >= 0.04:
                cleaned_core = best


    # Keep the original name exactly unless a candidate actually corrected its core.

    # If corrected, replace only the part before the first recognized form/strength,

    # preserving the original suffix and its order.

    # Keep form/packaging out of the medication name. Strength remains in the
    # original name if present; only a detected form suffix is removed.
    name_without_form = SPACE_RE.sub(" ", PACK_RE.sub(" ", raw_name)).strip(" .,-")
    final_form_match = next(
        (match for match in FORM_RE.finditer(name_without_form)
         if not name_without_form[match.end():].strip(" .,-;")),
        None,
    )
    if final_form_match:
        name_without_form = (
            name_without_form[:final_form_match.start()] + " " +
            name_without_form[final_form_match.end():]
        )
    name_without_form = SPACE_RE.sub(" ", name_without_form).strip(" .,-")

    # If the vocabulary confidently corrected the core, use it without adding
    # package counts or dosage-form suffixes from the vocabulary entry.
    cleaned_name = name_without_form
    if cleaned_core != core and core:
        tail = ""
        if raw_strength_match:
            tail = raw_strength_match.group(0).strip()
        cleaned_name = " ".join(
            part for part in (str(cleaned_core).strip(), tail) if part
        )



    # For metadata, only replace values with high-confidence vocabulary matches.

    # Otherwise keep the original string verbatim; don't destroy decimals,

    # punctuation, numeric values, or clinical wording.

    strength = raw_strength

    form = raw_form

    dose = raw_dose

    duration = raw_duration

    frequency = raw_frequency



    if isinstance(form, str) and form.strip():
        form_key = normalize(form).rstrip(" .")
        if form_key in FORM_ALIASES:
            form = FORM_ALIASES[form_key]
        else:
            candidate, score = _get_form(form, min_acc_thresh=0.90)
            if candidate and score >= 0.90:
                form = candidate



    if isinstance(dose, str) and dose.strip():

        candidate, score = _get_dose(dose, min_acc_thresh=0.90)

        if candidate and score >= 0.90:

            # Keep numbers exactly as present in the source.

            src_nums = re.findall(r"\d+(?:[.,٫]\d+)?", str(dose).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))

            out_nums = re.findall(r"\d+(?:[.,٫]\d+)?", str(candidate).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))

            if src_nums == out_nums:

                dose = candidate



    # Duration/frequency are left untouched unless the existing explicit

    # frequency parser recognizes a structured 3-part numeric pattern.

    # Never rewrite free-form clinical wording via fuzzy matching.

    if isinstance(raw_frequency, str) and raw_frequency.strip():

        # Only parse an explicit numeric x*x*x pattern. Do not fuzzy-rewrite

        # ordinary Arabic/English frequency phrases.

        numeric_pattern = r"\s*[0-9IlOo]+(?:[.,][0-9IlOo]+)?\s*[xX×*]\s*[0-9IlOo]+(?:[.,][0-9IlOo]+)?\s*[xX×*]\s*[0-9IlOo]+(?:[.,][0-9IlOo]+)?\s*"

        if re.fullmatch(numeric_pattern, raw_frequency.strip()):

            parsed_freq, parsed_dose, parsed_duration = _get_freq(raw_frequency)

            if parsed_dose is not None and parsed_duration is not None:

                # An explicit three-part pattern is the source of truth for
                # these three structured fields: dose x frequency x duration.
                frequency = parsed_freq
                dose = parsed_dose
                duration = parsed_duration



    if isinstance(duration, str) and duration.strip():

        candidate, score = _get_duration(duration, min_acc_thresh=0.90)

        if candidate and score >= 0.90:

            src_nums = re.findall(r"\d+(?:[.,٫]\d+)?", str(duration).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))

            out_nums = re.findall(r"\d+(?:[.,٫]\d+)?", str(candidate).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))

            if src_nums == out_nums:

                duration = candidate



    result = {
        "name": cleaned_name,
        "strength": strength,
        "dose": dose,
        "frequency": frequency,
        "duration": duration,
        "instructions": raw_instructions,
    }
    if isinstance(form, str) and form.strip():
        result["form"] = form
    return result


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

        # Support either a single prescription or a wrapper

        # containing a list of prescriptions.

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



            # Process every medication object. If its name is empty, metadata

            # still passes through the same processing path.

            try:

                cleaned = _medication_processing(medication)



                if not isinstance(cleaned, dict):

                    raise TypeError(

                        "_medication_processing() must return a dict"

                    )



                # Replace the medication while preserving any

                # additional fields not returned by the processor.

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