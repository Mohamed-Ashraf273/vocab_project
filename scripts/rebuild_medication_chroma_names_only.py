import argparse
import json
import re
import shutil
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_INPUT = Path(
    r"D:\projects\medical_ocr\reports\med_vocab\medication_vocab.json"
)
DEFAULT_DB_DIR = Path(
    r"D:\projects\medical_ocr\reports\med_vocab\chroma_medications"
)

COLLECTION_NAME = "medication_vocabulary"
STAGING_COLLECTION = "medication_vocabulary_staging"
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
BATCH_SIZE = 128

# These patterns invalidate a name, not automatically the entire medication.
BAD_NAME_PATTERNS = ("illegible", "?", "_")

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
NON_ALNUM = re.compile(r"[^\w\u0600-\u06FF]+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


def normalize(value: Any) -> str:
    text = str(value or "")
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(ARABIC_DIGITS)
    text = ARABIC_DIACRITICS.sub("", text)
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
    )
    text = NON_ALNUM.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def is_bad_name(value: Any) -> bool:
    raw = str(value or "").casefold().strip()
    return not raw or any(pattern in raw for pattern in BAD_NAME_PATTERNS)


def get_strings(value: Any) -> list[str]:
    """Read strings from a string, list, or list of alias dictionaries."""
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    output: list[str] = []
    seen: set[str] = set()

    for item in values:
        if isinstance(item, dict):
            item = item.get("value") or item.get("name") or item.get("canonical_name")
        if item is None:
            continue
        text = str(item).strip()
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def get_collection_names(client) -> set[str]:
    """Support Chroma versions that return names or Collection objects."""
    return {
        item.name if hasattr(item, "name") else str(item)
        for item in client.list_collections()
    }


def load_vocabulary(path: Path) -> tuple[list[dict], Counter]:
    if not path.is_file():
        raise FileNotFoundError(f"Vocabulary not found: {path}")

    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = next(
            (
                data[key]
                for key in ("medications", "medication_vocabulary", "vocabulary", "items", "data")
                if isinstance(data.get(key), list)
            ),
            None,
        )
        if records is None:
            raise ValueError("Could not find the medication list in JSON.")
    else:
        raise ValueError("Expected a JSON list or dictionary.")

    medications: list[dict] = []
    used_ids: set[str] = set()
    stats = Counter()

    for index, raw in enumerate(records, start=1):
        if not isinstance(raw, dict):
            stats["invalid_records"] += 1
            continue

        canonical = str(raw.get("canonical_name") or raw.get("name") or "").strip()
        if not canonical:
            stats["missing_canonical_name"] += 1
            continue
        if is_bad_name(canonical) or not normalize(canonical):
            stats["bad_canonical_records_skipped"] += 1
            continue

        medication_id = str(raw.get("medication_id") or f"med_{index:06d}").strip()
        if medication_id in used_ids:
            raise ValueError(f"Duplicate medication ID: {medication_id}")
        used_ids.add(medication_id)

        raw_normalized = str(raw.get("normalized_key") or "").strip()
        normalized_name = normalize(raw_normalized) if not is_bad_name(raw_normalized) else ""
        if not normalized_name:
            normalized_name = normalize(canonical)

        # Filter bad variants individually; keep the medication record.
        variants = []
        for variant in get_strings(raw.get("name_variants")):
            if is_bad_name(variant) or not normalize(variant):
                stats["bad_variants_skipped"] += 1
                continue
            variants.append(variant)

        arabic_names = []
        for name in get_strings(raw.get("arabic_names")):
            if is_bad_name(name) or not normalize(name):
                stats["bad_arabic_names_skipped"] += 1
                continue
            arabic_names.append(name)

        medications.append(
            {
                "id": medication_id,
                "normalized_name": normalized_name,
                "canonical_name": canonical,
                "arabic_names": arabic_names,
                "variants": variants,
            }
        )

    if not medications:
        raise ValueError("No valid medication records found.")
    return medications, stats


def build_medication_documents(medications: list[dict]):
    """Create exactly ONE Chroma document per medication record."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    stats = Counter()

    for med in medications:
        # Four name fields are included in each document. E5 passage prefix is required.
        arabic_text = " | ".join(med["arabic_names"]) or "none"
        variants_text = " | ".join(med["variants"]) or "none"
        document = (
            "passage: Medication entry. "
            f"Normalized name: {med['normalized_name']}. "
            f"Canonical name: {med['canonical_name']}. "
            f"Arabic names: {arabic_text}. "
            f"Name variants: {variants_text}."
        )

        ids.append(med["id"])
        documents.append(document)
        metadatas.append(
            {
                "medication_id": med["id"],
                "normalized_name": med["normalized_name"],
                "canonical_name": med["canonical_name"],
                # Chroma metadata values must be scalar; store arrays as JSON strings.
                "arabic_names_json": json.dumps(med["arabic_names"], ensure_ascii=False),
                "variants_json": json.dumps(med["variants"], ensure_ascii=False),
            }
        )
        stats["variants_indexed"] += len(med["variants"])
        stats["arabic_names_indexed"] += len(med["arabic_names"])

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document IDs detected.")
    return ids, documents, metadatas, stats


def backup_database(db_dir: Path):
    if not db_dir.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = db_dir.with_name(f"{db_dir.name}_backup_{timestamp}")
    shutil.copytree(db_dir, backup_dir)
    print(f"Database backup created: {backup_dir}")
    return backup_dir


def rebuild_database(input_path: Path, db_dir: Path, model_name: str):
    medications, load_stats = load_vocabulary(input_path)
    ids, documents, metadatas, build_stats = build_medication_documents(medications)

    print(f"Input vocabulary: {input_path}")
    print(f"Valid medication records: {len(medications):,}")
    print(f"Chroma documents to create: {len(documents):,} (one per medication)")
    print(f"Bad variants skipped: {load_stats['bad_variants_skipped']:,}")
    print(f"Bad Arabic names skipped: {load_stats['bad_arabic_names_skipped']:,}")
    print(f"Bad canonical records skipped: {load_stats['bad_canonical_records_skipped']:,}")

    db_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_database(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=str(db_dir))

    existing = get_collection_names(client)
    if STAGING_COLLECTION in existing:
        client.delete_collection(STAGING_COLLECTION)

    staging = client.create_collection(
        name=STAGING_COLLECTION,
        metadata={
            "hnsw:space": "cosine",
            "description": "Staging: one document per medication with normalized, canonical, Arabic, and variant names.",
        },
    )

    installed_successfully = False
    try:
        for start in range(0, len(documents), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(documents))
            embeddings = model.encode(
                documents[start:end],
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).tolist()
            staging.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings,
            )
            print(f"Staged {end:,}/{len(documents):,} medication entries")

        if staging.count() != len(documents):
            raise RuntimeError(
                f"Staging verification failed: expected {len(documents)}, got {staging.count()}"
            )

        # Replace the live collection only after staging has completed and verified.
        if COLLECTION_NAME in get_collection_names(client):
            client.delete_collection(COLLECTION_NAME)

        live = client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "description": "One document per medication; metadata includes normalized and canonical names, Arabic names, and variants.",
            },
        )

        for start in range(0, len(documents), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(documents))
            batch = staging.get(
                ids=ids[start:end],
                include=["documents", "metadatas", "embeddings"],
            )
            live.add(
                ids=batch["ids"],
                documents=batch["documents"],
                metadatas=batch["metadatas"],
                embeddings=batch["embeddings"],
            )
            print(f"Installed {end:,}/{len(documents):,} medication entries")

        if live.count() != len(documents):
            raise RuntimeError(
                f"Live verification failed: expected {len(documents)}, got {live.count()}. "
                "The database backup remains available."
            )

        installed_successfully = True
        print("\nRebuild completed successfully.")
        print(f"Database: {db_dir}")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Medication entries: {live.count():,}")
        print(f"Variants stored: {build_stats['variants_indexed']:,}")
        print(f"Arabic names stored: {build_stats['arabic_names_indexed']:,}")
        print("Each Chroma entry contains these four fields:")
        print("  - normalized_name")
        print("  - canonical_name")
        print("  - arabic_names_json")
        print("  - variants_json")
    finally:
        # Remove staging only after the live collection has been verified.
        if installed_successfully and STAGING_COLLECTION in get_collection_names(client):
            client.delete_collection(STAGING_COLLECTION)


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild medication ChromaDB with one entry per medication."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    rebuild_database(args.input, args.db_dir, args.model)


if __name__ == "__main__":
    main()
