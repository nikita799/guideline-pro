import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def ensure_pydantic_v1() -> None:
    try:
        import pydantic.v1 as pydantic_v1
    except Exception:
        return

    sys.modules["pydantic"] = pydantic_v1


ensure_pydantic_v1()

try:
    import chromadb
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'chromadb'. Install in your venv: pip install chromadb"
    ) from exc

try:
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
except ImportError as exc:
    raise SystemExit(
        "Missing chromadb OpenAI embedding function. Ensure chromadb is installed."
    ) from exc

DEFAULT_MODEL = "text-embedding-3-large"

COLLECTION_NAME_RE = re.compile(r"[^a-z0-9_-]+")


def safe_collection_name(name: str) -> str:
    slug = name.lower()
    slug = COLLECTION_NAME_RE.sub("-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = re.sub(r"^[^a-z0-9]+", "", slug)
    slug = re.sub(r"[^a-z0-9]+$", "", slug)

    if not slug:
        slug = "guideline"

    if len(slug) < 3:
        slug = f"{slug}-doc"

    if len(slug) > 63:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:7]
        base = slug[:55].rstrip("-_")
        if not base:
            base = "guideline"
        slug = f"{base}-{digest}"

    return slug


def resolve_data_root(value: str | None) -> Path:
    if value:
        return Path(value)

    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "data",
        script_dir / "data",
        Path.cwd() / "learning_splitting" / "data",
        script_dir / "learning_splitting" / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def iter_jsonl_files(data_root: Path) -> list[Path]:
    jsonl_paths: list[Path] = []
    for guideline_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        chunk_dir = guideline_dir / "30_chunks"
        jsonl_paths.extend(sorted(chunk_dir.glob("*.jsonl")))
    return jsonl_paths


def build_id(class_name: str, file_stem: str, chunk_id: object, index: int) -> str:
    try:
        chunk_num = int(chunk_id)
    except (TypeError, ValueError):
        chunk_num = index
    return f"{class_name}::{file_stem}::chunk_{chunk_num:06d}"


def embed_batch(embedding_fn, texts: list[str]) -> list[list[float]]:
    return embedding_fn(texts)


def get_existing_ids(collection) -> set[str]:
    result = collection.get(include=[])
    return set(result.get("ids", []))


def write_to_collection(collection, ids, embeddings, metadatas, documents) -> None:
    if hasattr(collection, "upsert"):
        collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )
    else:
        collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )


def create_chroma_client(path: str):
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=path)

    try:
        from chromadb.config import Settings
    except Exception as exc:
        raise SystemExit("Failed to import chromadb Settings.") from exc

    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=path,
        anonymized_telemetry=False,
    )
    return chromadb.Client(settings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed guideline chunks and store in a local Chroma DB."
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Root folder containing guideline folders. If omitted, the script "
            "auto-detects ./data or ../learning_splitting/data."
        ),
    )
    parser.add_argument(
        "--chroma-path",
        default="chroma_db",
        help="Local path for Chroma persistent storage.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks to embed per request.",
    )
    parser.add_argument(
        "--collection",
        default="guidelines",
        help="Single collection name for all guidelines (default: guidelines).",
    )
    parser.add_argument(
        "--split-by-guideline",
        action="store_true",
        help="Store each guideline in its own collection (legacy behavior).",
    )
    parser.add_argument(
        "--input-field",
        default="search_text",
        help="Chunk field to embed (default: search_text).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed and overwrite existing ids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if load_dotenv:
        load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in environment or .env file.")

    data_root = resolve_data_root(args.data_root)
    if not data_root.exists():
        raise SystemExit(
            f"Data root not found: {data_root}. "
            "Use --data-root to point at the guideline data folder."
        )

    jsonl_paths = iter_jsonl_files(data_root)
    if not jsonl_paths:
        print("No chunk files found to embed.")
        return

    embedding_fn = OpenAIEmbeddingFunction(api_key=api_key, model_name=args.model)
    chroma_client = create_chroma_client(args.chroma_path)

    collections = {}
    existing_ids = {}
    unified_collection = safe_collection_name(args.collection)

    for jsonl_path in jsonl_paths:
        guideline_dir = jsonl_path.parents[1]
        original_name = guideline_dir.name
        guideline_slug = safe_collection_name(original_name)
        if args.split_by_guideline:
            collection_key = guideline_slug
        else:
            collection_key = unified_collection

        if collection_key not in collections:
            collections[collection_key] = chroma_client.get_or_create_collection(
                name=collection_key, embedding_function=embedding_fn
            )
            if not args.force:
                existing_ids[collection_key] = get_existing_ids(collections[collection_key])
            else:
                existing_ids[collection_key] = set()

        collection = collections[collection_key]
        seen_ids = existing_ids[collection_key]

        ids: list[str] = []
        embedding_inputs: list[str] = []
        metadatas: list[dict] = []
        documents: list[str] = []

        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue

                record = json.loads(line)
                props = record.get("properties", {})
                class_name = record.get("class") or original_name

                text = (
                    props.get(args.input_field)
                    or props.get("search_text")
                    or props.get("text")
                )
                if not text:
                    continue

                chunk_id = props.get("chunk_id")
                doc_id = build_id(class_name, jsonl_path.stem, chunk_id, index)

                if not args.force and doc_id in seen_ids:
                    continue

                ids.append(doc_id)
                embedding_inputs.append(text)
                metadatas.append(
                    {
                        "source": props.get("source"),
                        "language": props.get("language"),
                        "breadcrumbs": props.get("breadcrumbs"),
                        "chunk_id": chunk_id,
                        "file": jsonl_path.stem,
                        "class": class_name,
                        "guideline": original_name,
                        "guideline_slug": guideline_slug,
                        "collection": collection_key,
                        "search_text": props.get("search_text"),
                    }
                )
                documents.append(props.get("text") or text)

                if len(ids) >= args.batch_size:
                    embeddings = embed_batch(embedding_fn, embedding_inputs)
                    write_to_collection(collection, ids, embeddings, metadatas, documents)
                    seen_ids.update(ids)
                    ids, embedding_inputs, metadatas, documents = [], [], [], []

        if ids:
            embeddings = embed_batch(embedding_fn, embedding_inputs)
            write_to_collection(collection, ids, embeddings, metadatas, documents)
            seen_ids.update(ids)

        print(
            f"Embedded {jsonl_path.name} into collection '{collection_key}' "
            f"(from folder '{original_name}')."
        )

    if hasattr(chroma_client, "persist"):
        chroma_client.persist()


if __name__ == "__main__":
    main()
