import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SplitDoc:
    page_content: str
    metadata: dict[str, str]


HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)$")


def split_markdown_by_headers(markdown_text: str) -> list[SplitDoc]:
    sections: list[SplitDoc] = []
    headers: dict[str, str | None] = {"Header 1": None, "Header 2": None, "Header 3": None}
    current_lines: list[str] = []

    def flush_section() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            metadata = {k: v for k, v in headers.items() if v}
            sections.append(SplitDoc(page_content=content, metadata=metadata))
        current_lines.clear()

    for line in markdown_text.splitlines():
        match = HEADER_RE.match(line)
        if match:
            flush_section()
            level = len(match.group(1))
            title = match.group(2).strip()
            if level == 1:
                headers["Header 1"] = title
                headers["Header 2"] = None
                headers["Header 3"] = None
            elif level == 2:
                headers["Header 2"] = title
                headers["Header 3"] = None
            else:
                headers["Header 3"] = title
            continue
        current_lines.append(line)

    flush_section()
    return sections


def process_markdown_to_jsonl(
    file_path: str,
    output_file_path: str,
    guideline_name: str,
    # Manual Configuration Fields
    class_name: str,
    version_id: str | None = None,
    published_year: int | None = None,
    language: str = "et",
) -> None:
    """
    Parses a markdown file into Weaviate-ready JSONL with integer IDs
    and combined search text for context-aware retrieval.
    """
    md_path = Path(file_path)

    # 1. Read Markdown File
    markdown_text = md_path.read_text(encoding="utf-8")

    # 2. Split Text by Markdown headers
    splits = split_markdown_by_headers(markdown_text)

    # 3. Write to JSONL
    # We use 'w' to overwrite or 'a' to append. 'w' is safer for fresh runs.
    with open(output_file_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(splits):

            # A. Build Breadcrumbs (Hierarchy)
            # Split sections store headers in metadata
            md = doc.metadata
            breadcrumbs_list = [md.get("Header 1"), md.get("Header 2"), md.get("Header 3")]
            # Filter out None values and join with " > "
            breadcrumbs = " > ".join([h for h in breadcrumbs_list if h])

            # B. Prepare Content Fields
            clean_text = doc.page_content

            # The "Context-Enriched" field for the Embedding Model
            # Combines hierarchy + content
            combined_search_text = f"{breadcrumbs}\n{clean_text}" if breadcrumbs else clean_text

            # C. Construct the Object
            properties = {
                "chunk_id": i,  # Simple Integer (0, 1, 2...)
                "source": guideline_name,
                "language": language,
                "breadcrumbs": breadcrumbs,
                "text": clean_text,  # Clean text for LLM/Reading
                "search_text": combined_search_text,  # Enriched text for Vectorizing
            }
            if version_id:
                properties["version_id"] = version_id
            if published_year is not None:
                properties["year"] = published_year

            record = {"class": class_name, "properties": properties}

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Successfully converted {len(splits)} chunks.")
    print(f"📂 Output saved to: {output_file_path}")


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


def find_markdown_sources(data_root: Path) -> list[Path]:
    markdown_paths: list[Path] = []
    for guideline_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        normalized_dir = guideline_dir / "20_normalized_md"
        canonical_dir = guideline_dir / "10_canonical_md"

        preferred_dir = normalized_dir if any(normalized_dir.glob("*.md")) else canonical_dir
        markdown_paths.extend(sorted(preferred_dir.glob("*.md")))
    return markdown_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split guideline markdown into chunks and save as JSONL."
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
        "--language",
        default="et",
        help="Language code for chunk metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    if not data_root.exists():
        raise SystemExit(
            f"Data root not found: {data_root}. "
            "Use --data-root to point at the guideline data folder."
        )

    markdown_paths = find_markdown_sources(data_root)
    if not markdown_paths:
        print("No markdown files found to split.")
        return

    for md_path in markdown_paths:
        guideline_dir = md_path.parents[1]
        class_name = guideline_dir.name
        output_dir = guideline_dir / "30_chunks"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{md_path.stem}.jsonl"

        process_markdown_to_jsonl(
            file_path=str(md_path),
            output_file_path=str(output_path),
            guideline_name=md_path.stem,
            class_name=class_name,
            language=args.language,
        )


if __name__ == "__main__":
    main()
