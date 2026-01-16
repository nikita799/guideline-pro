from __future__ import annotations

import argparse
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

try:
    from helper import pdf_to_markdown
except ModuleNotFoundError:  # Fallback when run from outside this folder.
    import sys

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from helper import pdf_to_markdown


def slugify(text: str) -> str:
    # Normalize common dash variants before ASCII folding.
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    norm = unicodedata.normalize("NFKD", text)
    ascii_text = norm.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return ascii_text or "guideline"


def collect_tasks_from_structured(data_root: Path) -> list[Tuple[Path, Path]]:
    tasks: list[Tuple[Path, Path]] = []
    for pdf_path in sorted(data_root.glob("*/00_raw/*.pdf")):
        guideline_dir = pdf_path.parents[1]
        out_path = guideline_dir / "10_canonical_md" / f"{pdf_path.stem}.md"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"Skip (exists): {out_path}")
            continue
        tasks.append((pdf_path, out_path))
    return tasks


def collect_tasks_from_flat(input_dir: Path, data_root: Path) -> list[Tuple[Path, Path]]:
    tasks: list[Tuple[Path, Path]] = []
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        slug = slugify(pdf_path.stem)
        out_path = data_root / slug / "10_canonical_md" / f"{slug}.md"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"Skip (exists): {out_path}")
            continue
        tasks.append((pdf_path, out_path))
    return tasks


def convert_one(
    pdf_path: Path,
    out_path: Path,
    *,
    model: str,
    max_pages: Optional[int],
) -> str:
    return pdf_to_markdown(
        str(pdf_path),
        str(out_path),
        model=model,
        max_pages=max_pages,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert guideline PDFs to canonical Markdown in parallel."
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
        "--input-dir",
        default=None,
        help=(
            "Optional flat directory of PDFs to ingest. If omitted, scans "
            "<data-root>/*/00_raw/*.pdf."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="OpenAI model name (default: gpt-5.2).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page limit per PDF.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 4),
        help="Number of parallel workers (default: min(4, CPU count)).",
    )
    return parser.parse_args()


def resolve_data_root(value: Optional[str]) -> Path:
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


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    input_dir = Path(args.input_dir) if args.input_dir else None

    if not data_root.exists():
        raise SystemExit(
            f"Data root not found: {data_root}. "
            "Use --data-root to point at the guideline data folder."
        )

    if input_dir:
        tasks = collect_tasks_from_flat(input_dir, data_root)
    else:
        tasks = collect_tasks_from_structured(data_root)

    if not tasks:
        print("No PDFs to convert.")
        return

    errors: list[Tuple[Path, Exception]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                convert_one,
                pdf_path,
                out_path,
                model=args.model,
                max_pages=args.max_pages,
            ): (pdf_path, out_path)
            for pdf_path, out_path in tasks
        }

        for future in as_completed(future_map):
            pdf_path, out_path = future_map[future]
            try:
                result_path = future.result()
                print(f"OK: {pdf_path.name} -> {result_path}")
            except Exception as exc:
                errors.append((pdf_path, exc))
                print(f"FAILED: {pdf_path.name} -> {out_path} ({exc})")

    if errors:
        raise SystemExit(f"{len(errors)} PDFs failed to convert.")


if __name__ == "__main__":
    main()
