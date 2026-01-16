import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


def pdf_to_markdown(
    file_path_in: str,
    file_path_out: str,
    *,
    model: str = "gpt-5.2",
    api_key_env: str = "OPENAI_API_KEY",
    max_pages: Optional[int] = None,
) -> str:
    """
    Convert a PDF to canonical Markdown:
      1) Extract raw text using ONLY pypdf.
      2) Ask OpenAI to clean/structure it into Markdown.
      3) Write output to a .md file whose name matches the input PDF stem.

    Args:
        file_path_in: Path to input PDF.
        file_path_out: Either:
            - a directory path (output file will be created inside it), OR
            - a full file path ending in .md
        model: OpenAI model name.
        api_key_env: Environment variable that stores the OpenAI API key.
        max_pages: If set, only process first N pages (useful for very large PDFs).

    Returns:
        The full output path of the written Markdown file.
    """
    load_dotenv()

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Set {api_key_env} in environment or .env")

    in_path = Path(file_path_in)
    if not in_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {in_path}")
    if in_path.suffix.lower() != ".pdf":
        raise ValueError(f"Input must be a PDF: {in_path}")

    out_path = Path(file_path_out)

    # If file_path_out is a directory, write <pdf_stem>.md into it.
    # If it's a file path (endswith .md), use it as-is.
    if out_path.suffix.lower() == ".md":
        final_out_path = out_path
        final_out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        final_out_path = out_path / f"{in_path.stem}.md"

    # --- Extract text with pypdf only ---
    reader = PdfReader(str(in_path))
    page_texts = []
    total_pages = len(reader.pages)
    limit = min(total_pages, max_pages) if max_pages else total_pages

    for i in range(limit):
        text = reader.pages[i].extract_text() or ""
        page_texts.append(text.strip())

    raw_text = "\n\n".join(t for t in page_texts if t)

    if not raw_text.strip():
        raise RuntimeError("No extractable text found in PDF (may be scanned or image-only).")

    instruction = f"""You are producing a canonical Markdown document for downstream
chunking and embedding.

Convert this PDF-extracted text into clean Markdown with headings, lists, and tables.
Preserve the original language. Do not invent content.

=== PDF TEXT (pypdf) ===
<<<
{raw_text}
>>>
"""

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": instruction}],
            }
        ],
    )

    markdown_output = response.output_text.strip()

    final_out_path.write_text(markdown_output + "\n", encoding="utf-8")
    return str(final_out_path)
