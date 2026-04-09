"""
phase1/pdf_parser.py
────────────────────
Extracts clean text from medical PDFs.

Tries pdfplumber first (better for clinical docs with tables),
falls back to PyPDF2. Handles scanned PDFs gracefully.

Usage:
    from phase1.pdf_parser import extract_documents_from_dir
    docs = extract_documents_from_dir("./data/kb_pdfs")
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:

    source_path: str
    filename: str
    text: str
    num_pages: int
    file_size_kb: float
    parse_method: str  # "pdfplumber" | "pypdf2" | "plaintext"
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"ParsedDocument(file='{self.filename}', "
            f"pages={self.num_pages}, "
            f"chars={len(self.text):,}, "
            f"method='{self.parse_method}')"
        )


def _clean_text(text: str) -> str:

    # Normalize unicode ligatures (common in PDFs)
    replacements = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are purely numbers (page numbers)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove hyphenation at line breaks (e.g., "diagno-\nsis" → "diagnosis")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Collapse excessive spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def _parse_with_pdfplumber(pdf_path: str) -> tuple[str, int]:
    """Extract text page-by-page using pdfplumber."""
    import pdfplumber

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if page_text:
                pages_text.append(page_text)

    return "\n\n".join(pages_text), num_pages


def _parse_with_pypdf2(pdf_path: str) -> tuple[str, int]:
    """Fallback: extract text using PyPDF2."""
    from PyPDF2 import PdfReader

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    pages_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    return "\n\n".join(pages_text), num_pages


def parse_pdf(pdf_path: str) -> Optional[ParsedDocument]:

    path = Path(pdf_path)
    if not path.exists():
        logger.error(f"File not found: {pdf_path}")
        return None

    file_size_kb = path.stat().st_size / 1024
    text = ""
    num_pages = 0
    parse_method = "unknown"

    # --- Attempt 1: pdfplumber ---
    try:
        text, num_pages = _parse_with_pdfplumber(pdf_path)
        parse_method = "pdfplumber"
        logger.debug(f"pdfplumber OK: {path.name} ({num_pages} pages)")
    except Exception as e:
        logger.warning(f"pdfplumber failed for {path.name}: {e}. Trying PyPDF2...")

        # --- Attempt 2: PyPDF2 ---
        try:
            text, num_pages = _parse_with_pypdf2(pdf_path)
            parse_method = "pypdf2"
            logger.debug(f"PyPDF2 OK: {path.name} ({num_pages} pages)")
        except Exception as e2:
            logger.error(f"Both parsers failed for {path.name}: {e2}")
            return None

    if len(text.strip()) < 50:
        logger.warning(
            f"Very little text extracted from {path.name} "
            f"({len(text)} chars) — may be a scanned PDF. "
            f"Consider OCR preprocessing."
        )

    text = _clean_text(text)

    return ParsedDocument(
        source_path=str(path.resolve()),
        filename=path.name,
        text=text,
        num_pages=num_pages,
        file_size_kb=round(file_size_kb, 2),
        parse_method=parse_method,
        metadata={
            "source": str(path.resolve()),
            "filename": path.name,
            "num_pages": num_pages,
            "file_size_kb": round(file_size_kb, 2),
        },
    )


def parse_text_file(txt_path: str) -> Optional[ParsedDocument]:
    """Parse a plain .txt file as a document."""
    path = Path(txt_path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        text = _clean_text(text)
        return ParsedDocument(
            source_path=str(path.resolve()),
            filename=path.name,
            text=text,
            num_pages=1,
            file_size_kb=round(path.stat().st_size / 1024, 2),
            parse_method="plaintext",
            metadata={
                "source": str(path.resolve()),
                "filename": path.name,
            },
        )
    except Exception as e:
        logger.error(f"Failed to read text file {path.name}: {e}")
        return None


def extract_documents_from_dir(
    data_dir: str,
    extensions: tuple = (".pdf", ".txt"),
) -> list[ParsedDocument]:
    """
    Scan a directory for supported files and parse all of them.

    Args:
        data_dir: Path to folder containing medical documents
        extensions: File extensions to include

    Returns:
        List of successfully parsed ParsedDocument objects
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"KB data directory not found: {data_dir}")

    files = [f for f in data_path.rglob("*") if f.suffix.lower() in extensions]

    if not files:
        logger.warning(f"No {extensions} files found in {data_dir}")
        return []

    logger.info(f"Found {len(files)} files to parse in {data_dir}")

    documents = []
    for file_path in sorted(files):
        logger.info(f"  Parsing: {file_path.name}")
        if file_path.suffix.lower() == ".pdf":
            doc = parse_pdf(str(file_path))
        else:
            doc = parse_text_file(str(file_path))

        if doc and len(doc.text) > 50:
            documents.append(doc)
            logger.info(
                f"    ✓ {doc.filename}: {len(doc.text):,} chars, {doc.num_pages} pages"
            )
        else:
            logger.warning(f"    ✗ {file_path.name}: skipped (empty or failed)")

    logger.info(f"Successfully parsed {len(documents)}/{len(files)} documents")
    return documents

