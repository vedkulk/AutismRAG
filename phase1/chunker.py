"""
phase1/chunker.py
─────────────────
Converts parsed documents into semantically coherent chunks
optimized for medical / clinical text embedding.

Strategy:
  1. Split on natural medical section boundaries first
     (headings like "Diagnosis", "Treatment", "History")
  2. Apply recursive character splitting within sections
  3. Add rich metadata to each chunk for later attribution

Usage:
    from phase1.chunker import chunk_documents
    chunks = chunk_documents(documents, chunk_size=800, chunk_overlap=150)
"""

import re
import logging
from typing import Optional
from dataclasses import dataclass


from phase1.pdf_parser import ParsedDocument

logger = logging.getLogger(__name__)


# ── Medical section headers we treat as hard split boundaries ──────────────
MEDICAL_SECTION_PATTERNS = [
    r"^(ABSTRACT|INTRODUCTION|BACKGROUND|METHODS?|RESULTS?|DISCUSSION|CONCLUSION)S?\s*$",
    r"^(DIAGNOSIS|DIFFERENTIAL DIAGNOSIS|TREATMENT|MANAGEMENT|PROGNOSIS)\s*$",
    r"^(HISTORY|EXAMINATION|INVESTIGATION|FOLLOW.UP|SUMMARY)\s*$",
    r"^(CHIEF COMPLAINT|PRESENT(?:ING)? ILLNESS|PAST MEDICAL HISTORY)\s*$",
    r"^\d+\.\s+[A-Z][A-Z\s]{5,}$",  # e.g., "1. CLINICAL FEATURES"
    r"^[A-Z][A-Z\s]{8,}:?\s*$",  # ALL CAPS headings
]

_SECTION_RE = re.compile(
    "|".join(MEDICAL_SECTION_PATTERNS),
    flags=re.MULTILINE | re.IGNORECASE,
)


@dataclass
class TextChunk:
    """A single chunk ready for embedding."""

    chunk_id: str  # Unique ID: filename_chunkN
    text: str  # The actual text content
    source_file: str  # Original filename
    source_path: str  # Full path
    chunk_index: int  # Position within the document
    total_chunks: int  # Total chunks in that document (filled later)
    section_hint: str  # Nearest section heading above this chunk
    char_start: int  # Character offset in original document
    metadata: dict  # For ChromaDB storage

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return f"Chunk({self.chunk_id}, '{preview}...')"


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split text on medical section headings.

    Returns list of (section_label, section_text) tuples.
    If no sections found, returns the full text as one section.
    """
    matches = list(_SECTION_RE.finditer(text))

    if not matches:
        return [("", text)]

    sections = []
    prev_end = 0
    prev_label = ""

    for match in matches:
        # Add text before this heading as previous section's content
        if match.start() > prev_end:
            section_text = text[prev_end:match.start()].strip()
            if section_text:
                sections.append((prev_label, section_text))

        prev_label = match.group().strip()
        prev_end = match.end()

    # Add final section
    remaining = text[prev_end:].strip()
    if remaining:
        sections.append((prev_label, remaining))

    return sections if sections else [("", text)]


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: Optional[list[str]] = None,
) -> list[str]:
    """
    Recursively split text using progressively finer separators.
    Mirrors LangChain's RecursiveCharacterTextSplitter logic.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[0]
    remaining_separators = separators[1:]

    splits = text.split(separator) if separator else list(text)

    chunks = []
    current = ""

    for split in splits:
        candidate = current + (separator if current else "") + split

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Save current chunk
            if current.strip():
                chunks.append(current.strip())

            # If single split > chunk_size, recurse with finer separator
            if len(split) > chunk_size and remaining_separators:
                sub_chunks = _recursive_split(
                    split,
                    chunk_size,
                    chunk_overlap,
                    remaining_separators,
                )
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = split

    if current.strip():
        chunks.append(current.strip())

    # Apply overlap: prefix each chunk (except first) with tail of previous
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append(prev_tail + " " + chunks[i])
        return overlapped

    return chunks


def chunk_document(
    doc: ParsedDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    min_chunk_size: int = 50,
) -> list[TextChunk]:
    """
    Chunk a single ParsedDocument into TextChunks.

    Steps:
      1. Split text on medical section boundaries
      2. Recursively split each section to chunk_size
      3. Prepend section header to each child chunk for context
      4. Attach metadata (source, section, char offset, etc.)
      5. Filter out chunks below min_chunk_size
    """
    sections = _split_into_sections(doc.text)
    raw_chunks: list[tuple[str, str, int]] = []  # (section_label, text, char_offset)

    char_offset = 0
    for section_label, section_text in sections:
        sub_chunks = _recursive_split(section_text, chunk_size, chunk_overlap)
        for sub in sub_chunks:
            # Prepend section header to each chunk for retrieval context
            if section_label and not sub.strip().upper().startswith(section_label.upper()):
                sub = f"[{section_label}]\n{sub}"
            raw_chunks.append((section_label, sub, char_offset))
        char_offset += len(section_text)

    # Filter tiny chunks
    raw_chunks = [
        (label, text, offset)
        for label, text, offset in raw_chunks
        if len(text.strip()) >= min_chunk_size
    ]

    base_name = doc.filename.replace(".pdf", "").replace(".txt", "")
    result: list[TextChunk] = []

    for i, (section_label, text, char_start) in enumerate(raw_chunks):
        chunk_id = f"{base_name}_chunk{i:04d}"

        metadata = {
            **doc.metadata,
            "chunk_id": chunk_id,
            "chunk_index": i,
            "section": section_label,
            "char_start": char_start,
            "text_length": len(text),
        }

        result.append(
            TextChunk(
                chunk_id=chunk_id,
                text=text,
                source_file=doc.filename,
                source_path=doc.source_path,
                chunk_index=i,
                total_chunks=0,  # filled below
                section_hint=section_label,
                char_start=char_start,
                metadata=metadata,
            )
        )

    # Fill in total_chunks now we know the count
    for chunk in result:
        chunk.total_chunks = len(result)
        chunk.metadata["total_chunks"] = len(result)

    return result


def chunk_documents(
    documents: list[ParsedDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    min_chunk_size: int = 50,
) -> list[TextChunk]:
    """
    Chunk a list of parsed documents.

    Args:
        documents:      Output of pdf_parser.extract_documents_from_dir()
        chunk_size:     Max characters per chunk
        chunk_overlap:  Overlap between adjacent chunks
        min_chunk_size: Discard chunks smaller than this

    Returns:
        Flat list of TextChunk objects across all documents
    """
    all_chunks: list[TextChunk] = []

    for doc in documents:
        chunks = chunk_document(doc, chunk_size, chunk_overlap, min_chunk_size)
        all_chunks.extend(chunks)
        logger.info(
            f"  Chunked '{doc.filename}': "
            f"{len(doc.text):,} chars → {len(chunks)} chunks"
        )

    logger.info(
        f"Total chunks across {len(documents)} documents: {len(all_chunks)}"
    )

    # Sanity stats
    if all_chunks:
        lengths = [len(c.text) for c in all_chunks]
        avg = sum(lengths) / len(lengths)
        logger.info(
            f"  Chunk size stats — "
            f"min: {min(lengths)}, max: {max(lengths)}, avg: {avg:.0f} chars"
        )

    return all_chunks


def print_chunk_stats(chunks: list[TextChunk]) -> None:
    """Print a readable summary of chunking results."""
    from collections import Counter

    files = Counter(c.source_file for c in chunks)
    sections = Counter(c.section_hint for c in chunks if c.section_hint)
    lengths = [len(c.text) for c in chunks]

    print("\n─── Chunk Statistics ───────────────────────────────")
    print(f"  Total chunks   : {len(chunks):,}")
    print(f"  Source files   : {len(files)}")
    print(f"  Avg chunk size : {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Min / Max      : {min(lengths)} / {max(lengths)} chars")
    print(f"\n  Chunks per file:")
    for fname, count in files.most_common():
        print(f"    {fname:<40} {count:>5} chunks")
    if sections:
        print(f"\n  Top sections detected:")
        for section, count in sections.most_common(8):
            print(f"    {section:<40} {count:>5} chunks")
    print("────────────────────────────────────────────────────\n")

