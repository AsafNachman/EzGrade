"""Syllabus parsing: raw syllabus (PDF / text file / pasted text) -> list[SyllabusTopic].

OWNER: <teammate name>
STATUS: stub. ``read_source_text`` works; ``parse_syllabus`` returns dummy topics.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from src.models import SyllabusTopic
from src.validator import sanitize_text

logger = logging.getLogger(__name__)

TEXT_FILE_SUFFIXES = {".txt", ".md"}
_MAX_PATH_LENGTH = 260

DUMMY_TOPICS: tuple[SyllabusTopic, ...] = (
    SyllabusTopic(
        id="t1",
        title="Data Structures",
        subtopics=["Arrays and linked lists", "Stacks and queues", "Hash tables"],
    ),
    SyllabusTopic(
        id="t2",
        title="Algorithms",
        subtopics=["Sorting", "Binary search", "Recursion"],
    ),
    SyllabusTopic(
        id="t3",
        title="Complexity Analysis",
        subtopics=["Big-O notation", "Amortized analysis"],
    ),
)


def _as_existing_file(value: str) -> Path | None:
    if "\n" in value or len(value) > _MAX_PATH_LENGTH:
        return None
    try:
        path = Path(value)
        return path if path.is_file() else None
    except (OSError, ValueError):
        return None


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_source_text(file_path_or_text: str) -> str:
    """Return raw text from a .pdf/.txt/.md path, or the input itself if it is not a file path."""
    path = _as_existing_file(file_path_or_text)
    if path is None:
        return file_path_or_text
    if path.suffix.lower() == ".pdf":
        return extract_pdf_text(path)
    if path.suffix.lower() in TEXT_FILE_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported syllabus file type: {path.suffix!r}")


def parse_syllabus(file_path_or_text: str) -> list[SyllabusTopic]:
    """Parse a syllabus into topics.

    TODO(parser): replace the dummy return with real extraction:
      1. Build a prompt asking Gemini to extract topics + subtopics from ``text``.
      2. Call ``ask_gemini_json(prompt, fallback_data={"topics": [...]},
         response_model=<a model wrapping list[SyllabusTopic]>)``.
      3. Re-number topic ids deterministically ("t1", "t2", ...) so they are stable.
      4. Optionally add a regex/heading-based offline parser (e.g. "Week 1: ...")
         to use when the LLM is unavailable, before falling back to DUMMY_TOPICS.
    """
    if not file_path_or_text or not file_path_or_text.strip():
        raise ValueError("Syllabus input is empty.")

    text = sanitize_text(read_source_text(file_path_or_text))
    logger.info("Parsed %d characters of syllabus text (dummy topics returned).", len(text))

    return [topic.model_copy(deep=True) for topic in DUMMY_TOPICS]
