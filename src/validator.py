"""Input sanitization and exam quality validation.

OWNER: <teammate name>
The structural checks below already work; the TODOs mark the "smart"
quality checks that still need implementing.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

from pydantic import ValidationError

from src.models import Exam

MAX_INPUT_CHARS = 50_000

# C0 control characters except tab (\x09), newline (\x0a) and carriage return (\x0d), plus DEL.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_WS_RE = re.compile(r"[ \t]+")
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")


def sanitize_text(text: str, max_length: int = MAX_INPUT_CHARS) -> str:
    """Normalize untrusted text (syllabus uploads, user input) before it reaches the LLM."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _HORIZONTAL_WS_RE.sub(" ", text)
    text = _EXCESS_NEWLINES_RE.sub("\n\n", text)
    # TODO(validator): add prompt-injection heuristics, e.g. flag/strip lines like
    #   "ignore previous instructions" before the text is embedded in a prompt.
    return text.strip()[:max_length]


def validate_exam(exam: Exam | dict[str, Any]) -> tuple[bool, list[str]]:
    """Return ``(is_valid, errors)``. Never raises for bad input."""
    if not isinstance(exam, Exam):
        try:
            exam = Exam.model_validate(exam)
        except ValidationError as exc:
            return False, [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]

    errors: list[str] = []
    topic_ids = set(exam.topics_by_id())

    id_counts = Counter(question.id for question in exam.questions)
    errors.extend(f"Duplicate question id '{qid}'." for qid, count in id_counts.items() if count > 1)

    text_counts = Counter(question.question_text.casefold() for question in exam.questions)
    errors.extend(f"Duplicate question text: '{text}'." for text, count in text_counts.items() if count > 1)

    for question in exam.questions:
        if question.topic_id not in topic_ids:
            errors.append(f"Question '{question.id}' references unknown topic '{question.topic_id}'.")
        if question.difficulty != exam.difficulty:
            errors.append(
                f"Question '{question.id}' has difficulty '{question.difficulty}', "
                f"exam is '{exam.difficulty}'."
            )

    # TODO(validator): reject ambiguous options ("All of the above", "None of the above").
    # TODO(validator): enforce length bounds on question_text / options / explanation.
    # TODO(validator): check topic coverage balance (no topic should dominate the exam).
    # TODO(validator): optional second LLM pass that independently answers each question
    #   and flags any whose answer disagrees with correct_index.

    return not errors, errors
