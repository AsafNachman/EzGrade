"""JSON persistence for exams/results and Markdown export.

Files live under ``<project>/data`` (override with the EZGRADE_DATA_DIR env var):
    data/exams/<exam_id>.json
    data/results/<exam_id>.json
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.models import Exam, ExamResult

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPTION_LABELS = "ABCD"

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

ModelT = TypeVar("ModelT", bound=BaseModel)


def get_data_dir(data_dir: Path | str | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    return Path(os.environ.get("EZGRADE_DATA_DIR") or PROJECT_ROOT / "data")


def _record_path(kind: str, record_id: str, data_dir: Path | str | None) -> Path:
    # Whitelisting the id prevents path traversal such as "../../etc/passwd".
    if not _SAFE_ID_RE.fullmatch(record_id):
        raise ValueError(f"Unsafe id for a file name: {record_id!r}")
    return get_data_dir(data_dir) / kind / f"{record_id}.json"


def _atomic_write(path: Path, text: str) -> None:
    """Write to a temp file then rename, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _save(model: BaseModel, kind: str, record_id: str, data_dir: Path | str | None) -> Path:
    path = _record_path(kind, record_id, data_dir)
    _atomic_write(path, model.model_dump_json(indent=2))
    return path


def _load(model_cls: type[ModelT], kind: str, record_id: str, data_dir: Path | str | None) -> ModelT | None:
    path = _record_path(kind, record_id, data_dir)
    if not path.is_file():
        return None
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


def save_exam(exam: Exam, data_dir: Path | str | None = None) -> Path:
    return _save(exam, "exams", exam.id, data_dir)


def load_exam(exam_id: str, data_dir: Path | str | None = None) -> Exam | None:
    return _load(Exam, "exams", exam_id, data_dir)


def save_result(result: ExamResult, data_dir: Path | str | None = None) -> Path:
    return _save(result, "results", result.exam_id, data_dir)


def load_result(exam_id: str, data_dir: Path | str | None = None) -> ExamResult | None:
    return _load(ExamResult, "results", exam_id, data_dir)


def list_results(data_dir: Path | str | None = None) -> list[ExamResult]:
    """All saved results, newest first. Corrupt files are skipped and logged."""
    results_dir = get_data_dir(data_dir) / "results"
    if not results_dir.is_dir():
        return []
    files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for path in files:
        try:
            results.append(ExamResult.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValidationError) as exc:
            logger.warning("Skipping unreadable result file %s: %s", path, exc)
    return results


def export_exam_to_markdown(exam: Exam, include_answers: bool = False) -> str:
    topic_titles = {topic.id: topic.title for topic in exam.topics}
    lines = [
        "# EzGrade Practice Exam",
        "",
        f"**Exam ID:** `{exam.id}`  ",
        f"**Difficulty:** {exam.difficulty.capitalize()}  ",
        f"**Questions:** {len(exam.questions)}",
        "",
        "## Topics",
        "",
        *(f"- {topic.title}" for topic in exam.topics),
        "",
        "## Questions",
        "",
    ]
    for number, question in enumerate(exam.questions, start=1):
        lines.append(f"### {number}. {question.question_text}")
        lines.append(f"*Topic: {topic_titles.get(question.topic_id, question.topic_id)}*")
        lines.append("")
        lines.extend(f"- **{OPTION_LABELS[i]})** {option}" for i, option in enumerate(question.options))
        lines.append("")

    if include_answers:
        lines += ["## Answer Key", ""]
        for number, question in enumerate(exam.questions, start=1):
            label = OPTION_LABELS[question.correct_index]
            lines.append(f"{number}. **{label}** - {question.explanation}")
        lines.append("")

    return "\n".join(lines)
