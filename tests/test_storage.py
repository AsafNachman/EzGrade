import pytest

from src.models import ExamResult, TopicScore
from src.storage import export_exam_to_markdown, list_results, load_exam, load_result, save_exam, save_result


@pytest.fixture
def sample_result(sample_exam) -> ExamResult:
    return ExamResult(
        exam_id=sample_exam.id,
        total_score=50.0,
        topic_breakdown=[TopicScore(topic_title="Algorithms", correct=1, total=2, percentage=50.0, is_weakness=True)],
        feedback_per_question={"q1": "Correct!"},
    )


def test_result_round_trip(tmp_path, sample_result):
    save_result(sample_result, data_dir=tmp_path)
    assert load_result(sample_result.exam_id, data_dir=tmp_path) == sample_result
    assert list_results(data_dir=tmp_path) == [sample_result]


def test_exam_round_trip(tmp_path, sample_exam):
    save_exam(sample_exam, data_dir=tmp_path)
    assert load_exam(sample_exam.id, data_dir=tmp_path) == sample_exam


def test_missing_result_returns_none(tmp_path):
    assert load_result("does-not-exist", data_dir=tmp_path) is None


def test_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_result("../../secrets", data_dir=tmp_path)


def test_markdown_export(sample_exam):
    markdown = export_exam_to_markdown(sample_exam)
    assert sample_exam.questions[0].question_text in markdown
    assert "Answer Key" not in markdown
    assert "Answer Key" in export_exam_to_markdown(sample_exam, include_answers=True)
