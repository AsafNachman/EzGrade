import pytest

from src.grader import calculate_percentage, grade_exam, is_weakness
from src.models import ExamResult, ExamSubmission


def test_mismatched_exam_id_raises(sample_exam):
    with pytest.raises(ValueError):
        grade_exam(sample_exam, ExamSubmission(exam_id="other", user_answers={}))


def test_returns_valid_result(sample_exam):
    result = grade_exam(sample_exam, ExamSubmission(exam_id=sample_exam.id, user_answers={}))
    assert isinstance(result, ExamResult)
    assert result.exam_id == sample_exam.id


@pytest.mark.parametrize(("percentage", "expected"), [(0.0, True), (69.99, True), (70.0, False), (100.0, False)])
def test_weakness_threshold(percentage, expected):
    assert is_weakness(percentage) is expected


def test_calculate_percentage_handles_zero_total():
    assert calculate_percentage(0, 0) == 0.0
    assert calculate_percentage(1, 3) == 33.33


@pytest.mark.xfail(reason="TODO(grader): real grading not implemented yet", strict=False)
def test_all_correct_scores_100(sample_exam):
    answers = {q.id: q.correct_index for q in sample_exam.questions}
    result = grade_exam(sample_exam, ExamSubmission(exam_id=sample_exam.id, user_answers=answers))
    assert result.total_score == 100.0
    assert not result.weaknesses


@pytest.mark.xfail(reason="TODO(grader): real grading not implemented yet", strict=False)
def test_wrong_topic_is_flagged_as_weakness(sample_exam):
    answers = {q.id: q.correct_index for q in sample_exam.questions if q.topic_id != "t2"}
    result = grade_exam(sample_exam, ExamSubmission(exam_id=sample_exam.id, user_answers=answers))
    assert [score.topic_title for score in result.weaknesses] == ["Algorithms"]
