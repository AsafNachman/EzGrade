import pytest

from src.grader import calculate_percentage, grade_exam, is_weakness
from src.models import Exam, ExamResult, ExamSubmission, SyllabusTopic


def _submit(exam: Exam, answers: dict[str, int]) -> ExamResult:
    return grade_exam(exam, ExamSubmission(exam_id=exam.id, user_answers=answers))


def test_mismatched_exam_id_raises(sample_exam):
    with pytest.raises(ValueError):
        grade_exam(sample_exam, ExamSubmission(exam_id="other", user_answers={}))


def test_returns_valid_result(sample_exam):
    result = _submit(sample_exam, {})
    assert isinstance(result, ExamResult)
    assert result.exam_id == sample_exam.id


@pytest.mark.parametrize(("percentage", "expected"), [(0.0, True), (69.99, True), (70.0, False), (100.0, False)])
def test_weakness_threshold(percentage, expected):
    assert is_weakness(percentage) is expected


def test_calculate_percentage_handles_zero_total():
    assert calculate_percentage(0, 0) == 0.0
    assert calculate_percentage(1, 3) == 33.33


def test_all_correct_scores_100(sample_exam):
    answers = {q.id: q.correct_index for q in sample_exam.questions}
    result = _submit(sample_exam, answers)
    assert result.total_score == 100.0
    assert not result.weaknesses


def test_no_answers_scores_0_and_every_topic_is_weak(sample_exam):
    result = _submit(sample_exam, {})
    assert result.total_score == 0.0
    assert len(result.weaknesses) == len(result.topic_breakdown) == 2


def test_wrong_topic_is_flagged_as_weakness(sample_exam):
    answers = {q.id: q.correct_index for q in sample_exam.questions if q.topic_id != "t2"}
    result = _submit(sample_exam, answers)
    assert [score.topic_title for score in result.weaknesses] == ["Algorithms"]


def test_partial_score_per_topic(sample_exam):
    answers = {q.id: q.correct_index for q in sample_exam.questions if q.topic_id == "t1"}
    result = _submit(sample_exam, answers)
    assert result.total_score == 50.0
    scores = {score.topic_title: (score.correct, score.total, score.percentage) for score in result.topic_breakdown}
    assert scores == {"Data Structures": (2, 2, 100.0), "Algorithms": (0, 2, 0.0)}


def test_feedback_messages(sample_exam):
    q1, q2, q3, _ = sample_exam.questions
    wrong_answer = (q2.correct_index + 1) % len(q2.options)
    result = _submit(sample_exam, {q1.id: q1.correct_index, q2.id: wrong_answer})

    assert result.feedback_per_question[q1.id] == "Correct!"
    assert result.feedback_per_question[q2.id].startswith("Incorrect")
    assert q2.options[q2.correct_index] in result.feedback_per_question[q2.id]
    assert result.feedback_per_question[q3.id].startswith("Not answered")
    assert set(result.feedback_per_question) == {q.id for q in sample_exam.questions}


def test_answers_for_unknown_questions_are_ignored(sample_exam):
    answers = {q.id: q.correct_index for q in sample_exam.questions}
    answers["q999"] = 0
    result = _submit(sample_exam, answers)
    assert result.total_score == 100.0
    assert "q999" not in result.feedback_per_question


def test_topics_without_questions_are_skipped(sample_exam):
    exam = Exam(
        id=sample_exam.id,
        topics=[*sample_exam.topics, SyllabusTopic(id="t3", title="Graphs", subtopics=[])],
        difficulty=sample_exam.difficulty,
        questions=sample_exam.questions,
    )
    result = _submit(exam, {})
    assert "Graphs" not in [score.topic_title for score in result.topic_breakdown]
