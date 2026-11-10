import pytest

from src.grader import calculate_percentage, grade_exam, is_weakness
from src.models import ExamResult, ExamSubmission, SyllabusTopic


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


def test_all_correct_scores_100(sample_exam):
    answers = {q.id: q.correct_index for q in sample_exam.questions}
    result = grade_exam(sample_exam, ExamSubmission(exam_id=sample_exam.id, user_answers=answers))
    assert result.total_score == 100.0
    assert not result.weaknesses
    assert all(feedback == "Correct!" for feedback in result.feedback_per_question.values())


def test_wrong_topic_is_flagged_as_weakness(sample_exam):
    answers = {q.id: q.correct_index for q in sample_exam.questions if q.topic_id != "t2"}
    result = grade_exam(sample_exam, ExamSubmission(exam_id=sample_exam.id, user_answers=answers))
    assert [score.topic_title for score in result.weaknesses] == ["Algorithms"]


def test_missing_answers_are_wrong_and_unknown_answers_are_ignored(sample_exam):
    submission = ExamSubmission(
        exam_id=sample_exam.id,
        user_answers={"q1": sample_exam.questions[0].correct_index, "not-in-exam": 0},
    )

    result = grade_exam(sample_exam, submission)

    assert result.total_score == 25.0
    assert result.feedback_per_question["q1"] == "Correct!"
    for question in sample_exam.questions[1:]:
        assert result.feedback_per_question[question.id] == (
            f"Incorrect - correct answer: {question.options[question.correct_index]}. "
            f"{question.explanation}"
        )


def test_topics_without_questions_are_omitted(sample_exam, sample_topics):
    exam = sample_exam.model_copy(
        update={"topics": [*sample_topics, SyllabusTopic(id="t3", title="Unused")]}
    )

    result = grade_exam(exam, ExamSubmission(exam_id=exam.id, user_answers={}))

    assert [score.topic_title for score in result.topic_breakdown] == ["Data Structures", "Algorithms"]
