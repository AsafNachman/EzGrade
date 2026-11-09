import pytest
from pydantic import ValidationError

from src.models import Exam, ExamSubmission, Question, TopicScore


def make_question(**overrides) -> dict:
    data = {
        "id": "q1",
        "topic_id": "t1",
        "question_text": "What is 2 + 2?",
        "options": ["1", "2", "3", "4"],
        "correct_index": 3,
        "explanation": "Basic arithmetic.",
        "difficulty": "easy",
    }
    data.update(overrides)
    return data


def test_valid_question():
    assert Question.model_validate(make_question()).correct_index == 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"options": ["a", "b", "c"]},
        {"options": ["a", "b", "c", "d", "e"]},
        {"options": ["a", "A", "c", "d"]},
        {"correct_index": 4},
        {"correct_index": -1},
        {"correct_index": "1"},  # strict mode: no str -> int coercion
        {"difficulty": "extreme"},
        {"question_text": "   "},
        {"unexpected_field": True},
    ],
)
def test_invalid_question_rejected(overrides):
    with pytest.raises(ValidationError):
        Question.model_validate(make_question(**overrides))


def test_exam_requires_questions(sample_topics):
    with pytest.raises(ValidationError):
        Exam(id="e1", topics=sample_topics, difficulty="easy", questions=[])


def test_exam_json_round_trip(sample_exam):
    assert Exam.model_validate_json(sample_exam.model_dump_json()) == sample_exam


def test_submission_rejects_out_of_range_answer():
    with pytest.raises(ValidationError):
        ExamSubmission(exam_id="e1", user_answers={"q1": 7})


def test_topic_score_correct_cannot_exceed_total():
    with pytest.raises(ValidationError):
        TopicScore(topic_title="X", correct=3, total=2, percentage=100.0, is_weakness=False)
