"""Exam grading and weakness mapping."""

from __future__ import annotations

from src.models import Exam, ExamResult, ExamSubmission, TopicScore

WEAKNESS_THRESHOLD = 70.0  # percent; a topic scoring strictly below this is a weakness


def calculate_percentage(correct: int, total: int) -> float:
    return 0.0 if total == 0 else round(100.0 * correct / total, 2)


def is_weakness(percentage: float, threshold: float = WEAKNESS_THRESHOLD) -> bool:
    return percentage < threshold


def grade_exam(exam: Exam, submission: ExamSubmission) -> ExamResult:
    """Grade a submission and map per-topic weaknesses."""
    if submission.exam_id != exam.id:
        raise ValueError(f"Submission is for exam {submission.exam_id!r}, not {exam.id!r}.")

    topic_totals: dict[str, int] = {}
    topic_correct: dict[str, int] = {}
    feedback: dict[str, str] = {}
    total_correct = 0

    for question in exam.questions:
        topic_totals[question.topic_id] = topic_totals.get(question.topic_id, 0) + 1
        answer = submission.user_answers.get(question.id)
        if answer == question.correct_index:
            total_correct += 1
            topic_correct[question.topic_id] = topic_correct.get(question.topic_id, 0) + 1
            feedback[question.id] = "Correct!"
        else:
            correct_option = question.options[question.correct_index]
            feedback[question.id] = f"Incorrect - correct answer: {correct_option}. {question.explanation}"

    topic_breakdown = []
    for topic in exam.topics:
        total = topic_totals.get(topic.id, 0)
        if total == 0:
            continue
        correct = topic_correct.get(topic.id, 0)
        percentage = calculate_percentage(correct, total)
        topic_breakdown.append(
            TopicScore(
                topic_title=topic.title,
                correct=correct,
                total=total,
                percentage=percentage,
                is_weakness=is_weakness(percentage),
            )
        )

    return ExamResult(
        exam_id=exam.id,
        total_score=calculate_percentage(total_correct, len(exam.questions)),
        topic_breakdown=topic_breakdown,
        feedback_per_question=feedback,
    )
