
"""Exam grading and weakness mapping."""

from __future__ import annotations

from collections import Counter

from src.models import Exam, ExamResult, ExamSubmission, TopicScore

WEAKNESS_THRESHOLD = 70.0  # percent; a topic scoring strictly below this is a weakness


def calculate_percentage(correct: int, total: int) -> float:
    return 0.0 if total == 0 else round(100.0 * correct / total, 2)


def is_weakness(percentage: float, threshold: float = WEAKNESS_THRESHOLD) -> bool:
    return percentage < threshold


def grade_exam(exam: Exam, submission: ExamSubmission) -> ExamResult:
    """Grade a submission and map per-topic weaknesses.

    A missing answer counts as wrong. Answers for question ids that are not in
    the exam are ignored. Runs in O(Q + T) for Q questions and T topics.
    """
    if submission.exam_id != exam.id:
        raise ValueError(f"Submission is for exam {submission.exam_id!r}, not {exam.id!r}.")

    correct_by_topic: Counter[str] = Counter()
    total_by_topic: Counter[str] = Counter()
    feedback: dict[str, str] = {}

    for question in exam.questions:
        answer = submission.user_answers.get(question.id)
        total_by_topic[question.topic_id] += 1
        if answer == question.correct_index:
            correct_by_topic[question.topic_id] += 1
            feedback[question.id] = "Correct!"
        else:
            correct_option = question.options[question.correct_index]
            prefix = "Not answered" if answer is None else "Incorrect"
            feedback[question.id] = f"{prefix} - correct answer: {correct_option}. {question.explanation}"

    topic_breakdown: list[TopicScore] = []
    for topic in exam.topics:
        total = total_by_topic[topic.id]
        if total == 0:
            continue
        correct = correct_by_topic[topic.id]
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

    total_correct = sum(correct_by_topic.values())
    return ExamResult(
        exam_id=exam.id,
        total_score=calculate_percentage(total_correct, len(exam.questions)),
        topic_breakdown=topic_breakdown,
        feedback_per_question=feedback,
    )
