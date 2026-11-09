"""Exam grading and weakness mapping.

OWNER: <teammate name>
STATUS: stub. ``grade_exam`` returns a structurally valid dummy result.
"""

from __future__ import annotations

from src.models import Exam, ExamResult, ExamSubmission, TopicScore

WEAKNESS_THRESHOLD = 70.0  # percent; a topic scoring strictly below this is a weakness


def calculate_percentage(correct: int, total: int) -> float:
    return 0.0 if total == 0 else round(100.0 * correct / total, 2)


def is_weakness(percentage: float, threshold: float = WEAKNESS_THRESHOLD) -> bool:
    return percentage < threshold


def grade_exam(exam: Exam, submission: ExamSubmission) -> ExamResult:
    """Grade a submission and map per-topic weaknesses.

    TODO(grader): replace the dummy result below with the real algorithm:
      1. q_by_id = exam.questions_by_id()                                   # O(Q)
      2. For each question: answer = submission.user_answers.get(q.id)
         (missing answer == wrong). Count correct/total per q.topic_id.     # O(Q)
      3. For each topic in exam.topics that has >= 1 question, build a
         TopicScore with calculate_percentage() and is_weakness().
      4. total_score = calculate_percentage(total_correct, len(exam.questions)).
      5. feedback_per_question[q.id]: "Correct!" or
         "Incorrect - correct answer: <option>. <explanation>".
      6. Ignore answers for question ids that are not in the exam.
    """
    if submission.exam_id != exam.id:
        raise ValueError(f"Submission is for exam {submission.exam_id!r}, not {exam.id!r}.")

    # ---- DUMMY RESULT (delete once implemented) ----
    topic_breakdown = [
        TopicScore(topic_title=topic.title, correct=0, total=0, percentage=0.0, is_weakness=is_weakness(0.0))
        for topic in exam.topics
    ]
    feedback = {question.id: "TODO: grading not implemented yet." for question in exam.questions}
    return ExamResult(
        exam_id=exam.id,
        total_score=0.0,
        topic_breakdown=topic_breakdown,
        feedback_per_question=feedback,
    )
