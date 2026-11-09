"""Domain models shared by every EzGrade module.

All models are *strict*: no silent type coercion (e.g. ``"1"`` is NOT accepted
for an ``int``), unknown fields are rejected, and assignments are re-validated.
This makes malformed LLM output fail loudly at the boundary instead of
propagating through the app.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NUM_OPTIONS = 4

Difficulty = Literal["easy", "medium", "hard"]
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
OptionIndex = Annotated[int, Field(ge=0, le=NUM_OPTIONS - 1)]
Percentage = Annotated[float, Field(ge=0.0, le=100.0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)


class SyllabusTopic(StrictModel):
    id: NonEmptyStr
    title: NonEmptyStr
    subtopics: list[NonEmptyStr] = Field(default_factory=list)


class Question(StrictModel):
    id: NonEmptyStr
    topic_id: NonEmptyStr
    question_text: NonEmptyStr
    options: Annotated[list[NonEmptyStr], Field(min_length=NUM_OPTIONS, max_length=NUM_OPTIONS)]
    correct_index: OptionIndex
    explanation: NonEmptyStr
    difficulty: Difficulty

    @model_validator(mode="after")
    def _options_must_be_distinct(self) -> Question:
        if len({option.casefold() for option in self.options}) != len(self.options):
            raise ValueError("options must be distinct (case-insensitive)")
        return self


class Exam(StrictModel):
    id: NonEmptyStr
    topics: Annotated[list[SyllabusTopic], Field(min_length=1)]
    difficulty: Difficulty
    questions: Annotated[list[Question], Field(min_length=1)]

    def topics_by_id(self) -> dict[str, SyllabusTopic]:
        return {topic.id: topic for topic in self.topics}

    def questions_by_id(self) -> dict[str, Question]:
        return {question.id: question for question in self.questions}


class ExamSubmission(StrictModel):
    exam_id: NonEmptyStr
    # question_id -> selected option index. Unanswered questions are simply absent.
    user_answers: dict[str, OptionIndex] = Field(default_factory=dict)


class TopicScore(StrictModel):
    topic_title: NonEmptyStr
    correct: Annotated[int, Field(ge=0)]
    total: Annotated[int, Field(ge=0)]
    percentage: Percentage
    is_weakness: bool

    @model_validator(mode="after")
    def _correct_not_above_total(self) -> TopicScore:
        if self.correct > self.total:
            raise ValueError("correct cannot exceed total")
        return self


class ExamResult(StrictModel):
    exam_id: NonEmptyStr
    total_score: Percentage
    topic_breakdown: list[TopicScore] = Field(default_factory=list)
    # question_id -> human-readable feedback for that question.
    feedback_per_question: dict[str, str] = Field(default_factory=dict)

    @property
    def weaknesses(self) -> list[TopicScore]:
        return [score for score in self.topic_breakdown if score.is_weakness]
