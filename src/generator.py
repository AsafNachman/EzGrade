"""Exam generation using the Strategy pattern.

``DifficultyStrategy`` encapsulates *how* questions of a given difficulty are
requested from the LLM; ``ExamGenerator`` (the context) is agnostic to which
strategy it holds, so new difficulties are added without touching it.

OWNER: <teammate name>
STATUS: the pipeline works end-to-end; the per-difficulty prompts are first drafts.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from src.llm_client import ask_gemini_json
from src.models import NUM_OPTIONS, Difficulty, Exam, Question, StrictModel, SyllabusTopic

MIN_QUESTIONS = 1
MAX_QUESTIONS = 30

LLMCallable = Callable[..., dict[str, Any]]


class QuestionBatch(StrictModel):
    """Expected top-level shape of the LLM response."""

    questions: list[Question]


class DifficultyStrategy(ABC):
    """Abstract strategy. Subclasses only supply the difficulty-specific instructions;
    ``build_prompt`` is a Template Method shared by all of them."""

    name: Difficulty

    @abstractmethod
    def difficulty_instructions(self) -> str:
        """Difficulty-specific guidance appended to the shared prompt."""

    def build_prompt(self, topics: Sequence[SyllabusTopic], num_questions: int) -> str:
        topic_lines = "\n".join(
            f'- id="{topic.id}" title="{topic.title}" subtopics={topic.subtopics}' for topic in topics
        )
        return (
            "You are an expert university examiner writing a multiple-choice practice exam.\n"
            f"Write exactly {num_questions} questions spread evenly across these topics:\n"
            f"{topic_lines}\n\n"
            f"Difficulty: {self.name.upper()}.\n"
            f"{self.difficulty_instructions()}\n\n"
            'Return JSON of the form {"questions": [ ... ]} where every question has:\n'
            '  "id" (string), "topic_id" (one of the ids above), "question_text" (string),\n'
            f'  "options" (exactly {NUM_OPTIONS} distinct strings), '
            f'"correct_index" (integer 0-{NUM_OPTIONS - 1}),\n'
            f'  "explanation" (why the answer is correct), "difficulty" ("{self.name}").'
        )


class EasyStrategy(DifficultyStrategy):
    name: Difficulty = "easy"

    def difficulty_instructions(self) -> str:
        # TODO(generator): refine - recall/definition questions, obviously wrong distractors.
        return "Test recall of definitions and basic facts. Distractors should be clearly wrong."


class MediumStrategy(DifficultyStrategy):
    name: Difficulty = "medium"

    def difficulty_instructions(self) -> str:
        # TODO(generator): refine - application questions, plausible distractors from common mistakes.
        return "Test application of concepts to short scenarios. Distractors should reflect common mistakes."


class HardStrategy(DifficultyStrategy):
    name: Difficulty = "hard"

    def difficulty_instructions(self) -> str:
        # TODO(generator): refine - multi-step reasoning, combine subtopics, near-miss distractors;
        #   consider a higher temperature or a second "make it harder" pass.
        return "Require multi-step reasoning that combines subtopics. All distractors must be plausible."


STRATEGY_REGISTRY: dict[Difficulty, type[DifficultyStrategy]] = {
    "easy": EasyStrategy,
    "medium": MediumStrategy,
    "hard": HardStrategy,
}


def get_strategy(difficulty: str) -> DifficultyStrategy:
    """Factory: difficulty name -> strategy instance."""
    try:
        return STRATEGY_REGISTRY[difficulty]()  # type: ignore[index]
    except KeyError:
        raise ValueError(
            f"Unknown difficulty {difficulty!r}; expected one of {list(STRATEGY_REGISTRY)}"
        ) from None


def build_placeholder_questions(
    topics: Sequence[SyllabusTopic], num_questions: int, difficulty: Difficulty
) -> list[Question]:
    """Deterministic offline questions so the app works without an API key."""
    questions = []
    for i in range(num_questions):
        topic = topics[i % len(topics)]
        subtopic = topic.subtopics[i % len(topic.subtopics)] if topic.subtopics else topic.title
        correct_index = i % NUM_OPTIONS
        questions.append(
            Question(
                id=f"q{i + 1}",
                topic_id=topic.id,
                question_text=f"[Sample {i + 1}] Which statement about '{subtopic}' ({topic.title}) is correct?",
                options=[
                    "This is the correct statement." if j == correct_index else f"Incorrect statement #{j + 1}."
                    for j in range(NUM_OPTIONS)
                ],
                correct_index=correct_index,
                explanation="Placeholder question generated offline (no Gemini API key configured).",
                difficulty=difficulty,
            )
        )
    return questions


class ExamGenerator:
    """Strategy context. The LLM function is injected (Dependency Injection) so tests
    can pass a fake instead of hitting the network."""

    def __init__(self, strategy: DifficultyStrategy, llm: LLMCallable = ask_gemini_json) -> None:
        self.strategy = strategy
        self._llm = llm

    def set_strategy(self, strategy: DifficultyStrategy) -> None:
        self.strategy = strategy

    def generate(self, topics: Sequence[SyllabusTopic], num_questions: int = 5) -> Exam:
        if not topics:
            raise ValueError("At least one topic is required to generate an exam.")
        if not MIN_QUESTIONS <= num_questions <= MAX_QUESTIONS:
            raise ValueError(f"num_questions must be between {MIN_QUESTIONS} and {MAX_QUESTIONS}.")

        difficulty = self.strategy.name
        placeholder = build_placeholder_questions(topics, num_questions, difficulty)
        fallback = {"questions": [question.model_dump(mode="json") for question in placeholder]}

        raw = self._llm(self.strategy.build_prompt(topics, num_questions), fallback, response_model=QuestionBatch)
        try:
            questions = QuestionBatch.model_validate(raw).questions[:num_questions]
        except ValidationError:
            questions = []
        if not questions:
            questions = placeholder

        # TODO(generator): if the LLM used an unknown topic_id, remap it (e.g. fuzzy-match on
        #   title) instead of relying on the validator to reject the exam.
        # TODO(generator): shuffle options per question (and fix correct_index) so the
        #   answer position is not biased by the LLM.

        # Ids are reassigned so they are unique and stable regardless of what the LLM returned.
        questions = [
            question.model_copy(update={"id": f"q{i}", "difficulty": difficulty})
            for i, question in enumerate(questions, start=1)
        ]
        return Exam(id=uuid.uuid4().hex, topics=list(topics), difficulty=difficulty, questions=questions)
