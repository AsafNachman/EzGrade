"""Exam generation using the Strategy pattern.

``DifficultyStrategy`` encapsulates *how* questions of a given difficulty are
requested from the LLM; ``ExamGenerator`` (the context) is agnostic to which
strategy it holds, so new difficulties are added without touching it.

Pipeline of :meth:`ExamGenerator.generate_exam`:

1. Build a prompt from the topics and the strategy's instructions.
2. Call the LLM (``ask_gemini_json``) with deterministic offline placeholders
   as the fallback, so the app keeps working without an API key.
3. Validate every returned question on its own; malformed ones are dropped
   instead of discarding the whole batch.
4. Repair what can be repaired (unknown ``topic_id``), shuffle the options so
   the correct answer's position is not biased, and assign stable ids.
"""

from __future__ import annotations

import logging
import random
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ConfigDict, ValidationError

from src.llm_client import ask_gemini_json
from src.models import NUM_OPTIONS, Difficulty, Exam, Question, StrictModel, SyllabusTopic

logger = logging.getLogger(__name__)

MIN_QUESTIONS = 1
MAX_QUESTIONS = 30

LLMCallable = Callable[..., dict[str, Any]]


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
class DifficultyStrategy(ABC):
    """Abstract difficulty strategy.

    Subclasses supply only the difficulty-specific guidance via
    :meth:`get_prompt_instructions`; :meth:`build_prompt` is a Template Method
    shared by all of them.
    """

    name: Difficulty

    @abstractmethod
    def get_prompt_instructions(self) -> str:
        """Return the difficulty-specific guidance embedded in the LLM prompt."""

    def build_prompt(self, topics: Sequence[SyllabusTopic], num_questions: int) -> str:
        """Build the full LLM prompt for ``num_questions`` questions on ``topics``."""
        topic_lines = "\n".join(
            f'- id="{topic.id}" title="{topic.title}" subtopics={list(topic.subtopics)}'
            for topic in topics
        )
        return (
            "You are an expert university examiner writing a multiple-choice practice exam.\n"
            f"Write exactly {num_questions} questions, spread as evenly as possible across "
            "these topics (and their subtopics):\n"
            f"{topic_lines}\n\n"
            f"Difficulty: {self.name.upper()}.\n"
            f"{self.get_prompt_instructions()}\n\n"
            "General rules for every question:\n"
            f"- Exactly {NUM_OPTIONS} options, all distinct, with exactly one correct answer.\n"
            "- Options are similar in length and style so the answer cannot be guessed from form.\n"
            '- Never use "all of the above" or "none of the above".\n'
            "- The explanation teaches: say why the correct option is right AND why each "
            "distractor is wrong.\n"
            "- Write in the same language as the topic titles.\n\n"
            'Return JSON of the form {"questions": [ ... ]} where every question has:\n'
            '  "topic_id" (one of the ids above), "question_text" (string),\n'
            f'  "options" (list of exactly {NUM_OPTIONS} distinct strings),\n'
            f'  "correct_index" (integer 0-{NUM_OPTIONS - 1}, the index of the correct option),\n'
            '  "explanation" (string).'
        )


class EasyDifficultyStrategy(DifficultyStrategy):
    """Basic recall: definitions, direct concepts, straightforward distractors."""

    name: Difficulty = "easy"

    def get_prompt_instructions(self) -> str:
        return (
            "Focus on basic recall: definitions, key terms, and direct, single-concept facts "
            "a student should know after one reading of the material.\n"
            "- Ask one thing per question, in short and unambiguous wording.\n"
            "- No calculations or multi-step reasoning.\n"
            "- Distractors are straightforward: clearly wrong to a student who knows the "
            "definition, but still on-topic (no joke answers)."
        )


class MediumDifficultyStrategy(DifficultyStrategy):
    """Concept application: scenario analysis, plausible distractors."""

    name: Difficulty = "medium"

    def get_prompt_instructions(self) -> str:
        return (
            "Focus on applying concepts rather than reciting them.\n"
            "- Most questions present a short, concrete scenario (a code snippet, a case, "
            "a small data set) and ask the student to analyse it, predict an outcome, or pick "
            "the most suitable approach.\n"
            "- Each question may require one or two reasoning steps.\n"
            "- Distractors are plausible: each reflects a common misconception or a typical "
            "mistake a partially-prepared student would make."
        )


class HardDifficultyStrategy(DifficultyStrategy):
    """Complex edge cases, multi-concept synthesis, challenging distractors."""

    name: Difficulty = "hard"

    def get_prompt_instructions(self) -> str:
        return (
            "Focus on deep understanding at the level of a demanding final exam.\n"
            "- Questions synthesise several concepts or subtopics, and probe edge cases, "
            "boundary conditions, trade-offs, and exceptions to general rules.\n"
            "- Solving a question requires multi-step reasoning; the answer must not be "
            "findable by keyword matching.\n"
            "- Distractors are challenging near-misses: each is correct in a slightly "
            "different setting, or results from one subtle reasoning error.\n"
            "- Still exactly one option is unambiguously correct."
        )


# Backwards-compatible aliases for the original skeleton names.
EasyStrategy = EasyDifficultyStrategy
MediumStrategy = MediumDifficultyStrategy
HardStrategy = HardDifficultyStrategy

STRATEGY_REGISTRY: dict[Difficulty, type[DifficultyStrategy]] = {
    "easy": EasyDifficultyStrategy,
    "medium": MediumDifficultyStrategy,
    "hard": HardDifficultyStrategy,
}


def get_strategy(difficulty: str) -> DifficultyStrategy:
    """Factory: difficulty name -> strategy instance."""
    try:
        return STRATEGY_REGISTRY[difficulty]()  # type: ignore[index]
    except KeyError:
        raise ValueError(
            f"Unknown difficulty {difficulty!r}; expected one of {list(STRATEGY_REGISTRY)}"
        ) from None


# --------------------------------------------------------------------------- #
# LLM response parsing
# --------------------------------------------------------------------------- #
class GeneratedQuestion(StrictModel):
    """Shape of one question as returned by the LLM.

    Extra keys (e.g. an ``id`` or ``difficulty`` the model added anyway) are
    ignored: the generator assigns those itself.
    """

    model_config = ConfigDict(strict=True, extra="ignore", validate_assignment=True)

    topic_id: str
    question_text: str
    options: list[str]
    correct_index: int
    explanation: str


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
                question_text=(
                    f"[Sample {i + 1}] Which statement about '{subtopic}' ({topic.title}) is correct?"
                ),
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


def _resolve_topic_id(raw_topic_id: str, topics: Sequence[SyllabusTopic], position: int) -> str:
    """Map the LLM's ``topic_id`` onto a real topic id.

    Accepts an exact id, a case-insensitive id, or a topic *title*; anything
    else is assigned round-robin so the question is kept rather than dropped.
    """
    wanted = raw_topic_id.strip().casefold()
    for topic in topics:
        if wanted in (topic.id.casefold(), topic.title.casefold()):
            return topic.id
    logger.info("LLM used unknown topic_id %r; assigning round-robin.", raw_topic_id)
    return topics[position % len(topics)].id


def _shuffle_options(question: Question, rng: random.Random) -> Question:
    """Return a copy of ``question`` with shuffled options and an updated ``correct_index``."""
    order = list(range(NUM_OPTIONS))
    rng.shuffle(order)
    return question.model_copy(
        update={
            "options": [question.options[i] for i in order],
            "correct_index": order.index(question.correct_index),
        }
    )


def parse_llm_questions(
    raw: Any, topics: Sequence[SyllabusTopic], difficulty: Difficulty
) -> list[Question]:
    """Convert the LLM's JSON into validated :class:`Question` objects.

    Each item is validated independently; malformed items are skipped. Ids are
    temporary and are reassigned by the caller.
    """
    items = raw.get("questions") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []

    questions: list[Question] = []
    for position, item in enumerate(items):
        try:
            generated = GeneratedQuestion.model_validate(item)
            questions.append(
                Question(
                    id=f"q{position + 1}",
                    topic_id=_resolve_topic_id(generated.topic_id, topics, position),
                    question_text=generated.question_text,
                    options=generated.options,
                    correct_index=generated.correct_index,
                    explanation=generated.explanation,
                    difficulty=difficulty,
                )
            )
        except (ValidationError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed LLM question #%d: %s", position + 1, exc)
    return questions


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
class ExamGenerator:
    """Strategy context that turns syllabus topics into an :class:`Exam`.

    The LLM function and the random generator are injectable (Dependency
    Injection) so tests can run deterministically without the network.
    """

    def __init__(
        self,
        strategy: DifficultyStrategy | None = None,
        llm: LLMCallable | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """Create a generator.

        Args:
            strategy: Difficulty strategy; defaults to :class:`MediumDifficultyStrategy`.
            llm: Callable with the signature of ``ask_gemini_json``; defaults to it.
            rng: Random source used to shuffle options; defaults to a fresh ``Random()``.
        """
        self.strategy: DifficultyStrategy = (
            strategy if strategy is not None else MediumDifficultyStrategy()
        )
        self._llm = llm
        self._rng = rng if rng is not None else random.Random()

    def set_strategy(self, strategy: DifficultyStrategy) -> None:
        """Swap the difficulty strategy at runtime."""
        if not isinstance(strategy, DifficultyStrategy):
            raise TypeError(f"strategy must be a DifficultyStrategy, got {type(strategy).__name__}")
        self.strategy = strategy

    def generate_exam(self, topics: Sequence[SyllabusTopic], num_questions: int = 5) -> Exam:
        """Generate an exam of ``num_questions`` MCQs on ``topics``.

        Falls back to deterministic placeholder questions if the LLM is
        unavailable or returns nothing usable.

        Raises:
            ValueError: if ``topics`` is empty or ``num_questions`` is out of range.
            TypeError: if ``num_questions`` is not an integer.
        """
        if not topics:
            raise ValueError("At least one topic is required to generate an exam.")
        if isinstance(num_questions, bool) or not isinstance(num_questions, int):
            raise TypeError("num_questions must be an integer.")
        if not MIN_QUESTIONS <= num_questions <= MAX_QUESTIONS:
            raise ValueError(f"num_questions must be between {MIN_QUESTIONS} and {MAX_QUESTIONS}.")

        difficulty = self.strategy.name
        placeholder = build_placeholder_questions(topics, num_questions, difficulty)
        fallback = {"questions": [question.model_dump(mode="json") for question in placeholder]}

        # Resolved at call time (not as a default argument) so tests can patch it.
        llm = self._llm if self._llm is not None else ask_gemini_json
        raw = llm(self.strategy.build_prompt(topics, num_questions), fallback)

        questions = parse_llm_questions(raw, topics, difficulty)[:num_questions]
        if not questions:
            logger.warning("No usable questions from the LLM; using offline placeholders.")
            questions = placeholder
        elif len(questions) < num_questions:
            logger.warning(
                "LLM returned %d of %d requested questions.", len(questions), num_questions
            )

        # Ids are reassigned so they are unique and stable regardless of what the LLM returned.
        questions = [
            _shuffle_options(question, self._rng).model_copy(update={"id": f"q{i}"})
            for i, question in enumerate(questions, start=1)
        ]
        return Exam(
            id=uuid.uuid4().hex, topics=list(topics), difficulty=difficulty, questions=questions
        )

    def generate(self, topics: Sequence[SyllabusTopic], num_questions: int = 5) -> Exam:
        """Alias of :meth:`generate_exam`, kept for existing callers (``src/app.py``)."""
        return self.generate_exam(topics, num_questions)
