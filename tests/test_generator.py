"""Unit tests for ``src.generator``. The live Gemini API is never called."""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from src.generator import (
    DifficultyStrategy,
    EasyDifficultyStrategy,
    EasyStrategy,
    ExamGenerator,
    HardDifficultyStrategy,
    HardStrategy,
    MediumDifficultyStrategy,
    MediumStrategy,
    get_strategy,
)
from src.models import NUM_OPTIONS, Exam, Question

ALL_STRATEGIES = (EasyDifficultyStrategy, MediumDifficultyStrategy, HardDifficultyStrategy)


def make_llm_question(topic_id: str, n: int, correct_index: int = 0) -> dict:
    """One well-formed question as the LLM would return it."""
    return {
        "topic_id": topic_id,
        "question_text": f"Question {n}?",
        "options": [f"Q{n} option {j}" for j in range(NUM_OPTIONS)],
        "correct_index": correct_index,
        "explanation": f"Option {correct_index} is right because ...",
    }


def correct_answer(question: Question) -> str:
    return question.options[question.correct_index]


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
def test_strategy_base_class_is_abstract():
    with pytest.raises(TypeError):
        DifficultyStrategy()  # type: ignore[abstract]


@pytest.mark.parametrize("cls", ALL_STRATEGIES)
def test_strategy_instructions_are_non_empty(cls):
    instructions = cls().get_prompt_instructions()
    assert isinstance(instructions, str)
    assert instructions.strip()


def test_strategy_instructions_are_unique():
    instructions = [cls().get_prompt_instructions() for cls in ALL_STRATEGIES]
    assert len(set(instructions)) == len(ALL_STRATEGIES)


@pytest.mark.parametrize(
    ("cls", "keyword"),
    [
        (EasyDifficultyStrategy, "recall"),
        (MediumDifficultyStrategy, "scenario"),
        (HardDifficultyStrategy, "edge cases"),
    ],
)
def test_strategy_instructions_match_their_focus(cls, keyword):
    assert keyword in cls().get_prompt_instructions().lower()


@pytest.mark.parametrize(
    ("name", "cls"),
    [("easy", EasyDifficultyStrategy), ("medium", MediumDifficultyStrategy), ("hard", HardDifficultyStrategy)],
)
def test_get_strategy(name, cls):
    strategy = get_strategy(name)
    assert isinstance(strategy, cls)
    assert strategy.name == name


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        get_strategy("impossible")


def test_legacy_aliases_point_to_new_classes():
    assert EasyStrategy is EasyDifficultyStrategy
    assert MediumStrategy is MediumDifficultyStrategy
    assert HardStrategy is HardDifficultyStrategy


def test_prompt_contains_topics_count_and_instructions(sample_topics):
    strategy = HardDifficultyStrategy()
    prompt = strategy.build_prompt(sample_topics, 3)
    assert "Data Structures" in prompt and "Algorithms" in prompt
    assert 'id="t1"' in prompt and 'id="t2"' in prompt
    assert "exactly 3 questions" in prompt
    assert strategy.get_prompt_instructions() in prompt
    assert "correct_index" in prompt and "explanation" in prompt


# --------------------------------------------------------------------------- #
# Strategy switching
# --------------------------------------------------------------------------- #
def test_default_strategy_is_medium():
    assert isinstance(ExamGenerator().strategy, MediumDifficultyStrategy)


def test_set_strategy_switches_strategy():
    generator = ExamGenerator(EasyDifficultyStrategy())
    hard = HardDifficultyStrategy()
    generator.set_strategy(hard)
    assert generator.strategy is hard


def test_set_strategy_rejects_non_strategy():
    with pytest.raises(TypeError):
        ExamGenerator().set_strategy("hard")  # type: ignore[arg-type]


def test_switched_strategy_drives_prompt_and_difficulty(sample_topics):
    with patch("src.generator.ask_gemini_json", side_effect=lambda prompt, fallback, **kw: fallback) as mock_llm:
        generator = ExamGenerator(EasyDifficultyStrategy())
        assert generator.generate_exam(sample_topics, 2).difficulty == "easy"

        generator.set_strategy(HardDifficultyStrategy())
        exam = generator.generate_exam(sample_topics, 2)

    assert exam.difficulty == "hard"
    assert {q.difficulty for q in exam.questions} == {"hard"}
    last_prompt = mock_llm.call_args.args[0]
    assert HardDifficultyStrategy().get_prompt_instructions() in last_prompt
    assert EasyDifficultyStrategy().get_prompt_instructions() not in last_prompt


# --------------------------------------------------------------------------- #
# generate_exam with a mocked LLM
# --------------------------------------------------------------------------- #
def test_generate_exam_builds_valid_exam_from_mocked_llm(sample_topics):
    llm_response = {
        "questions": [
            make_llm_question("t1", 1, correct_index=2),
            make_llm_question("t2", 2, correct_index=0),
            make_llm_question("t1", 3, correct_index=3),
        ]
    }
    with patch("src.generator.ask_gemini_json", return_value=llm_response) as mock_llm:
        exam = ExamGenerator(MediumDifficultyStrategy(), rng=random.Random(0)).generate_exam(
            sample_topics, num_questions=3
        )

    mock_llm.assert_called_once()
    prompt, fallback = mock_llm.call_args.args
    assert "exactly 3 questions" in prompt
    assert len(fallback["questions"]) == 3

    assert isinstance(exam, Exam)
    assert exam.difficulty == "medium"
    assert exam.topics == sample_topics
    assert [q.id for q in exam.questions] == ["q1", "q2", "q3"]
    for question, source in zip(exam.questions, llm_response["questions"]):
        assert isinstance(question, Question)
        assert question.topic_id == source["topic_id"]
        assert question.question_text == source["question_text"]
        assert len(question.options) == NUM_OPTIONS
        assert len(set(question.options)) == NUM_OPTIONS
        assert 0 <= question.correct_index < NUM_OPTIONS
        # Options may be shuffled, but the correct answer must be preserved.
        assert sorted(question.options) == sorted(source["options"])
        assert correct_answer(question) == source["options"][source["correct_index"]]
        assert question.explanation == source["explanation"]
        assert question.difficulty == "medium"


def test_generate_exam_without_api_key_uses_placeholders(sample_topics):
    exam = ExamGenerator(get_strategy("hard")).generate_exam(sample_topics, 5)
    assert len(exam.questions) == 5
    assert {q.difficulty for q in exam.questions} == {"hard"}
    assert len({q.id for q in exam.questions}) == 5
    assert all(correct_answer(q) == "This is the correct statement." for q in exam.questions)


def test_generate_alias_still_works(sample_topics):
    exam = ExamGenerator(get_strategy("easy")).generate(sample_topics, 2)
    assert len(exam.questions) == 2


def test_injected_llm_is_used_instead_of_gemini(sample_topics):
    calls = []

    def fake_llm(prompt, fallback, **kwargs):
        calls.append(prompt)
        return {"questions": [make_llm_question("t2", 1)]}

    with patch("src.generator.ask_gemini_json") as mock_gemini:
        exam = ExamGenerator(llm=fake_llm).generate_exam(sample_topics, 1)

    mock_gemini.assert_not_called()
    assert len(calls) == 1
    assert exam.questions[0].question_text == "Question 1?"


def test_malformed_questions_are_dropped_and_valid_ones_kept(sample_topics):
    llm_response = {
        "questions": [
            make_llm_question("t1", 1),
            {"garbage": True},
            {**make_llm_question("t1", 3), "options": ["a", "b", "c"]},  # too few options
            {**make_llm_question("t1", 4), "options": ["a", "A", "b", "c"]},  # not distinct
            {**make_llm_question("t1", 5), "correct_index": 7},  # out of range
            {**make_llm_question("t1", 6), "correct_index": "1"},  # wrong type
            make_llm_question("t2", 7),
        ]
    }
    with patch("src.generator.ask_gemini_json", return_value=llm_response):
        exam = ExamGenerator().generate_exam(sample_topics, 7)

    assert [q.question_text for q in exam.questions] == ["Question 1?", "Question 7?"]
    assert [q.id for q in exam.questions] == ["q1", "q2"]


def test_all_invalid_llm_output_falls_back_to_placeholders(sample_topics):
    with patch("src.generator.ask_gemini_json", return_value={"questions": [{"garbage": True}]}):
        exam = ExamGenerator(get_strategy("easy")).generate_exam(sample_topics, 3)
    assert len(exam.questions) == 3
    assert all(q.question_text.startswith("[Sample") for q in exam.questions)


@pytest.mark.parametrize("bad_response", [{}, {"questions": "nope"}, {"questions": []}])
def test_wrong_response_shape_falls_back_to_placeholders(sample_topics, bad_response):
    with patch("src.generator.ask_gemini_json", return_value=bad_response):
        exam = ExamGenerator().generate_exam(sample_topics, 2)
    assert len(exam.questions) == 2


def test_extra_questions_are_truncated(sample_topics):
    llm_response = {"questions": [make_llm_question("t1", n) for n in range(1, 6)]}
    with patch("src.generator.ask_gemini_json", return_value=llm_response):
        exam = ExamGenerator().generate_exam(sample_topics, 2)
    assert [q.question_text for q in exam.questions] == ["Question 1?", "Question 2?"]


def test_llm_supplied_id_and_difficulty_are_overridden(sample_topics):
    question = {**make_llm_question("t1", 1), "id": "dup", "difficulty": "easy"}
    with patch("src.generator.ask_gemini_json", return_value={"questions": [question, question]}):
        exam = ExamGenerator(HardDifficultyStrategy()).generate_exam(sample_topics, 2)
    assert [q.id for q in exam.questions] == ["q1", "q2"]
    assert {q.difficulty for q in exam.questions} == {"hard"}


@pytest.mark.parametrize(
    ("raw_topic_id", "expected"),
    [("T2", "t2"), ("algorithms", "t2"), ("Data Structures", "t1"), ("unknown", "t1")],
)
def test_unknown_topic_ids_are_remapped(sample_topics, raw_topic_id, expected):
    with patch("src.generator.ask_gemini_json", return_value={"questions": [make_llm_question(raw_topic_id, 1)]}):
        exam = ExamGenerator().generate_exam(sample_topics, 1)
    assert exam.questions[0].topic_id == expected


def test_options_are_shuffled_but_answer_preserved(sample_topics):
    llm_response = {"questions": [make_llm_question("t1", n, correct_index=0) for n in range(1, 21)]}
    with patch("src.generator.ask_gemini_json", return_value=llm_response):
        exam = ExamGenerator(rng=random.Random(42)).generate_exam(sample_topics, 20)

    positions = {q.correct_index for q in exam.questions}
    assert len(positions) > 1, "correct answer should not always sit in the same position"
    for n, question in enumerate(exam.questions, start=1):
        assert correct_answer(question) == f"Q{n} option 0"


def test_same_seed_gives_same_option_order(sample_topics):
    llm_response = {"questions": [make_llm_question("t1", n) for n in range(1, 6)]}
    with patch("src.generator.ask_gemini_json", return_value=llm_response):
        first = ExamGenerator(rng=random.Random(7)).generate_exam(sample_topics, 5)
        second = ExamGenerator(rng=random.Random(7)).generate_exam(sample_topics, 5)
    assert [q.options for q in first.questions] == [q.options for q in second.questions]


@pytest.mark.parametrize("num_questions", [0, -1, 31, 1000])
def test_rejects_out_of_range_question_count(sample_topics, num_questions):
    with pytest.raises(ValueError):
        ExamGenerator(get_strategy("easy")).generate_exam(sample_topics, num_questions)


@pytest.mark.parametrize("num_questions", ["5", 2.5, True])
def test_rejects_non_integer_question_count(sample_topics, num_questions):
    with pytest.raises(TypeError):
        ExamGenerator().generate_exam(sample_topics, num_questions)  # type: ignore[arg-type]


def test_rejects_empty_topics():
    with patch("src.generator.ask_gemini_json") as mock_llm, pytest.raises(ValueError):
        ExamGenerator().generate_exam([], 3)
    mock_llm.assert_not_called()
