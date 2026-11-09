import pytest

from src.generator import EasyStrategy, ExamGenerator, HardStrategy, MediumStrategy, get_strategy


@pytest.mark.parametrize(
    ("name", "cls"), [("easy", EasyStrategy), ("medium", MediumStrategy), ("hard", HardStrategy)]
)
def test_get_strategy(name, cls):
    strategy = get_strategy(name)
    assert isinstance(strategy, cls)
    assert strategy.name == name


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        get_strategy("impossible")


def test_prompt_mentions_topics(sample_topics):
    prompt = get_strategy("medium").build_prompt(sample_topics, 3)
    assert "Data Structures" in prompt and "Algorithms" in prompt


def test_offline_generation(sample_topics):
    exam = ExamGenerator(get_strategy("hard")).generate(sample_topics, 5)
    assert len(exam.questions) == 5
    assert {q.difficulty for q in exam.questions} == {"hard"}
    assert len({q.id for q in exam.questions}) == 5


def test_invalid_llm_output_falls_back_to_placeholders(sample_topics):
    def broken_llm(prompt, fallback, **kwargs):
        return {"questions": [{"garbage": True}]}

    exam = ExamGenerator(get_strategy("easy"), llm=broken_llm).generate(sample_topics, 3)
    assert len(exam.questions) == 3


@pytest.mark.parametrize("num_questions", [0, 1000])
def test_rejects_bad_question_count(sample_topics, num_questions):
    with pytest.raises(ValueError):
        ExamGenerator(get_strategy("easy")).generate(sample_topics, num_questions)
