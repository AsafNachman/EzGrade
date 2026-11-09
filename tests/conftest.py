import pytest

from src.generator import build_placeholder_questions
from src.llm_client import API_KEY_ENV_VARS
from src.models import Exam, SyllabusTopic


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """Guarantee tests never call the real Gemini API."""
    for env_var in API_KEY_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


@pytest.fixture
def sample_topics() -> list[SyllabusTopic]:
    return [
        SyllabusTopic(id="t1", title="Data Structures", subtopics=["Stacks", "Queues"]),
        SyllabusTopic(id="t2", title="Algorithms", subtopics=["Sorting"]),
    ]


@pytest.fixture
def sample_exam(sample_topics) -> Exam:
    return Exam(
        id="exam123",
        topics=sample_topics,
        difficulty="easy",
        questions=build_placeholder_questions(sample_topics, 4, "easy"),
    )
