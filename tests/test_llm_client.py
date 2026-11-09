import pytest

from src import llm_client
from src.llm_client import ask_gemini_json, extract_json_text
from src.models import StrictModel

FALLBACK = {"answer": "fallback", "items": [1, 2]}


class Answer(StrictModel):
    answer: str
    items: list[int]


@pytest.fixture
def fake_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def fake_gemini(monkeypatch, text: str | Exception):
    def _call(prompt, api_key, model):
        if isinstance(text, Exception):
            raise text
        return text

    monkeypatch.setattr(llm_client, "_call_gemini", _call)


def test_no_api_key_returns_fallback_copy():
    result = ask_gemini_json("prompt", FALLBACK)
    assert result == FALLBACK
    assert result is not FALLBACK


@pytest.mark.parametrize(
    "raw",
    ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here you go: {"a": 1} Hope that helps.'],
)
def test_extract_json_text(raw):
    assert extract_json_text(raw) == '{"a": 1}'


def test_valid_response_is_returned(monkeypatch, fake_key):
    fake_gemini(monkeypatch, '```json\n{"answer": "llm", "items": [3]}\n```')
    assert ask_gemini_json("prompt", FALLBACK, response_model=Answer) == {"answer": "llm", "items": [3]}


@pytest.mark.parametrize(
    "response",
    ["not json at all", "[1, 2, 3]", '{"answer": "missing items key"}', RuntimeError("network down")],
)
def test_bad_responses_fall_back(monkeypatch, fake_key, response):
    fake_gemini(monkeypatch, response)
    assert ask_gemini_json("prompt", FALLBACK) == FALLBACK


def test_schema_mismatch_falls_back(monkeypatch, fake_key):
    fake_gemini(monkeypatch, '{"answer": "llm", "items": ["not-an-int"]}')
    assert ask_gemini_json("prompt", FALLBACK, response_model=Answer) == FALLBACK
