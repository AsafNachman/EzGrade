import pytest

from src.models import SyllabusTopic
from src.parser import parse_syllabus, read_source_text


def test_returns_topics_for_text():
    topics = parse_syllabus("Week 1: Graphs\nWeek 2: Dynamic Programming")
    assert topics
    assert all(isinstance(topic, SyllabusTopic) for topic in topics)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        parse_syllabus("   ")


def test_reads_text_file(tmp_path):
    path = tmp_path / "syllabus.txt"
    path.write_text("Week 1: Graphs", encoding="utf-8")
    assert read_source_text(str(path)) == "Week 1: Graphs"


def test_plain_text_is_returned_unchanged():
    assert read_source_text("Week 1: Graphs") == "Week 1: Graphs"


@pytest.mark.xfail(reason="TODO(parser): real topic extraction not implemented yet", strict=False)
def test_topics_reflect_syllabus_content():
    titles = [topic.title.lower() for topic in parse_syllabus("Week 1: Graphs\nWeek 2: Dynamic Programming")]
    assert any("graph" in title for title in titles)
    assert any("dynamic programming" in title for title in titles)
