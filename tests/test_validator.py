from src.validator import sanitize_text, validate_exam


def test_valid_exam_passes(sample_exam):
    assert validate_exam(sample_exam) == (True, [])


def test_duplicate_question_ids_flagged(sample_exam):
    sample_exam.questions[1] = sample_exam.questions[1].model_copy(update={"id": sample_exam.questions[0].id})
    is_valid, errors = validate_exam(sample_exam)
    assert not is_valid
    assert any("Duplicate question id" in error for error in errors)


def test_unknown_topic_flagged(sample_exam):
    sample_exam.questions[0] = sample_exam.questions[0].model_copy(update={"topic_id": "nope"})
    is_valid, errors = validate_exam(sample_exam)
    assert not is_valid
    assert any("unknown topic" in error for error in errors)


def test_malformed_dict_is_reported_not_raised():
    is_valid, errors = validate_exam({"id": "x"})
    assert not is_valid and errors


def test_sanitize_text_removes_control_chars_and_truncates():
    assert sanitize_text("  a\x00b\t\t c\n\n\n\nd  ") == "ab c\n\nd"
    assert len(sanitize_text("x" * 100, max_length=10)) == 10
