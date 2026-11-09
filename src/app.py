"""EzGrade Streamlit UI.  Run with:  streamlit run src/app.py

Flow (a simple state machine driven by ``st.session_state.step``):
    0 Upload syllabus -> 1 Configure exam -> 2 Take exam -> 3 Results
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# `streamlit run src/app.py` only puts src/ on sys.path; add the project root so `src.*` imports work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.generator import MAX_QUESTIONS, MIN_QUESTIONS, ExamGenerator, get_strategy  # noqa: E402
from src.grader import WEAKNESS_THRESHOLD, grade_exam  # noqa: E402
from src.llm_client import get_model_name, is_llm_available  # noqa: E402
from src.models import DIFFICULTIES, Exam, ExamResult, ExamSubmission  # noqa: E402
from src.parser import parse_syllabus  # noqa: E402
from src.storage import OPTION_LABELS, export_exam_to_markdown, list_results, save_exam, save_result  # noqa: E402
from src.validator import validate_exam  # noqa: E402

STEPS = ("Upload syllabus", "Configure exam", "Take exam", "Results")

SAMPLE_SYLLABUS = """Introduction to Computer Science
Week 1: Data Structures - arrays, linked lists, stacks, queues, hash tables
Week 2: Algorithms - sorting, binary search, recursion
Week 3: Complexity Analysis - Big-O notation, amortized analysis
"""

_DEFAULT_STATE = {
    "step": 0,
    "topics": None,
    "exam": None,
    "result": None,
    "validation_errors": [],
}


def init_state() -> None:
    for key, value in _DEFAULT_STATE.items():
        st.session_state.setdefault(key, value)


def go_to(step: int) -> None:
    st.session_state.step = step
    st.rerun()


def reset() -> None:
    for key, value in _DEFAULT_STATE.items():
        st.session_state[key] = value
    st.rerun()


def answer_key(exam: Exam, question_id: str) -> str:
    return f"answer_{exam.id}_{question_id}"


def parse_uploaded_file(uploaded_file) -> list:
    suffix = Path(uploaded_file.name).suffix.lower()
    # Written to a temp file because parse_syllabus works on paths; closed before parsing (Windows locks open files).
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        return parse_syllabus(tmp_path)
    finally:
        os.unlink(tmp_path)


def render_sidebar() -> None:
    with st.sidebar:
        st.title("EzGrade")
        if is_llm_available():
            st.success(f"Gemini connected ({get_model_name()})")
        else:
            st.warning("Offline mode: no GEMINI_API_KEY set, using sample data.")

        st.subheader("Progress")
        for index, name in enumerate(STEPS):
            marker = "▶" if index == st.session_state.step else ("✔" if index < st.session_state.step else "○")
            st.write(f"{marker} {index + 1}. {name}")

        st.caption(f"Saved results: {len(list_results())}")
        if st.button("Start over", width="stretch"):
            reset()


def render_upload_step() -> None:
    st.header("1. Upload your syllabus")
    uploaded_file = st.file_uploader("Syllabus file", type=["pdf", "txt", "md"])
    pasted_text = st.text_area("...or paste the syllabus text", value=SAMPLE_SYLLABUS, height=200)

    if st.button("Extract topics", type="primary"):
        try:
            with st.spinner("Extracting topics..."):
                topics = parse_uploaded_file(uploaded_file) if uploaded_file else parse_syllabus(pasted_text)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state.topics = topics
        go_to(1)


def render_config_step() -> None:
    st.header("2. Configure your exam")
    topics = st.session_state.topics

    with st.expander("Extracted topics", expanded=True):
        for topic in topics:
            st.markdown(f"**{topic.title}** - {', '.join(topic.subtopics) or 'no subtopics'}")

    titles = [topic.title for topic in topics]
    selected_titles = st.multiselect("Topics to include", titles, default=titles)
    difficulty = st.select_slider("Difficulty", options=list(DIFFICULTIES), value="medium")
    num_questions = st.number_input("Number of questions", MIN_QUESTIONS, MAX_QUESTIONS, value=5)

    back_col, next_col = st.columns(2)
    if back_col.button("Back"):
        go_to(0)
    if next_col.button("Generate exam", type="primary", disabled=not selected_titles):
        selected = [topic for topic in topics if topic.title in selected_titles]
        with st.spinner("Generating questions..."):
            exam = ExamGenerator(get_strategy(difficulty)).generate(selected, int(num_questions))
        is_valid, errors = validate_exam(exam)
        st.session_state.exam = exam
        st.session_state.validation_errors = [] if is_valid else errors
        st.session_state.result = None
        try:
            save_exam(exam)
        except OSError as exc:
            st.warning(f"Could not save exam: {exc}")
        go_to(2)


def render_exam_step() -> None:
    exam: Exam = st.session_state.exam
    st.header(f"3. Practice exam ({exam.difficulty}, {len(exam.questions)} questions)")
    for error in st.session_state.validation_errors:
        st.warning(f"Validation: {error}")

    with st.form("exam_form"):
        for number, question in enumerate(exam.questions, start=1):
            st.radio(
                f"**Q{number}.** {question.question_text}",
                options=list(range(len(question.options))),
                # Default arg binds this question's options (avoids the late-binding closure pitfall).
                format_func=lambda i, opts=question.options: f"{OPTION_LABELS[i]}) {opts[i]}",
                index=None,
                key=answer_key(exam, question.id),
            )
        submitted = st.form_submit_button("Submit answers", type="primary")

    if submitted:
        answers = {
            question.id: st.session_state[answer_key(exam, question.id)]
            for question in exam.questions
            if st.session_state.get(answer_key(exam, question.id)) is not None
        }
        result = grade_exam(exam, ExamSubmission(exam_id=exam.id, user_answers=answers))
        st.session_state.result = result
        try:
            save_result(result)
        except OSError as exc:
            st.warning(f"Could not save result: {exc}")
        go_to(3)


def render_results_step() -> None:
    exam: Exam = st.session_state.exam
    result: ExamResult = st.session_state.result
    st.header("4. Results")

    st.metric("Total score", f"{result.total_score:.1f}%")

    st.subheader("Topic breakdown")
    st.dataframe([score.model_dump() for score in result.topic_breakdown], width="stretch")

    st.subheader(f"Weaknesses (< {WEAKNESS_THRESHOLD:.0f}%)")
    if result.weaknesses:
        for score in result.weaknesses:
            st.error(f"{score.topic_title}: {score.percentage:.1f}% ({score.correct}/{score.total})")
    else:
        st.success("No weak topics - great job!")

    st.subheader("Question feedback")
    for number, question in enumerate(exam.questions, start=1):
        chosen = st.session_state.get(answer_key(exam, question.id))
        with st.expander(f"Q{number}. {question.question_text}"):
            your_answer = f"{OPTION_LABELS[chosen]}) {question.options[chosen]}" if chosen is not None else "-"
            correct = question.correct_index
            st.write(f"**Your answer:** {your_answer}")
            st.write(f"**Correct answer:** {OPTION_LABELS[correct]}) {question.options[correct]}")
            st.write(f"**Explanation:** {question.explanation}")
            st.info(result.feedback_per_question.get(question.id, "No feedback."))

    st.download_button(
        "Download exam (Markdown)",
        data=export_exam_to_markdown(exam, include_answers=True),
        file_name=f"ezgrade_exam_{exam.id}.md",
        mime="text/markdown",
    )
    if st.button("Generate another exam"):
        go_to(1)


STEP_RENDERERS = (render_upload_step, render_config_step, render_exam_step, render_results_step)


def main() -> None:
    st.set_page_config(page_title="EzGrade", page_icon="📝", layout="centered")
    init_state()
    render_sidebar()
    STEP_RENDERERS[st.session_state.step]()


if __name__ == "__main__":
    main()
