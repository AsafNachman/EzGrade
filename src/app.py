
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

STEPS = ("העלאת סילבוס", "הגדרת מבחן", "פתרון המבחן", "תוצאות")

DIFFICULTY_LABELS = {"easy": "קל", "medium": "בינוני", "hard": "קשה"}

RTL_CSS = """
<style>
.stApp, [data-testid="stSidebar"] { direction: rtl; text-align: right; }
.stApp textarea, .stApp input { direction: rtl; text-align: right; }
.stApp code, .stApp pre { direction: ltr; text-align: left; }
</style>
"""

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
            st.success(f"Gemini מחובר ({get_model_name()})")
        else:
            st.warning("מצב אופליין: לא הוגדר GEMINI_API_KEY, משתמשים בנתוני דוגמה.")

        st.subheader("התקדמות")
        for index, name in enumerate(STEPS):
            marker = "▶" if index == st.session_state.step else ("✔" if index < st.session_state.step else "○")
            st.write(f"{marker} {index + 1}. {name}")

        st.caption(f"תוצאות שמורות: {len(list_results())}")
        if st.button("להתחיל מחדש", width="stretch"):
            reset()


def render_upload_step() -> None:
    st.header("1. העלאת הסילבוס")
    uploaded_file = st.file_uploader("קובץ סילבוס", type=["pdf", "txt", "md"])
    pasted_text = st.text_area("...או הדביקו את טקסט הסילבוס", value=SAMPLE_SYLLABUS, height=200)

    if st.button("חילוץ נושאים", type="primary"):
        try:
            with st.spinner("מחלצים נושאים..."):
                topics = parse_uploaded_file(uploaded_file) if uploaded_file else parse_syllabus(pasted_text)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state.topics = topics
        go_to(1)


def render_config_step() -> None:
    st.header("2. הגדרת המבחן")
    topics = st.session_state.topics

    with st.expander("הנושאים שנמצאו", expanded=True):
        for topic in topics:
            st.markdown(f"**{topic.title}** - {', '.join(topic.subtopics) or 'אין תתי-נושאים'}")

    titles = [topic.title for topic in topics]
    selected_titles = st.multiselect("נושאים למבחן", titles, default=titles)
    difficulty = st.select_slider(
        "רמת קושי", options=list(DIFFICULTIES), value="medium", format_func=DIFFICULTY_LABELS.get
    )
    num_questions = st.number_input("מספר שאלות", MIN_QUESTIONS, MAX_QUESTIONS, value=5)

    back_col, next_col = st.columns(2)
    if back_col.button("חזרה"):
        go_to(0)
    if next_col.button("יצירת מבחן", type="primary", disabled=not selected_titles):
        selected = [topic for topic in topics if topic.title in selected_titles]
        with st.spinner("יוצרים שאלות..."):
            exam = ExamGenerator(get_strategy(difficulty)).generate(selected, int(num_questions))
        is_valid, errors = validate_exam(exam)
        st.session_state.exam = exam
        st.session_state.validation_errors = [] if is_valid else errors
        st.session_state.result = None
        try:
            save_exam(exam)
        except OSError as exc:
            st.warning(f"לא ניתן לשמור את המבחן: {exc}")
        go_to(2)


def render_exam_step() -> None:
    exam: Exam = st.session_state.exam
    st.header(f"3. מבחן תרגול ({DIFFICULTY_LABELS[exam.difficulty]}, {len(exam.questions)} שאלות)")
    for error in st.session_state.validation_errors:
        st.warning(f"בדיקת תקינות: {error}")

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
        submitted = st.form_submit_button("הגשת תשובות", type="primary")

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
            st.warning(f"לא ניתן לשמור את התוצאה: {exc}")
        go_to(3)


def render_results_step() -> None:
    exam: Exam = st.session_state.exam
    result: ExamResult = st.session_state.result
    st.header("4. תוצאות")

    st.metric("ציון כולל", f"{result.total_score:.1f}%")

    st.subheader("פירוט לפי נושא")
    st.dataframe(
        [
            {
                "נושא": score.topic_title,
                "נכונות": score.correct,
                "סה״כ": score.total,
                "אחוז": score.percentage,
                "נקודת חולשה": score.is_weakness,
            }
            for score in result.topic_breakdown
        ],
        width="stretch",
    )

    st.subheader(f"נקודות חולשה (מתחת ל-{WEAKNESS_THRESHOLD:.0f}%)")
    if result.weaknesses:
        for score in result.weaknesses:
            st.error(f"{score.topic_title}: {score.percentage:.1f}% ({score.correct}/{score.total})")
    else:
        st.success("אין נקודות חולשה, כל הכבוד!")

    st.subheader("משוב לכל שאלה")
    for number, question in enumerate(exam.questions, start=1):
        chosen = st.session_state.get(answer_key(exam, question.id))
        with st.expander(f"Q{number}. {question.question_text}"):
            your_answer = f"{OPTION_LABELS[chosen]}) {question.options[chosen]}" if chosen is not None else "-"
            correct = question.correct_index
            st.write(f"**התשובה שלך:** {your_answer}")
            st.write(f"**התשובה הנכונה:** {OPTION_LABELS[correct]}) {question.options[correct]}")
            st.write(f"**הסבר:** {question.explanation}")
            if chosen == correct:
                st.success("נכון!")
            elif chosen is None:
                st.warning("לא נענתה")
            else:
                st.error("לא נכון")

    st.download_button(
        "הורדת המבחן (Markdown)",
        data=export_exam_to_markdown(exam, include_answers=True),
        file_name=f"ezgrade_exam_{exam.id}.md",
        mime="text/markdown",
    )
    if st.button("יצירת מבחן נוסף"):
        go_to(1)


STEP_RENDERERS = (render_upload_step, render_config_step, render_exam_step, render_results_step)


def main() -> None:
    st.set_page_config(page_title="EzGrade", page_icon="📝", layout="centered")
    st.markdown(RTL_CSS, unsafe_allow_html=True)
    init_state()
    render_sidebar()
    STEP_RENDERERS[st.session_state.step]()


if __name__ == "__main__":
    main()
