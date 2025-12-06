import streamlit as st
import pandas as pd
import random
import os

# ----- CSV PATH -----
CSV_PATH = os.path.join("data", "NCERT_MCQs_with_difficulty_by_chapter.csv")


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)

    # Normalise key columns
    df["class"] = df["class"].astype(str)
    df["chapter"] = df["chapter"].fillna("Unknown Chapter")
    df["difficulty"] = df["difficulty"].fillna("Medium")
    df["answer"] = df["answer"].astype(str).str.strip().str.upper()

    return df


def init_quiz_state():
    """Reset quiz-related session_state variables."""
    st.session_state.quiz_indices = []
    st.session_state.quiz_pos = 0
    st.session_state.score = 0
    st.session_state.total_answered = 0
    st.session_state.quiz_params = None
    st.session_state.quiz_finished = False


def main():
    st.title("NCERT MCQ Practice (Classes 6–10)")
    st.write(
        "MCQs loaded from `data/NCERT_MCQs_with_difficulty_by_chapter.csv`.\n\n"
        "Choose filters, select how many questions you want in this quiz, and then attempt them one by one."
    )

    df = load_data()

    # ---------- Sidebar Filters ----------
    st.sidebar.header("Filters")

    class_opts = ["All"] + sorted(df["class"].unique().tolist())
    sel_class = st.sidebar.selectbox("Class", class_opts)

    filtered = df.copy()
    if sel_class != "All":
        filtered = filtered[filtered["class"] == sel_class]

    chapter_opts = ["All"] + sorted(filtered["chapter"].unique().tolist())
    sel_chapter = st.sidebar.selectbox("Chapter", chapter_opts)

    if sel_chapter != "All":
        filtered = filtered[filtered["chapter"] == sel_chapter]

    diff_opts = ["All", "Easy", "Medium", "Hard"]
    sel_diff = st.sidebar.selectbox("Difficulty", diff_opts)

    if sel_diff != "All":
        filtered = filtered[filtered["difficulty"] == sel_diff]

    if filtered.empty:
        st.warning("No questions match the current filters. Try changing the filters.")
        return

    # Number of questions for this quiz
    max_q = len(filtered)
    default_q = min(5, max_q)
    num_questions = st.sidebar.number_input(
        "Number of questions in this quiz",
        min_value=1,
        max_value=max_q,
        value=default_q,
        step=1,
    )

    # ---------- Initialise / Reset Quiz State ----------
    if "quiz_indices" not in st.session_state:
        init_quiz_state()

    # Quiz parameters that define a "session"
    current_params = {
        "class": sel_class,
        "chapter": sel_chapter,
        "difficulty": sel_diff,
        "num_questions": int(num_questions),
    }

    # If filters / num_questions changed -> reset quiz
    if st.session_state.quiz_params != current_params:
        #
