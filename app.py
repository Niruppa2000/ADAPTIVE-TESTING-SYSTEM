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
        # Sample question indices from filtered df
        all_indices = filtered.index.tolist()
        if len(all_indices) <= num_questions:
            chosen = all_indices
        else:
            chosen = random.sample(all_indices, num_questions)

        st.session_state.quiz_indices = chosen
        st.session_state.quiz_pos = 0
        st.session_state.score = 0
        st.session_state.total_answered = 0
        st.session_state.quiz_finished = False
        st.session_state.quiz_params = current_params

    # ---------- If quiz finished ----------
    if st.session_state.quiz_finished:
        st.subheader("Quiz Finished ✅")
        st.write(
            f"You answered **{st.session_state.total_answered}** question(s).\n\n"
            f"Correct: **{st.session_state.score}**\n\n"
        )
        if st.session_state.total_answered > 0:
            acc = st.session_state.score / st.session_state.total_answered
            st.write(f"Accuracy: **{acc * 100:.1f}%**")

        if st.button("Start a new quiz with current filters"):
            # Force re-init by clearing params
            st.session_state.quiz_params = None
            st.rerun()
        return

    # ---------- Get current question ----------
    # safety check
    if st.session_state.quiz_pos >= len(st.session_state.quiz_indices):
        st.session_state.quiz_finished = True
        st.rerun()
        return

    current_global_idx = st.session_state.quiz_indices[st.session_state.quiz_pos]
    q_row = filtered.loc[current_global_idx]

    # ---------- Sidebar Performance ----------
    st.sidebar.header("Your Performance (this quiz)")
    total = st.session_state.total_answered
    correct = st.session_state.score
    if total > 0:
        acc = correct / total
    else:
        acc = 0.0

    if acc >= 0.8:
        level = "Advanced"
    elif acc >= 0.5:
        level = "Intermediate"
    else:
        level = "Beginner"

    st.sidebar.write(f"Questions to attempt: {num_questions}")
    st.sidebar.write(f"Answered: {total}")
    st.sidebar.write(f"Correct: {correct}")
    st.sidebar.write(f"Accuracy: {acc*100:.1f}%")
    st.sidebar.write(f"Level: **{level}**")

    # ---------- Show Question ----------
    q_number = st.session_state.quiz_pos + 1
    st.subheader(f"Question {q_number} of {num_questions}")
    st.markdown(f"**Class {q_row['class']} | {q_row['chapter']}**")
    st.markdown(f"**Q:** {q_row['question']}")

    options = ["A", "B", "C", "D"]
    labels = [
        f"A. {q_row['A']}",
        f"B. {q_row['B']}",
        f"C. {q_row['C']}",
        f"D. {q_row['D']}",
    ]

    user_choice = st.radio(
        "Choose an option:",
        options=options,
        index=0,
        format_func=lambda x: labels[options.index(x)],
        key=f"q_radio_{q_number}",  # ensure different key each rerun
    )

    # ---------- Submit ----------
    if st.button("Submit"):
        st.session_state.total_answered += 1

        # Normalise answer letter
        correct_letter = str(q_row["answer"]).strip().upper()

        # Map letter -> option text
        option_map = {
            "A": q_row.get("A", ""),
            "B": q_row.get("B", ""),
            "C": q_row.get("C", ""),
            "D": q_row.get("D", ""),
        }
        correct_text = option_map.get(correct_letter, "")

        if user_choice == correct_letter:
            st.session_state.score += 1
            st.success(f"✅ Correct! ({correct_letter}. {correct_text})")
        else:
            st.error(f"❌ Incorrect. Correct answer: {correct_letter}. {correct_text}")

        st.info(f"Difficulty: **{q_row['difficulty']}**")

        # Move to next question or finish quiz
        st.session_state.quiz_pos += 1
        if st.session_state.quiz_pos >= len(st.session_state.quiz_indices):
            st.session_state.quiz_finished = True

        st.rerun()

    # ---------- Optional Metadata ----------
    with st.expander("Show question metadata"):
        st.write(f"Topic: {q_row.get('topic', '')}")
        st.write(f"Sources: {q_row.get('sources', '')}")


if __name__ == "__main__":
    main()
