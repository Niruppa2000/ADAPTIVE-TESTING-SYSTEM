import streamlit as st
import pandas as pd
import random
import os

# ----- CSV PATH -----
CSV_PATH = os.path.join("data", "NCERT_MCQs_with_difficulty_by_chapter.csv")


@st.cache_data
def load_data():
    # Load CSV
    df = pd.read_csv(CSV_PATH)

    # Normalise some columns
    df["class"] = df["class"].astype(str)
    df["chapter"] = df["chapter"].fillna("Unknown Chapter")
    df["difficulty"] = df["difficulty"].fillna("Medium")
    # Clean answer column: ensure A/B/C/D with no extra spaces
    df["answer"] = df["answer"].astype(str).str.strip().str.upper()

    return df


def main():
    st.title("NCERT MCQ Practice (Classes 6–10)")
    st.write(
        "MCQs loaded from `data/NCERT_MCQs_with_difficulty_by_chapter.csv`.\n\n"
        "Filter by class, chapter, and difficulty, then practice and see your performance."
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

    # ---------- Session State ----------
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "total" not in st.session_state:
        st.session_state.total = 0
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = random.randint(0, len(filtered) - 1)

    # ---------- Sidebar Performance ----------
    st.sidebar.header("Your Performance")
    if st.session_state.total > 0:
        acc = st.session_state.score / st.session_state.total
    else:
        acc = 0.0

    if acc >= 0.8:
        level = "Advanced"
    elif acc >= 0.5:
        level = "Intermediate"
    else:
        level = "Beginner"

    st.sidebar.write(f"Questions answered: {st.session_state.total}")
    st.sidebar.write(f"Correct: {st.session_state.score}")
    st.sidebar.write(f"Accuracy: {acc*100:.1f}%")
    st.sidebar.write(f"Level: **{level}**")

    # ---------- Current Question ----------
    q_row = filtered.iloc[st.session_state.current_idx]

    st.subheader(f"Class {q_row['class']} | {q_row['chapter']}")
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
    )

    # ---------- Submit ----------
    if st.button("Submit"):
        st.session_state.total += 1

        # Normalise answer letter
        correct_letter = str(q_row["answer"]).strip().upper()

        # Safe mapping from letter -> text
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

        # Next question
        st.session_state.current_idx = random.randint(0, len(filtered) - 1)
        st.rerun()  # <-- updated from st.experimental_rerun()

    # ---------- Optional: show metadata ----------
    with st.expander("Show question metadata"):
        st.write(f"Topic: {q_row.get('topic', '')}")
        st.write(f"Sources: {q_row.get('sources', '')}")


if __name__ == "__main__":
    main()
