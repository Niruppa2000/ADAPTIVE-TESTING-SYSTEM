import streamlit as st
import pandas as pd
import random
import os

# Path to your generated CSV inside the repo
CSV_PATH = os.path.join("data", "NCERT_MCQs_with_difficulty_by_chapter.csv")


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    # Ensure class is string
    df["class"] = df["class"].astype(str)
    # Fill any missing chapter names to avoid issues
    df["chapter"] = df["chapter"].fillna("Unknown Chapter")
    df["difficulty"] = df["difficulty"].fillna("Medium")
    return df


def main():
    st.title("NCERT MCQ Practice (Classes 6–10)")
    st.write(
        "MCQs auto-generated from NCERT (Classes 6–10) using a RAG pipeline with FLAN-T5 "
        "and Sentence-BERT. This app lets you practice and track your performance."
    )

    # Load data
    df = load_data()

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    class_options = ["All"] + sorted(df["class"].unique().tolist())
    selected_class = st.sidebar.selectbox("Select Class", class_options)

    filtered_df = df.copy()
    if selected_class != "All":
        filtered_df = filtered_df[filtered_df["class"] == selected_class]

    chapter_options = ["All"] + sorted(filtered_df["chapter"].unique().tolist())
    selected_chapter = st.sidebar.selectbox("Select Chapter", chapter_options)

    if selected_chapter != "All":
        filtered_df = filtered_df[filtered_df["chapter"] == selected_chapter]

    difficulty_options = ["All", "Easy", "Medium", "Hard"]
    selected_diff = st.sidebar.selectbox("Select Difficulty", difficulty_options)

    if selected_diff != "All":
        filtered_df = filtered_df[filtered_df["difficulty"] == selected_diff]

    if filtered_df.empty:
        st.warning("No questions match the current filters. Try changing filters.")
        return

    # --- Session State for Quiz ---
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "total" not in st.session_state:
        st.session_state.total = 0
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = random.randint(0, len(filtered_df) - 1)

    # --- Sidebar Performance Metrics ---
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
    st.sidebar.write(f"Accuracy: {acc * 100:.1f}%")
    st.sidebar.write(f"Level: **{level}**")

    # --- Display Current Question ---
    q_row = filtered_df.iloc[st.session_state.current_idx]

    st.subheader(f"Class {q_row['class']} | {q_row['chapter']}")
    st.markdown(f"**Q:** {q_row['question']}")

    options = ["A", "B", "C", "D"]
    labels = [f"A. {q_row['A']}",
              f"B. {q_row['B']}",
              f"C. {q_row['C']}",
              f"D. {q_row['D']}"]

    user_choice = st.radio(
        "Choose an option:",
        options=options,
        index=0,
        format_func=lambda x: labels[options.index(x)],
    )

    if st.button("Submit"):
        st.session_state.total += 1

# Normalise the stored answer letter
       correct = str(q_row["answer"]).strip().upper()

# Map options safely
     option_map = {
       "A": q_row.get("A", ""),
       "B": q_row.get("B", ""),
       "C": q_row.get("C", ""),
       "D": q_row.get("D", ""),
}

      correct_text = option_map.get(correct, "")

if user_choice == correct:
    st.session_state.score += 1
    st.success(f"✅ Correct! ({correct}. {correct_text})")
else:
    st.error(f"❌ Incorrect. Correct answer: {correct}. {correct_text})")

        # Pick next question randomly from filtered set
        st.session_state.current_idx = random.randint(0, len(filtered_df) - 1)
        st.experimental_rerun()

    # Optional: show sources used in generation
    with st.expander("Show metadata for this question"):
        st.write(f"Topic: {q_row.get('topic', '')}")
        st.write(f"Sources: {q_row.get('sources', '')}")


if __name__ == "__main__":
    main()

