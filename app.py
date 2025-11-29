import time
import random

import numpy as np
import pandas as pd
import streamlit as st
import joblib

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon once
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()


# ======================
# 1. Cached Loaders
# ======================

@st.cache_data
def load_data():
    """
    Load questions.csv and attempts.csv.
    """
    questions = pd.read_csv("questions.csv")
    attempts = pd.read_csv("attempts.csv")

    topics = sorted(questions["topic"].unique().tolist())
    difficulty_map = {1: "Easy", 2: "Medium", 3: "Hard"}

    return questions, attempts, topics, difficulty_map


@st.cache_resource
def load_models():
    """
    Load RandomForest difficulty model and LabelEncoder.
    """
    rf_clf = joblib.load("models/difficulty_model.pkl")
    le_topic = joblib.load("models/topic_label_encoder.pkl")
    return rf_clf, le_topic


@st.cache_data
def build_question_stats():
    """
    Build average correctness/time per question and get predicted difficulty
    using the RandomForest model.
    """
    questions_df, attempts_df, _, _ = load_data()
    rf_clf, le_topic = load_models()

    q_stats = attempts_df.groupby("question_id").agg(
        avg_correct=("is_correct", "mean"),
        avg_time=("time_taken_sec", "mean")
    ).reset_index()

    q_stats = q_stats.merge(
        questions_df[["question_id", "topic"]],
        on="question_id",
        how="left"
    )

    q_stats["topic_encoded"] = le_topic.transform(q_stats["topic"])

    X = q_stats[["avg_correct", "avg_time", "topic_encoded"]]
    q_stats["predicted_difficulty"] = rf_clf.predict(X)

    result = {}
    for _, row in q_stats.iterrows():
        result[int(row["question_id"])] = {
            "avg_correct": float(row["avg_correct"]),
            "avg_time": float(row["avg_time"]),
            "topic": row["topic"],
            "topic_encoded": int(row["topic_encoded"]),
            "predicted_difficulty": int(row["predicted_difficulty"])
        }
    return result


# ======================
# 2. Helper Functions
# ======================

def init_session(topics, starting_difficulties, total_questions, student_id):
    st.session_state.student_id = student_id
    st.session_state.asked_qids = set()
    st.session_state.correct_count = 0
    st.session_state.current_index = 0
    st.session_state.total_questions = total_questions
    st.session_state.responses = []
    st.session_state.current_question = None
    st.session_state.quiz_finished = False
    st.session_state.start_time = None
    st.session_state.initialized = True

    st.session_state.topic_difficulty = dict(starting_difficulties)


def pick_topic(topics):
    """
    Prefer weakest topic so far, else random.
    """
    if not st.session_state.responses:
        return random.choice(topics)

    df = pd.DataFrame(st.session_state.responses)
    topic_perf = df.groupby("topic")["is_correct"].mean().to_dict()

    for t in topics:
        topic_perf.setdefault(t, 1.0)

    weakest_two = sorted(topic_perf, key=topic_perf.get)[:2]
    return random.choice(weakest_two)


def pick_question(questions_df, topics):
    """
    Pick a question from chosen topic with desired difficulty.
    """
    topic = pick_topic(topics)
    desired_diff = st.session_state.topic_difficulty.get(topic, 2)

    pool = questions_df[
        (questions_df["topic"] == topic) &
        (questions_df["difficulty_level"] == desired_diff) &
        (~questions_df["question_id"].isin(st.session_state.asked_qids))
    ]

    if pool.empty:
        pool = questions_df[
            (questions_df["topic"] == topic) &
            (~questions_df["question_id"].isin(st.session_state.asked_qids))
        ]

    if pool.empty:
        return None

    row = pool.sample(1).iloc[0]
    return row


def update_difficulty(topic, is_correct, time_taken, difficulty_level, sentiment_simple):
    cur = st.session_state.topic_difficulty.get(topic, 2)

    if is_correct:
        if sentiment_simple == "positive" and time_taken < 45:
            cur = min(3, cur + 1)
        elif sentiment_simple == "neutral":
            if random.random() < 0.4:
                cur = min(3, cur + 1)
    else:
        cur = max(1, cur - 1)

    st.session_state.topic_difficulty[topic] = cur


def interpret_sentiment(feedback_text):
    """Use VADER sentiment analysis."""
    if not feedback_text.strip():
        return "neutral"

    score = sia.polarity_scores(feedback_text)
    comp = score["compound"]

    if comp >= 0.5:
        return "positive"
    elif comp <= -0.3:
        return "negative"
    else:
        return "neutral"


def compute_knowledge_gaps():
    df = pd.DataFrame(st.session_state.responses)
    if df.empty:
        return []

    topic_perf = df.groupby("topic")["is_correct"].mean().to_dict()

    summary = []
    for topic, acc in topic_perf.items():
        if acc >= 0.7:
            status = "Strong"
        elif acc >= 0.4:
            status = "Moderate"
        else:
            status = "Weak"
        summary.append((topic, round(acc * 100, 1), status))

    summary.sort(key=lambda x: x[1])
    return summary


def starting_difficulty_map(q_stats_map, topics):
    d = {}
    for t in topics:
        preds = [
            v["predicted_difficulty"]
            for _, v in q_stats_map.items()
            if v["topic"] == t
        ]
        if preds:
            d[t] = int(np.mean(preds))
        else:
            d[t] = 2
    return d


# ======================
# 3. Streamlit UI
# ======================

def main():
    st.set_page_config(page_title="NCERT Adaptive Maths Tutor", layout="centered")

    st.title("📘 NCERT Maths – Adaptive Learning System")
    st.write(
        """
        This AI tutor generates maths questions from **NCERT Class 6–10**
        and adapts based on:
        - Your performance  
        - Your speed  
        - Your feedback sentiment  
        """
    )

    questions_df, attempts_df, topics, difficulty_map = load_data()
    q_stats_map = build_question_stats()
    topic_start_diff = starting_difficulty_map(q_stats_map, topics)

    # Sidebar
    st.sidebar.header("Settings")
    student_id = st.sidebar.text_input("Student ID", "student1")
    total_questions = st.sidebar.number_input(
        "Number of Questions", min_value=3, max_value=50, value=10
    )

    if "initialized" not in st.session_state:
        init_session(topics, topic_start_diff, total_questions, student_id)

    if st.sidebar.button("🔁 Start / Restart Test"):
        init_session(topics, topic_start_diff, total_questions, student_id)
        first_q = pick_question(questions_df, topics)
        if first_q is not None:
            st.session_state.current_question = first_q
            st.session_state.asked_qids.add(int(first_q["question_id"]))
            st.session_state.start_time = time.time()

    if (
        not st.session_state.initialized
        or st.session_state.current_question is None
    ):
        st.info("Click **Start / Restart Test** to begin.")
        return

    if st.session_state.quiz_finished:
        st.success("🎉 Test Completed!")
        st.write(
            f"**Final Score:** {st.session_state.correct_count} / "
            f"{st.session_state.total_questions}"
        )

        st.subheader("📊 Knowledge Gap Analysis")
        gaps = compute_knowledge_gaps()
        if gaps:
            for topic, acc, status in gaps:
                st.write(f"- **{topic}**: {acc}% ({status})")
        else:
            st.write("No data available.")

        st.write("### Detailed Responses")
        st.dataframe(pd.DataFrame(st.session_state.responses))
        return

    # ---------------------------------
    # Show Current Question
    # ---------------------------------
    q = st.session_state.current_question
    qid = int(q["question_id"])

    topic_str = q["topic"]
    diff_level = int(q["difficulty_level"])
    diff_label = difficulty_map.get(diff_level, "Unknown")

    st.markdown(
        f"**Question {st.session_state.current_index + 1}** / "
        f"{st.session_state.total_questions}"
    )
    st.markdown(f"**Topic:** {topic_str}")
    st.markdown(f"**Difficulty:** {diff_label}")
    st.write("---")

    st.write(q["question_text"])

    options = {
        "A": q["option_A"],
        "B": q["option_B"],
        "C": q["option_C"],
        "D": q["option_D"],
    }

    answer = st.radio(
        "Choose your answer:",
        list(options.keys()),
        format_func=lambda k: f"{k}) {options[k]}",
        index=0,
        key=f"opt_{qid}_{st.session_state.current_index}",
    )

    feedback = st.text_input(
        "Optional: Tell us how this question felt (easy, confusing, hard, etc.):",
        key=f"feed_{qid}_{st.session_state.current_index}",
    )

    if st.button("Submit Answer ✓"):
        end = time.time()
        time_taken = max(1.0, end - st.session_state.start_time)

        is_correct = int(answer == q["correct_option"])
        if is_correct:
            st.success("Correct!")
            st.session_state.correct_count += 1
        else:
            st.error(f"Incorrect. Correct answer: **{q['correct_option']}**")

        sentiment_simple = interpret_sentiment(feedback)

        # Save response
        st.session_state.responses.append(
            {
                "question_id": qid,
                "topic": topic_str,
                "difficulty_level": diff_level,
                "selected_option": answer,
                "correct_option": q["correct_option"],
                "is_correct": is_correct,
                "time_taken": round(time_taken, 2),
                "feedback": feedback,
                "sentiment": sentiment_simple,
            }
        )

        # Update adaptive difficulty
        update_difficulty(topic_str, is_correct, time_taken, diff_level, sentiment_simple)

        # Move to next question
        st.session_state.current_index += 1

        if st.session_state.current_index >= st.session_state.total_questions:
            st.session_state.quiz_finished = True
            st.rerun()
        else:
            next_q = pick_question(questions_df, topics)
            if next_q is None:
                st.session_state.quiz_finished = True
            else:
                st.session_state.current_question = next_q
                st.session_state.asked_qids.add(int(next_q["question_id"]))
                st.session_state.start_time = time.time()
            st.rerun()


if __name__ == "__main__":
    main()

