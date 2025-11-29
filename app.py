import time
import random

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from transformers import pipeline


# ======================
# 1. Cached loaders
# ======================

@st.cache_data
def load_data():
    """
    Load questions and attempts generated from the NCERT pipeline.

    questions.csv must have at least:
      - question_id
      - topic
      - difficulty_level
      - question_text
      - option_A, option_B, option_C, option_D
      - correct_option

    attempts.csv must have:
      - student_id
      - question_id
      - topic
      - difficulty_level
      - is_correct
      - time_taken_sec
    """
    questions = pd.read_csv("questions.csv")
    attempts = pd.read_csv("attempts.csv")

    topics = sorted(questions["topic"].unique().tolist())
    difficulty_map = {1: "Easy", 2: "Medium", 3: "Hard"}

    return questions, attempts, topics, difficulty_map


@st.cache_resource
def load_models():
    """
    Load the RandomForest difficulty model and LabelEncoder for topics.
    """
    rf_clf = joblib.load("models/difficulty_model.pkl")
    le_topic = joblib.load("models/topic_label_encoder.pkl")
    return rf_clf, le_topic


@st.cache_resource
def load_sentiment_model():
    """
    Load a HuggingFace sentiment analysis pipeline.
    This runs locally, no API key needed.
    """
    return pipeline("sentiment-analysis")


@st.cache_data
def build_question_stats():
    """
    Build per-question statistics and use the RandomForest model
    to predict difficulty from historical performance data.

    This function does NOT take arguments (to avoid Streamlit
    unhashable cache issues).
    """
    questions_df, attempts_df, _, _ = load_data()
    rf_clf, le_topic = load_models()

    q_stats = attempts_df.groupby("question_id").agg(
        avg_correct=("is_correct", "mean"),
        avg_time=("time_taken_sec", "mean"),
    ).reset_index()

    q_stats = q_stats.merge(
        questions_df[["question_id", "topic"]],
        on="question_id",
        how="left",
    )

    # Encode topic
    q_stats["topic_encoded"] = le_topic.transform(q_stats["topic"])

    X = q_stats[["avg_correct", "avg_time", "topic_encoded"]]
    q_stats["predicted_difficulty"] = rf_clf.predict(X)

    # Build question_id -> stats map
    q_map = {}
    for _, row in q_stats.iterrows():
        q_map[int(row["question_id"])] = {
            "avg_correct": float(row["avg_correct"]),
            "avg_time": float(row["avg_time"]),
            "topic": row["topic"],
            "topic_encoded": int(row["topic_encoded"]),
            "predicted_difficulty": int(row["predicted_difficulty"]),
        }

    return q_map


# ======================
# 2. Helper functions
# ======================

def init_session_state(topics, topic_default_diff, num_questions, student_id):
    """Initialize the Streamlit session state for a fresh test."""
    st.session_state.student_id = student_id
    st.session_state.num_questions = num_questions
    st.session_state.asked_qids = set()
    st.session_state.correct_count = 0
    st.session_state.current_index = 0

    # Difficulty per topic (Easy=1, Medium=2, Hard=3).
    # Start from model-based defaults.
    st.session_state.topic_difficulty = dict(topic_default_diff)

    st.session_state.responses = []
    st.session_state.current_question = None
    st.session_state.quiz_finished = False
    st.session_state.start_time = None
    st.session_state.initialized = True


def pick_topic(topics):
    """
    - If no responses yet: pick a random topic
    - Else: choose randomly among the 2 weakest topics so far
      (based on accuracy), to focus on weaker areas but still
      keep variety.
    """
    if not st.session_state.responses:
        return random.choice(topics)

    df = pd.DataFrame(st.session_state.responses)
    topic_perf = df.groupby("topic")["is_correct"].mean().to_dict()

    # topics not seen yet are treated as "strong" initially
    for t in topics:
        topic_perf.setdefault(t, 1.0)

    # sort by accuracy (weakest first)
    weakest_two = sorted(topic_perf, key=topic_perf.get)[:2]
    return random.choice(weakest_two)


def pick_question(questions_df, topics, difficulty_map):
    """
    Pick the next question using:
      - topic chosen by pick_topic
      - current difficulty level for that topic
      - avoid already asked questions
    """
    topic = pick_topic(topics)
    desired_diff = st.session_state.topic_difficulty.get(topic, 2)

    pool = questions_df[
        (questions_df["topic"] == topic)
        & (questions_df["difficulty_level"] == desired_diff)
        & (~questions_df["question_id"].isin(st.session_state.asked_qids))
    ]

    if pool.empty:
        # fallback: any remaining question in this topic
        pool = questions_df[
            (questions_df["topic"] == topic)
            & (~questions_df["question_id"].isin(st.session_state.asked_qids))
        ]

    if pool.empty:
        return None

    row = pool.sample(1).iloc[0]
    return row


def interpret_sentiment_label(label: str, score: float) -> str:
    """
    Convert raw HF sentiment label + score into a simpler
    class: "positive" | "negative" | "neutral".
    """
    label = label.upper()
    if label == "POSITIVE" and score >= 0.7:
        return "positive"
    elif label == "NEGATIVE" and score >= 0.6:
        return "negative"
    else:
        return "neutral"


def update_difficulty(topic, is_correct, time_taken, difficulty_level, sentiment_simple):
    """
    Adaptive difficulty logic that uses both performance and sentiment:

    - If correct & positive & reasonably fast -> increase difficulty (up to 3).
    - If correct & neutral -> sometimes increase difficulty.
    - If correct but negative sentiment (e.g. "too hard") -> keep same level.
    - If wrong -> reduce difficulty (down to 1).
    """
    cur = st.session_state.topic_difficulty.get(topic, 2)

    if is_correct:
        if sentiment_simple == "positive" and time_taken < 45 and difficulty_level < 3:
            cur += 1
        elif sentiment_simple == "neutral" and difficulty_level < 3:
            if random.random() < 0.5:
                cur += 1
        else:
            # negative but correct: keep same
            cur = cur
    else:
        if difficulty_level > 1:
            cur -= 1

    st.session_state.topic_difficulty[topic] = max(1, min(3, cur))


def build_topic_default_difficulty(question_stats_map, topics):
    """
    Use model-predicted difficulty to set starting
    difficulty per topic.
    """
    topic_default = {}
    for t in topics:
        preds = [
            v["predicted_difficulty"]
            for _, v in question_stats_map.items()
            if v["topic"] == t
        ]
        if preds:
            topic_default[t] = int(round(np.mean(preds)))
        else:
            topic_default[t] = 2  # default Medium

    return topic_default


def knowledge_gap_summary():
    """
    Build topic-wise accuracy summary from session responses.
    """
    df = pd.DataFrame(st.session_state.responses)
    if df.empty:
        return []

    topic_perf = df.groupby("topic")["is_correct"].mean()
    summary = []
    for topic, acc in topic_perf.items():
        if acc >= 0.7:
            status = "Strong"
        elif acc >= 0.4:
            status = "Moderate"
        else:
            status = "Weak"
        summary.append((topic, round(acc * 100, 1), status))

    # sort by weakest first
    summary.sort(key=lambda x: x[1])
    return summary


# ======================
# 3. Streamlit UI
# ======================

def main():
    st.set_page_config(page_title="NCERT Adaptive Maths Tutor", layout="centered")

    st.title("📘 NCERT Maths – AI-Powered Adaptive Testing")
    st.write(
        """
        This app uses **NCERT Class 6–10 Mathematics** textbooks to generate questions.
        It adapts:
        - **Topic** (Algebra, Geometry, Arithmetic, etc.)
        - **Difficulty** (Easy / Medium / Hard)
        based on your answers **and** your feedback sentiment.
        """
    )

    # Load data and models
    questions_df, attempts_df, topics, difficulty_map = load_data()
    _ = load_models()
    sentiment_model = load_sentiment_model()
    question_stats_map = build_question_stats()
    topic_default_diff = build_topic_default_difficulty(question_stats_map, topics)

    # Sidebar configuration
    st.sidebar.header("Test Configuration")
    student_id = st.sidebar.text_input("Student ID", value="student1")
    num_questions = st.sidebar.number_input(
        "Number of questions",
        min_value=3,
        max_value=50,
        value=10,
        step=1,
    )

    if "initialized" not in st.session_state:
        init_session_state(topics, topic_default_diff, num_questions, student_id)

    if st.sidebar.button("🔁 Start / Restart Test"):
        init_session_state(topics, topic_default_diff, num_questions, student_id)
        first_q = pick_question(questions_df, topics, difficulty_map)
        if first_q is not None:
            st.session_state.current_question = first_q
            st.session_state.asked_qids.add(int(first_q["question_id"]))
            st.session_state.start_time = time.time()

    # If nothing initialized yet
    if not st.session_state.initialized or st.session_state.current_question is None:
        st.info("Click **Start / Restart Test** in the sidebar to begin.")
        return

    # If quiz finished, show summary
    if st.session_state.quiz_finished:
        st.success("✅ Test completed!")
        st.write(
            f"**Final Score:** {st.session_state.correct_count} / "
            f"{st.session_state.num_questions}"
        )

        st.subheader("📊 Knowledge Gap Analysis by Topic")
        summary = knowledge_gap_summary()
        if summary:
            for topic, acc, status in summary:
                st.write(f"- **{topic}**: {acc}% ({status})")
        else:
            st.write("No responses recorded.")

        with st.expander("Show detailed responses"):
            st.dataframe(pd.DataFrame(st.session_state.responses))

        return

    # ----- Show current question -----
    q = st.session_state.current_question
    qid = int(q["question_id"])
    topic_str = q["topic"]
    diff_level = int(q["difficulty_level"])
    diff_label = difficulty_map.get(diff_level, str(diff_level))

    # Split "Subtopic - Class X" if available
    if " - " in topic_str:
        subtopic, class_name = topic_str.split(" - ", 1)
    else:
        subtopic, class_name = topic_str, ""

    st.markdown(f"**Student:** `{st.session_state.student_id}`")
    st.markdown(
        f"**Question {st.session_state.current_index + 1} / "
        f"{st.session_state.num_questions}**"
    )
    st.markdown(
        f"**Subtopic:** {subtopic} &nbsp;&nbsp; "
        f"**{class_name}** &nbsp;&nbsp; "
        f"**Difficulty:** {diff_label}"
    )

    q_stats = question_stats_map.get(qid)
    if q_stats is not None:
        st.caption(
            "Model-predicted difficulty (from previous data): "
            f"Level {q_stats['predicted_difficulty']}"
        )

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
        options=list(options.keys()),
        format_func=lambda k: f"{k}) {options[k]}",
        index=0,
        key=f"answer_{qid}_{st.session_state.current_index}",
    )

    st.write("💬 Optional: Tell the system how this question felt for you.")
    feedback_text = st.text_input(
        "Example: 'This was easy', 'I am confused', 'Too hard', etc.",
        key=f"feedback_{qid}_{st.session_state.current_index}",
    )

    submitted = st.button("Submit Answer ✅")

    if submitted:
        end_time = time.time()
        time_taken = max(1.0, end_time - st.session_state.start_time)

        is_correct = int(answer == q["correct_option"])
        if is_correct:
            st.success("Correct! 🎯")
            st.session_state.correct_count += 1
        else:
            st.error(f"Wrong. Correct option is **{q['correct_option']}**.")

        # Sentiment analysis on feedback (if provided)
        if feedback_text.strip():
            try:
                sent_res = sentiment_model(feedback_text[:512])[0]
                sentiment_label = sent_res["label"]
                sentiment_score = float(sent_res["score"])
                sentiment_simple = interpret_sentiment_label(
                    sentiment_label, sentiment_score
                )
                st.caption(
                    f"Detected sentiment: {sentiment_label} "
                    f"({sentiment_score:.2f}) → {sentiment_simple}"
                )
            except Exception:
                sentiment_label = "NEUTRAL"
                sentiment_score = 0.0
                sentiment_simple = "neutral"
        else:
            sentiment_label = "NEUTRAL"
            sentiment_score = 0.0
            sentiment_simple = "neutral"

        # Store response
        st.session_state.responses.append(
            {
                "question_id": qid,
                "topic": topic_str,
                "difficulty_level": diff_level,
                "selected_option": answer,
                "correct_option": q["correct_option"],
                "is_correct": is_correct,
                "time_taken": round(time_taken, 1),
                "feedback_text": feedback_text,
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
                "sentiment_simple": sentiment_simple,
            }
        )

        # Update adaptive difficulty
        update_difficulty(
            topic_str,
            is_correct,
            time_taken,
            diff_level,
            sentiment_simple,
        )

        # Move to next question or finish
        st.session_state.current_index += 1

        if st.session_state.current_index >= st.session_state.num_questions:
            st.session_state.quiz_finished = True
            st.rerun()
        else:
            next_q = pick_question(questions_df, topics, difficulty_map)
            if next_q is None:
                st.session_state.quiz_finished = True
            else:
                st.session_state.current_question = next_q
                st.session_state.asked_qids.add(int(next_q["question_id"]))
                st.session_state.start_time = time.time()
            st.rerun()


if __name__ == "__main__":
    main()

