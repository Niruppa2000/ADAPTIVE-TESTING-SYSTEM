import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ======================
# 1. Cached loaders
# ======================

@st.cache_data
def load_data():
    questions = pd.read_csv("questions.csv")
    attempts = pd.read_csv("attempts.csv")
    topics = sorted(questions["topic"].unique().tolist())
    difficulty_map = {1: "Easy", 2: "Medium", 3: "Hard"}
    return questions, attempts, topics, difficulty_map


@st.cache_resource
def load_models():
    rf_clf = joblib.load("models/difficulty_model.pkl")
    le_topic = joblib.load("models/topic_label_encoder.pkl")
    return rf_clf, le_topic


@st.cache_data
def build_question_stats():
    """
    Build per-question statistics and use the RandomForest model
    to predict difficulty from student performance data.

    NOTE: No arguments here, so cache hashing is safe.
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

    # build a lookup dict
    q_map = {}
    for _, row in q_stats.iterrows():
        q_map[int(row["question_id"])] = {
            "avg_correct": float(row["avg_correct"]),
            "avg_time": float(row["avg_time"]),
            "topic": row["topic"],
            "topic_encoded": int(row["topic_encoded"]),
            "predicted_difficulty": int(row["predicted_difficulty"])
        }
    return q_map


# ======================
# 2. Helper functions
# ======================

def init_session_state(topics, topic_default_diff, num_questions, student_id):
    st.session_state.student_id = student_id
    st.session_state.num_questions = num_questions
    st.session_state.asked_qids = set()
    st.session_state.correct_count = 0
    st.session_state.current_index = 0
    # difficulty per topic, starting from model-based default
    st.session_state.topic_difficulty = dict(topic_default_diff)
    st.session_state.responses = []
    st.session_state.current_question = None
    st.session_state.quiz_finished = False
    st.session_state.start_time = None
    st.session_state.initialized = True


def pick_topic(topics):
    """
    If no responses yet -> random topic.
    Else -> choose weakest topic (lowest accuracy).
    """
    if not st.session_state.responses:
        return random.choice(topics)
    df = pd.DataFrame(st.session_state.responses)
    topic_perf = df.groupby("topic")["is_correct"].mean().to_dict()

    # fill missing topics as strong
    for t in topics:
        topic_perf.setdefault(t, 1.0)

    # weakest topic first
    return min(topic_perf, key=topic_perf.get)


def pick_question(questions_df, topics, difficulty_map):
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


def update_difficulty(topic, is_correct, time_taken, difficulty_level):
    """
    Simple rule-based adaptation:
    - If correct & fast -> increase difficulty up to 3
    - If wrong -> decrease difficulty down to 1
    """
    cur = st.session_state.topic_difficulty.get(topic, 2)
    if is_correct and time_taken < 45 and difficulty_level < 3:
        cur = cur + 1
    elif (not is_correct) and difficulty_level > 1:
        cur = cur - 1
    st.session_state.topic_difficulty[topic] = cur


def build_topic_default_difficulty(question_stats_map, topics):
    """
    Use model-predicted difficulty to create
    default starting level per topic.
    """
    topic_default = {}
    for t in topics:
        preds = [
            v["predicted_difficulty"]
            for k, v in question_stats_map.items()
            if v["topic"] == t
        ]
        if preds:
            topic_default[t] = int(round(np.mean(preds)))
        else:
            topic_default[t] = 2
    return topic_default


def knowledge_gap_summary():
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
    return summary


# ======================
# 3. Streamlit UI
# ======================

def main():
    st.set_page_config(page_title="AI Adaptive Testing System", layout="centered")

    st.title("🧠 AI-Powered Adaptive Testing System")
    st.write(
        """
        This app adjusts question difficulty in real-time based on your answers,
        and provides topic-wise knowledge gap analysis at the end.
        """
    )

    # ---- Load data and models ----
    questions_df, attempts_df, topics, difficulty_map = load_data()
    _ = load_models()  # models used inside build_question_stats
    question_stats_map = build_question_stats()
    topic_default_diff = build_topic_default_difficulty(question_stats_map, topics)

    # ---- Sidebar: configuration ----
    st.sidebar.header("Test Configuration")
    student_id = st.sidebar.text_input("Student ID", value="demo_user")
    num_questions = st.sidebar.number_input(
        "Number of questions",
        min_value=3,
        max_value=30,
        value=10,
        step=1,
    )

    if "initialized" not in st.session_state:
        init_session_state(topics, topic_default_diff, num_questions, student_id)

    if st.sidebar.button("🔁 Start / Restart Test"):
        init_session_state(topics, topic_default_diff, num_questions, student_id)
        # first question
        q = pick_question(questions_df, topics, difficulty_map)
        if q is not None:
            st.session_state.current_question = q
            st.session_state.asked_qids.add(int(q["question_id"]))
            st.session_state.start_time = time.time()

    # If not initialized or no current question yet, show info
    if not st.session_state.initialized or st.session_state.current_question is None:
        st.info("Click **Start / Restart Test** in the sidebar to begin.")
        return

    # ---- If quiz finished, show summary ----
    if st.session_state.quiz_finished:
        st.success("Test completed! 🎉")
        st.write(f"**Final Score:** {st.session_state.correct_count} / {st.session_state.num_questions}")

        summary = knowledge_gap_summary()
        st.subheader("📊 Knowledge Gap Analysis (Topic-wise Accuracy)")
        if summary:
            for topic, acc, status in summary:
                st.write(f"- **{topic}**: {acc}% ({status})")
        else:
            st.write("No responses recorded.")

        with st.expander("Show detailed responses"):
            st.dataframe(pd.DataFrame(st.session_state.responses))

        return

    # ---- Display current question ----
    q = st.session_state.current_question
    qid = int(q["question_id"])
    topic = q["topic"]
    diff_level = int(q["difficulty_level"])
    diff_label = difficulty_map[diff_level]

    st.markdown(f"**Student:** `{st.session_state.student_id}`")
    st.markdown(
        f"**Question {st.session_state.current_index + 1} / {st.session_state.num_questions}**"
    )
    st.markdown(f"**Topic:** {topic} | **Difficulty:** {diff_label}")

    # show model-predicted difficulty for information (optional)
    q_stats = question_stats_map.get(qid)
    if q_stats is not None:
        st.caption(
            f"Model-predicted difficulty level (from past data): "
            f"{q_stats['predicted_difficulty']}"
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

    submitted = st.button("Submit Answer ✅")

    if submitted:
        end_time = time.time()
        time_taken = max(1.0, end_time - st.session_state.start_time)

        is_correct = int(answer == q["correct_option"])
        if is_correct:
            st.success("Correct! 🎯")
            st.session_state.correct_count += 1
        else:
            st.error(f"Wrong. Correct answer is **{q['correct_option']}**.")

        # store response
        st.session_state.responses.append(
            {
                "question_id": qid,
                "topic": topic,
                "difficulty_level": diff_level,
                "selected_option": answer,
                "correct_option": q["correct_option"],
                "is_correct": is_correct,
                "time_taken": round(time_taken, 1),
            }
        )

        # update difficulty for this topic
        update_difficulty(topic, is_correct, time_taken, diff_level)

        st.session_state.current_index += 1

        if st.session_state.current_index >= st.session_state.num_questions:
            st.session_state.quiz_finished = True
            st.rerun()
        else:
            # pick next question
            q_next = pick_question(questions_df, topics, difficulty_map)
            if q_next is None:
                st.session_state.quiz_finished = True
            else:
                st.session_state.current_question = q_next
                st.session_state.asked_qids.add(int(q_next["question_id"]))
                st.session_state.start_time = time.time()
            st.rerun()


if __name__ == "__main__":
    main()

