import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
from pathlib import Path

# ---------- PATHS ----------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

SBERT_PATH = MODEL_DIR / "sbert_qg"
DIFF_MODEL_PATH = MODEL_DIR / "difficulty_clf.joblib"
QUESTION_BANK_PATH = DATA_DIR / "question_bank.parquet"

DIFF_MAP = {"easy": 0, "medium": 1, "hard": 2}
INV_DIFF_MAP = {v: k for k, v in DIFF_MAP.items()}


# ---------- CACHED LOADERS ----------

@st.cache_resource
def load_sbert():
    """Load fine-tuned Sentence-BERT model (once)."""
    model = SentenceTransformer(str(SBERT_PATH))
    return model


@st.cache_resource
def load_difficulty_model():
    """Load difficulty classifier (RandomForest)."""
    clf = joblib.load(DIFF_MODEL_PATH)
    return clf


@st.cache_resource
def load_question_bank_and_embeddings():
    """
    Load question bank and pre-compute embeddings for each question
    (for difficulty prediction / similarity if needed).
    """
    if QUESTION_BANK_PATH.suffix == ".parquet":
        df = pd.read_parquet(QUESTION_BANK_PATH)
    else:
        df = pd.read_csv(QUESTION_BANK_PATH)

    # Ensure difficulty id
    if "difficulty" not in df.columns:
        raise ValueError("question_bank must contain 'difficulty' column.")
    df["diff_id"] = df["difficulty"].map(DIFF_MAP)

    sbert = load_sbert()
    # Precompute question embeddings once
    q_embs = sbert.encode(
        df["question"].tolist(),
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    df["q_emb"] = list(q_embs)
    return df


# ---------- ADAPTIVE LOGIC ----------

def choose_initial_question(df: pd.DataFrame, class_level: int, asked_ids):
    """Start from medium difficulty question."""
    candidates = df[df["difficulty"] == "medium"]
    candidates = candidates[~candidates.index.isin(asked_ids)]
    if candidates.empty:
        candidates = df[~df.index.isin(asked_ids)]
    row = candidates.sample(1).iloc[0]
    return int(row.name)


def choose_next_question(df: pd.DataFrame, prev_correct: bool, prev_diff_id: int, asked_ids):
    """
    Adaptive rule:
    - correct  -> difficulty same or +1
    - incorrect-> difficulty same or -1
    """
    if prev_correct:
        target_diff = min(prev_diff_id + 1, 2)
    else:
        target_diff = max(prev_diff_id - 1, 0)

    candidates = df[df["diff_id"] == target_diff]
    candidates = candidates[~candidates.index.isin(asked_ids)]
    if candidates.empty:
        candidates = df[~df.index.isin(asked_ids)]
        if candidates.empty:
            return None
    row = candidates.sample(1).iloc[0]
    return int(row.name)


def evaluate_answer(student_answer: str, correct_answer: str, sbert: SentenceTransformer, threshold: float = 0.6):
    """
    Semantic similarity scoring using SBERT.
    Returns (bool_is_correct, similarity_score).
    """
    emb_student = sbert.encode([student_answer], convert_to_numpy=True)[0]
    emb_correct = sbert.encode([correct_answer], convert_to_numpy=True)[0]

    sim = np.dot(emb_student, emb_correct) / (
        np.linalg.norm(emb_student) * np.linalg.norm(emb_correct) + 1e-8
    )
    is_correct = bool(sim >= threshold)
    return is_correct, float(sim)


def predict_difficulty(question: str, sbert: SentenceTransformer, clf) -> str:
    """Predict difficulty of a custom question typed by user."""
    emb = sbert.encode([question], convert_to_numpy=True)
    pred_id = int(clf.predict(emb)[0])
    return INV_DIFF_MAP.get(pred_id, "unknown")


# ---------- UI HELPERS ----------

def init_session_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.student_name = ""
        st.session_state.class_level = None
        st.session_state.num_questions = 10

        st.session_state.current_qid = None
        st.session_state.score = 0
        st.session_state.num_attempted = 0
        st.session_state.asked_ids = []
        st.session_state.history = []  # list of dicts
        st.session_state.test_started = False
        st.session_state.test_finished = False


def start_test(df):
    s_name = st.session_state.student_name.strip()
    c_level = st.session_state.class_level
    if not s_name or c_level is None:
        st.warning("Please enter your name and class level to start.")
        return

    st.session_state.score = 0
    st.session_state.num_attempted = 0
    st.session_state.asked_ids = []
    st.session_state.history = []
    st.session_state.test_finished = False
    st.session_state.test_started = True

    first_qid = choose_initial_question(df, c_level, st.session_state.asked_ids)
    st.session_state.current_qid = first_qid
    st.session_state.asked_ids.append(first_qid)


def finish_test():
    st.session_state.test_finished = True
    st.session_state.test_started = False


# ---------- MAIN APP ----------

def main():
    st.set_page_config(page_title="NCERT AI-Powered Adaptive Test", layout="wide")
    init_session_state()

    st.title("📚 AI-Powered Adaptive Testing System (NCERT Social Science)")
    st.write(
        "This app uses a fine-tuned **Sentence-BERT** model and a question "
        "bank generated from NCERT Social Science textbooks (Class 6–12). "
        "It adapts question difficulty based on your performance in real time."
    )

    # Load models & data
    with st.spinner("Loading models and question bank..."):
        df_q = load_question_bank_and_embeddings()
        sbert = load_sbert()
        diff_clf = load_difficulty_model()

    # ----- SIDEBAR: User info & control -----
    with st.sidebar:
        st.header("🧑‍🎓 Test Settings")

        st.text_input(
            "Student name",
            key="student_name",
            placeholder="Enter your name",
        )
        st.selectbox(
            "Class level",
            options=list(range(6, 13)),
            key="class_level",
        )
        st.number_input(
            "Number of questions in test",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            key="num_questions",
        )

        if st.button("Start / Restart Test"):
            start_test(df_q)

        if st.button("Finish Test Now"):
            if st.session_state.test_started:
                finish_test()

        st.markdown("---")
        st.subheader("🔍 Difficulty Tester")
        custom_q = st.text_area(
            "Type a custom question to estimate its difficulty:",
            height=80,
        )
        if st.button("Predict Difficulty"):
            if custom_q.strip():
                diff = predict_difficulty(custom_q.strip(), sbert, diff_clf)
                st.info(f"Predicted difficulty: **{diff.upper()}**")
            else:
                st.warning("Please enter a question first.")

    # ----- MAIN PANEL: Test flow -----

    # Show summary if finished
    if st.session_state.test_finished:
        show_summary(df_q)
        return

    # If test not started yet
    if not st.session_state.test_started:
        st.info("Set your name, class, and click **Start / Restart Test** in the sidebar to begin.")
        return

    # There is an active test
    qid = st.session_state.current_qid
    if qid is None:
        st.error("No current question. Click 'Start / Restart Test' to begin.")
        return

    row = df_q.loc[qid]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Question {st.session_state.num_attempted + 1} of {st.session_state.num_questions}")
        st.markdown(f"**Student:** {st.session_state.student_name}  |  **Class:** {st.session_state.class_level}")
        st.markdown(f"**Current difficulty:** `{row['difficulty']}`")

        st.markdown("##### Context")
        st.write(row["context"])

        st.markdown("##### Question")
        st.write(row["question"])

        with st.form(key="answer_form", clear_on_submit=True):
            ans = st.text_area("Your answer:", height=120)
            submitted = st.form_submit_button("Submit answer")

        if submitted:
            if not ans.strip():
                st.warning("Please type an answer before submitting.")
            else:
                is_correct, sim = evaluate_answer(ans.strip(), row["answer"], sbert)
                st.session_state.num_attempted += 1
                if is_correct:
                    st.session_state.score += 1
                    st.success(f"Correct! (similarity: {sim:.2f})")
                else:
                    st.error(f"Not quite. (similarity: {sim:.2f})")
                    with st.expander("Show reference answer"):
                        st.write(row["answer"])

                # Save history
                st.session_state.history.append(
                    {
                        "qid": int(qid),
                        "difficulty": row["difficulty"],
                        "diff_id": int(row["diff_id"]),
                        "correct": bool(is_correct),
                        "similarity": float(sim),
                    }
                )

                # Decide next question or finish
                if st.session_state.num_attempted >= st.session_state.num_questions:
                    finish_test()
                    st.experimental_rerun()
                else:
                    next_qid = choose_next_question(
                        df_q,
                        prev_correct=is_correct,
                        prev_diff_id=int(row["diff_id"]),
                        asked_ids=st.session_state.asked_ids,
                    )
                    if next_qid is None:
                        finish_test()
                        st.experimental_rerun()
                    else:
                        st.session_state.current_qid = next_qid
                        st.session_state.asked_ids.append(next_qid)
                        st.experimental_rerun()

    with col2:
        st.subheader("📈 Live Stats")
        attempted = st.session_state.num_attempted
        score = st.session_state.score
        if attempted > 0:
            acc = 100.0 * score / attempted
        else:
            acc = 0.0
        st.metric("Questions attempted", attempted)
        st.metric("Score", score)
        st.metric("Accuracy (%)", f"{acc:.1f}")

        # difficulty-wise performance
        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history)
            st.markdown("##### Performance by difficulty")
            for diff in ["easy", "medium", "hard"]:
                sub = hist_df[hist_df["difficulty"] == diff]
                if len(sub) == 0:
                    st.write(f"- {diff}: N/A")
                else:
                    acc_d = 100.0 * sub["correct"].mean()
                    st.write(f"- {diff}: {acc_d:.1f}% correct")


def show_summary(df_q: pd.DataFrame):
    st.header("📊 Test Summary")

    score = st.session_state.score
    attempted = st.session_state.num_attempted
    if attempted > 0:
        acc = 100.0 * score / attempted
    else:
        acc = 0.0

    st.markdown(f"**Student:** {st.session_state.student_name}")
    st.markdown(f"**Class:** {st.session_state.class_level}")
    st.markdown(f"**Questions attempted:** {attempted}")
    st.markdown(f"**Total correct:** {score}")
    st.markdown(f"**Overall accuracy:** {acc:.1f}%")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)

        st.subheader("Knowledge Gap Analysis (by difficulty)")
        cols = st.columns(3)
        for i, diff in enumerate(["easy", "medium", "hard"]):
            sub = hist_df[hist_df["difficulty"] == diff]
            with cols[i]:
                if len(sub) == 0:
                    st.metric(diff.capitalize(), "N/A")
                else:
                    acc_d = 100.0 * sub["correct"].mean()
                    st.metric(diff.capitalize(), f"{acc_d:.1f}%")

        with st.expander("See detailed question history"):
            merged = hist_df.merge(
                df_q[["question", "answer"]],
                left_on="qid",
                right_index=True,
                how="left",
            )
            st.dataframe(merged)

    st.info("You can start another test from the sidebar by clicking **Start / Restart Test**.")


if __name__ == "__main__":
    main()

