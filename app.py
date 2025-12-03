from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer


# ------------------------------
# PATHS & CONSTANTS
# ------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

SBERT_PATH = MODEL_DIR / "sbert_qg"
DIFF_MODEL_PATH = MODEL_DIR / "difficulty_clf.joblib"
QUESTION_BANK_PATH = DATA_DIR / "question_bank.parquet"

DIFF_MAP = {"easy": 0, "medium": 1, "hard": 2}
INV_DIFF_MAP = {v: k for k, v in DIFF_MAP.items()}


# ------------------------------
# MODEL LOADERS
# ------------------------------
@st.cache_resource
def load_sbert():
    """
    Load fine-tuned SBERT model if available.
    If missing or corrupted → fall back to base Sentence-BERT model.
    """
    try:
        if SBERT_PATH.exists() and any(SBERT_PATH.iterdir()):
            st.info(f"Loading fine-tuned SBERT from {SBERT_PATH}")
            return SentenceTransformer(str(SBERT_PATH), local_files_only=True)
        else:
            st.warning(
                f"Fine-tuned SBERT folder not found at {SBERT_PATH}.\n"
                "Using fallback base model: sentence-transformers/all-MiniLM-L6-v2"
            )
    except Exception:
        st.warning(
            "Fine-tuned SBERT exists but could not be loaded. "
            "Falling back to base model."
        )

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_difficulty_model():
    """Load RandomForest difficulty classifier if present."""
    if not DIFF_MODEL_PATH.exists():
        st.warning("⚠ Difficulty classifier not found. Predictions disabled.")
        return None
    try:
        return joblib.load(DIFF_MODEL_PATH)
    except Exception:
        st.error("Difficulty model file is corrupted.")
        return None


def _quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop obviously bad questions/answers:
    - question length 5–30 words
    - contains '?'
    - answer length 2–40 words
    """
    df = df.copy()
    q_words = df["question"].astype(str).str.split().str.len()
    a_words = df["answer"].astype(str).str.split().str.len()

    mask = (
        (q_words >= 5)
        & (q_words <= 30)
        & df["question"].str.contains(r"\?", regex=True)
        & (a_words >= 2)
        & (a_words <= 40)
    )
    df = df[mask]
    df = df.reset_index(drop=True)
    return df


@st.cache_resource
def load_question_bank():
    """
    Load and clean the question bank.

    Each row should have:
      - context  (paragraph from NCERT)
      - question (one-line question)
      - answer   (reference answer)
      - difficulty ('easy'/'medium'/'hard')
      - source_pdf (optional)
    """
    if not QUESTION_BANK_PATH.exists():
        st.error(
            f"`question_bank.parquet` missing at {QUESTION_BANK_PATH}.\n\n"
            "Upload it to your GitHub repo under `data/`."
        )
        st.stop()

    df = pd.read_parquet(QUESTION_BANK_PATH)

    required_cols = {"context", "question", "answer", "difficulty"}
    if not required_cols.issubset(set(df.columns)):
        st.error(
            f"Parquet file must contain columns: {required_cols}. "
            f"Found: {list(df.columns)}"
        )
        st.stop()

    df = _quality_filter(df)
    df["diff_id"] = df["difficulty"].map(DIFF_MAP)
    return df


# ------------------------------
# ADAPTIVE LOGIC
# ------------------------------
def choose_initial_question(df, class_level, asked_ids):
    """Start with a medium question if possible."""
    candidates = df[df["difficulty"] == "medium"]
    candidates = candidates[~candidates.index.isin(asked_ids)]
    if candidates.empty:
        candidates = df[~df.index.isin(asked_ids)]
    return int(candidates.sample(1).index[0])


def choose_next_question(df, prev_correct, prev_diff_id, asked_ids):
    """
    If correct → go harder; if wrong → go easier.
    Always avoid repeating already asked questions.
    """
    target = prev_diff_id + 1 if prev_correct else prev_diff_id - 1
    target = max(0, min(2, target))

    candidates = df[df["diff_id"] == target]
    candidates = candidates[~candidates.index.isin(asked_ids)]

    if candidates.empty:
        candidates = df[~df.index.isin(asked_ids)]

    if candidates.empty:
        return None

    return int(candidates.sample(1).index[0])


def evaluate_answer(student_answer, correct_answer, sbert, threshold=0.6):
    """
    Compare student answer to reference answer using SBERT cosine similarity.
    """
    v1 = sbert.encode([student_answer], convert_to_numpy=True)[0]
    v2 = sbert.encode([correct_answer], convert_to_numpy=True)[0]

    sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
    is_correct = sim >= threshold
    return is_correct, sim


def predict_difficulty(question, sbert, clf):
    if clf is None:
        return "unknown"
    emb = sbert.encode([question], convert_to_numpy=True)
    label = int(clf.predict(emb)[0])
    return INV_DIFF_MAP[label]


# ------------------------------
# SESSION STATE
# ------------------------------
def init_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.student_name = ""
        st.session_state.class_level = None
        st.session_state.num_questions = 10

        st.session_state.current_qid = None
        st.session_state.asked_ids = []
        st.session_state.score = 0
        st.session_state.num_attempted = 0
        st.session_state.test_started = False
        st.session_state.test_finished = False
        st.session_state.history = []


def start_test(df):
    st.session_state.score = 0
    st.session_state.num_attempted = 0
    st.session_state.asked_ids = []
    st.session_state.history = []

    st.session_state.test_started = True
    st.session_state.test_finished = False

    qid = choose_initial_question(df, st.session_state.class_level, [])
    st.session_state.current_qid = qid
    st.session_state.asked_ids.append(qid)


def finish_test():
    st.session_state.test_finished = True
    st.session_state.test_started = False


# ------------------------------
# MAIN APP
# ------------------------------
def main():
    st.set_page_config(page_title="Adaptive NCERT Testing System", layout="wide")
    init_state()

    st.title("📘 AI-Powered Adaptive Testing System (NCERT Social Science)")

    with st.spinner("Loading models and question bank…"):
        df_q = load_question_bank()
        sbert = load_sbert()
        diff_clf = load_difficulty_model()

    # SIDEBAR
    with st.sidebar:
        st.header("🧑‍🎓 Test Setup")
        st.text_input("Student Name", key="student_name")
        st.selectbox("Class Level", list(range(6, 13)), key="class_level")
        st.number_input("Number of Questions", 5, 30, 10, key="num_questions")

        if st.button("Start Test / Restart"):
            if not st.session_state.student_name or st.session_state.class_level is None:
                st.warning("Please enter your name and class level first.")
            else:
                start_test(df_q)
                st.rerun()

        if st.button("Finish Test"):
            finish_test()
            st.rerun()

        st.markdown("---")
        st.subheader("🔍 Test Custom Question Difficulty")

        custom_q = st.text_area("Enter a question")
        if st.button("Predict Difficulty"):
            if custom_q.strip():
                diff = predict_difficulty(custom_q, sbert, diff_clf)
                st.success(f"Predicted Difficulty: **{diff.upper()}**")
            else:
                st.warning("Enter a question first.")

    # MAIN CONTENT
    if st.session_state.test_finished:
        show_summary()
        return

    if not st.session_state.test_started:
        st.info("Enter your name and class level, then click **Start Test**.")
        return

    qid = st.session_state.current_qid
    row = df_q.loc[qid]

    st.subheader(
        f"Question {st.session_state.num_attempted + 1} of {st.session_state.num_questions}"
    )
    st.markdown(f"**Difficulty:** `{row['difficulty']}`")

    st.markdown("### Context")
    st.write(row["context"])

    st.markdown("### Question")
    # enforce one-line display visually (question text itself already filtered)
    st.write(row["question"].strip())

    with st.form("answer_form", clear_on_submit=True):
        ans = st.text_area("Your answer", height=140)
        submitted = st.form_submit_button("Submit Answer")

    if submitted:
        if not ans.strip():
            st.warning("Please type an answer before submitting.")
            return

        is_correct, sim = evaluate_answer(ans.strip(), row["answer"], sbert)
        st.session_state.num_attempted += 1

        if is_correct:
            st.success(f"✅ Correct! (Similarity {sim:.2f})")
            st.session_state.score += 1
        else:
            st.error(f"❌ Not quite. (Similarity {sim:.2f})")
            with st.expander("Show reference answer"):
                st.write(row["answer"])

        # log history (for analysis / knowledge gaps)
        st.session_state.history.append(
            {
                "qid": int(qid),
                "difficulty": row["difficulty"],
                "correct": bool(is_correct),
                "similarity": float(sim),
                "student_answer": ans.strip(),
                "reference_answer": row["answer"],
            }
        )

        # always move to NEXT question, even if wrong
        if st.session_state.num_attempted >= st.session_state.num_questions:
            finish_test()
            st.rerun()
        else:
            next_qid = choose_next_question(
                df_q, is_correct, int(row["diff_id"]), st.session_state.asked_ids
            )
            if next_qid is None:
                finish_test()
                st.rerun()
            else:
                st.session_state.current_qid = next_qid
                st.session_state.asked_ids.append(next_qid)
                st.rerun()


# ------------------------------
# SUMMARY PAGE
# ------------------------------
def show_summary():
    st.header("📊 Test Summary")

    score = st.session_state.score
    attempted = st.session_state.num_attempted
    acc = (score / attempted) * 100 if attempted else 0

    st.metric("Total Questions Attempted", attempted)
    st.metric("Correct Answers", score)
    st.metric("Accuracy", f"{acc:.2f}%")

    if st.session_state.history:
        st.subheader("Detailed Answer Breakdown")
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist)


if __name__ == "__main__":
    main()
