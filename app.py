import os
import random
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib


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
    if not DIFF_MODEL_PATH.exists():
        st.warning("⚠ Difficulty classifier not found. Predictions disabled.")
        return None

    try:
        return joblib.load(DIFF_MODEL_PATH)
    except Exception:
        st.error("Difficulty model file is corrupted.")
        return None


@st.cache_resource
def load_question_bank_and_embeddings():
    """
    Load question bank and pre-compute question embeddings.
    """
    if not QUESTION_BANK_PATH.exists():
        st.error(
            f"`question_bank.parquet` missing at {QUESTION_BANK_PATH}.\n\n"
            "Upload it to your GitHub repo under `data/`."
        )
        st.stop()

    df = pd.read_parquet(QUESTION_BANK_PATH)

    if "difficulty" not in df.columns:
        st.error("Parquet file must include a `difficulty` column.")
        st.stop()

    df["diff_id"] = df["difficulty"].map(DIFF_MAP)

    sbert = load_sbert()

    st.write("📐 Embedding questions…")
    q_embs = sbert.encode(
        df["question"].tolist(),
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    df["q_emb"] = list(q_embs)
    return df


# ------------------------------
# ADAPTIVE LOGIC
# ------------------------------
def choose_initial_question(df, class_level, asked_ids):
    candidates = df[df["difficulty"] == "medium"]
    candidates = candidates[~candidates.index.isin(asked_ids)]

    if candidates.empty:
        candidates = df[~df.index.isin(asked_ids)]

    return int(candidates.sample(1).index[0])


def choose_next_question(df, prev_correct, prev_diff_id, asked_ids):
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
    Compute SBERT cosine similarity between student answer and reference answer.
    Used for logging / analysis even in MCQ mode.
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
# MCQ OPTION GENERATION
# ------------------------------
def generate_mcq_options(df: pd.DataFrame, qid: int, num_options: int = 4):
    """
    Build MCQ options for a given question:
    - 1 correct option (reference answer)
    - num_options-1 distractors from other answers (prefer same source_pdf)
    """
    correct = df.loc[qid, "answer"]
    source = df.loc[qid].get("source_pdf", None)

    # Candidate pool for distractors
    if source is not None and "source_pdf" in df.columns:
        pool = df[(df.index != qid) & (df["source_pdf"] == source)]
    else:
        pool = df[df.index != qid]

    if len(pool) == 0:
        distractors = []
    else:
        distractors = pool["answer"].sample(
            min(num_options - 1, len(pool)),
            replace=False
        ).tolist()

    options = distractors + [correct]
    random.shuffle(options)
    return options, correct


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

        # For stable MCQ options per question
        st.session_state.current_options = None
        st.session_state.current_correct_option = None


def reset_current_options():
    st.session_state.current_options = None
    st.session_state.current_correct_option = None


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
    reset_current_options()


def finish_test():
    st.session_state.test_finished = True
    st.session_state.test_started = False
    reset_current_options()


# ------------------------------
# MAIN APP
# ------------------------------
def main():
    st.set_page_config(page_title="Adaptive NCERT Testing System", layout="wide")
    init_state()

    st.title("📘 AI-Powered Adaptive Testing System (NCERT Social Science)")

    with st.spinner("Loading models and question bank…"):
        df_q = load_question_bank_and_embeddings()
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
        show_summary(df_q)
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
    st.write(row["question"])

    # ---- MCQ options (stable for this question) ----
    if st.session_state.current_options is None or st.session_state.current_correct_option is None:
        options, correct_option = generate_mcq_options(df_q, qid)
        st.session_state.current_options = options
        st.session_state.current_correct_option = correct_option

    options = st.session_state.current_options
    correct_option = st.session_state.current_correct_option

    with st.form("answer_form", clear_on_submit=True):
        selected = st.radio(
            "Choose the correct answer:",
            options,
            index=None,
            key=f"q_radio_{qid}",
        )
        submitted = st.form_submit_button("Submit Answer")

    if submitted:
        if selected is None:
            st.warning("Please select an option.")
            return

        # MCQ correctness
        is_correct = (selected == correct_option)
        # For logging: semantic similarity between chosen option and reference answer
        _, sim = evaluate_answer(selected, row["answer"], sbert)

        st.session_state.num_attempted += 1

        if is_correct:
            st.success(f"✅ Correct! (Similarity {sim:.2f})")
            st.session_state.score += 1
        else:
            st.error(f"❌ Incorrect. (Similarity {sim:.2f})")
            with st.expander("Show Correct Answer"):
                st.write(row["answer"])

        # Save history
        st.session_state.history.append(
            {
                "qid": int(qid),
                "difficulty": row["difficulty"],
                "correct": bool(is_correct),
                "similarity": float(sim),
                "chosen_option": selected,
                "correct_option": correct_option,
            }
        )

        # Prepare for next question
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
                reset_current_options()
                st.rerun()


# ------------------------------
# SUMMARY PAGE
# ------------------------------
def show_summary(df_q):
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
