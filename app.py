import os
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
    If missing or corrupted → fall back to a base Sentence-BERT model.
    Prevents Streamlit OSError crashes.
    """
    try:
        if SBERT_PATH.exists() and any(SBERT_PATH.iterdir()):
            st.info(f"Loading fine-tuned SBERT from `{SBERT_PATH}`")
            return SentenceTransformer(str(SBERT_PATH), local_files_only=True)
        else:
            st.warning(
                f"Fine-tuned SBERT folder not found at `{SBERT_PATH}`.\n"
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
    if not QUESTION_BANK_PATH.exists():
        st.error(
            f"`question_bank.parquet` missing at `{QUESTION_BANK_PATH}`.\n\n"
            "Upload it to your GitHub repo under `/data/`."
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

