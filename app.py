import re

import streamlit as st
import joblib

MODEL_PATH = "models/model.joblib"

# --- same cleaning function used at training time ---
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_review(text: str) -> str:
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = text.lower()
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.set_page_config(
    page_title="Movie Review Sentiment",
    page_icon="🍿",
    layout="centered",
)

st.title("🍿 Movie Review Sentiment Checker")
st.write(
    "Paste a movie review below and the model will guess whether it reads as "
    "a positive or a negative review."
)

review_text = st.text_area(
    "Your review",
    height=180,
    placeholder="e.g. The pacing dragged in the middle but the ending pulled it together...",
)

col1, col2 = st.columns([1, 3])
with col1:
    run = st.button("Check sentiment", type="primary")

if run:
    if not review_text.strip():
        st.warning("Type a review first.")
    else:
        cleaned = clean_review(review_text)
        pred = model.predict([cleaned])[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([cleaned])[0][int(pred)]

        if pred == 1:
            st.success("Predicted: Positive 🙂" if proba is None else f"Predicted: Positive 🙂 ({proba:.0%} confidence)")
        else:
            st.error("Predicted: Negative 🙁" if proba is None else f"Predicted: Negative 🙁 ({proba:.0%} confidence)")

st.divider()
with st.expander("How this works"):
    st.write(
        "The text is cleaned with the same regex steps used during training "
        "(HTML/URL stripped, letters only, lowercased), then passed through a "
        "TF-IDF + Logistic Regression pipeline loaded from `models/model.joblib`."
    )
