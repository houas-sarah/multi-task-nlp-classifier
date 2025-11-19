"""
Streamlit application for a multi-task NLP classifier.

Model:
    - Shared BiLSTM backbone
    - Three classification heads:
        1) Emotion classification
        2) Gender-based violence type classification
        3) Hate / offensive / neutral classification

Files expected:
    - models/multitask_lstm.h5
    - models/tokenizer.json
    - models/label_maps.pkl

Run:
    streamlit run app.py
"""

import json
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Configuration

MODEL_PATH = "models/multitask_lstm.h5"
TOKENIZER_PATH = "models/tokenizer.json"
LABEL_MAPS_PATH = "models/label_maps.pkl"

# Must be the same as in the training notebook
MAX_SEQUENCE_LEN = 60


# Model and assets loading

@st.cache_resource
def load_model_and_assets():
    """Load the trained Keras model, tokenizer and label mappings."""
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load tokenizer as raw JSON string
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tok_json = f.read()
    tokenizer = tokenizer_from_json(tok_json)

    # Load label mappings: {"emotion": {0: "...", ...}, ...}
    with open(LABEL_MAPS_PATH, "rb") as f:
        label_maps = pickle.load(f)

    return model, tokenizer, label_maps


model, tokenizer, label_maps = load_model_and_assets()


# Preprocessing and prediction helpers

def preprocess_text(text: str) -> str:
    """
    Basic preprocessing: lowercasing and stripping.

    If your notebook uses a more advanced preprocessing function
    (stopword removal, punctuation, etc.), copy it here to keep
    training and inference consistent.
    """
    return str(text).lower().strip()


def texts_to_padded(texts):
    """Convert a list of texts into padded sequences."""
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        seqs,
        maxlen=MAX_SEQUENCE_LEN,
        padding="post",
        truncating="post",
    )
    return padded


def predict_all_tasks(text: str) -> dict:
    """
    Run the multi-task model on a single input text.

    Returns a dictionary with:
        cleaned, emotion, violence, hate
    """
    cleaned = preprocess_text(text)
    x = texts_to_padded([cleaned])

    # Model outputs: [emotion_output, violence_output, hate_output]
    emo_pred, vio_pred, hate_pred = model.predict(
        {
            "emotion_input": x,
            "violence_input": x,
            "hate_input": x,
        },
        verbose=0,
    )

    emo_idx = int(np.argmax(emo_pred, axis=1)[0])
    vio_idx = int(np.argmax(vio_pred, axis=1)[0])
    hate_idx = int(np.argmax(hate_pred, axis=1)[0])

    emo_label = label_maps["emotion"][emo_idx]
    vio_label = label_maps["violence"][vio_idx]
    hate_label = label_maps["hate"][hate_idx]

    return {
        "cleaned": cleaned,
        "emotion": emo_label,
        "violence": vio_label,
        "hate": hate_label,
    }


# styling

st.set_page_config(
    page_title="Multi-Task NLP Classifier",
    layout="wide",
)

CUSTOM_CSS = """
<style>
/* Make the main area a bit narrower and centered */
.main-container {
    max-width: 980px;
    margin: 0 auto;
    padding: 2.5rem 1.5rem 4rem 1.5rem;
}

/* Tweak background and fonts */
body, .stApp {
    background-color: #050509;
}

/* Title and subtitle */
.app-title {
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.app-subtitle {
    font-size: 0.98rem;
    color: #9ca3af;
    max-width: 720px;
    margin-bottom: 2.0rem;
}

/* Text area */
textarea[aria-label="Input text"] {
    font-family: "JetBrains Mono", Menlo, Consolas, monospace;
    font-size: 0.95rem;
}

/* Button */
.stButton > button {
    border-radius: 999px;
    padding: 0.4rem 1.6rem;
    font-weight: 500;
}

/* Prediction block */
.results-wrapper {
    margin-top: 2rem;
}

.cleaned-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    background: #111827;
    border: 1px solid #1f2937;
    font-size: 0.85rem;
    font-family: "JetBrains Mono", Menlo, Consolas, monospace;
}

/* Cards for the three tasks */
.prediction-card {
    background: radial-gradient(circle at top left, #111827 0, #020617 60%);
    border-radius: 0.9rem;
    padding: 1.1rem 1.3rem;
    border: 1px solid #1f2937;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
    animation: fadeInUp 0.45s ease-out;
}

.prediction-label {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 0.35rem;
}

.prediction-value {
    font-size: 1.02rem;
    font-weight: 600;
    padding: 0.2rem 0;
}

/* Simple entrance animation */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translate3d(0, 12px, 0);
    }
    to {
        opacity: 1;
        transform: translate3d(0, 0, 0);
    }
}

/* Hide default Streamlit menu/footer to look cleaner in screenshots */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Layout

st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown('<div class="app-title">Multi-Task NLP Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'This interface wraps a shared BiLSTM model trained jointly on three tasks: '
    'emotion recognition, gender-based violence type classification, and hate/offensive speech detection.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("**Input text**")
text_input = st.text_area(
    "",
    height=140,
    placeholder="Type a short sentence or tweet, for example: I love you, thank you for your help.",
)

run_button = st.button("Run prediction")

if run_button:
    if text_input.strip() == "":
        st.warning("Please enter some text before running a prediction.")
    else:
        result = predict_all_tasks(text_input)

        st.markdown("### Model output")

        st.markdown(
            f'<span class="cleaned-pill">cleaned: {result["cleaned"]}</span>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
                <div class="prediction-card">
                    <div class="prediction-label">Emotion</div>
                    <div class="prediction-value">{}</div>
                </div>
                """.format(result["emotion"]),
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="prediction-card">
                    <div class="prediction-label">Violence type</div>
                    <div class="prediction-value">{}</div>
                </div>
                """.format(result["violence"]),
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
                <div class="prediction-card">
                    <div class="prediction-label">Hate / offensive</div>
                    <div class="prediction-value">{}</div>
                </div>
                """.format(result["hate"]),
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown(
            "---\n"
            "This page is a minimal demo interface. The core work is in the notebook: "
            "data preparation, multi-task model design, training and evaluation.",
        )

st.markdown("</div>", unsafe_allow_html=True)
