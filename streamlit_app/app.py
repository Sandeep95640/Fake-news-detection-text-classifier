import streamlit as st
import joblib
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../model")

LR_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

MAXLEN = 100  # should match what you used in LSTM training

# Load models
@st.cache_resource
def load_models():
    log_reg_model = joblib.load(LR_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    lstm_model = load_model(LSTM_MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    return log_reg_model, vectorizer, lstm_model, tokenizer

log_reg_model, vectorizer, lstm_model, tokenizer = load_models()


# Helper functions
def preprocess_text_for_lr(text):
    return vectorizer.transform([text])

def preprocess_text_for_lstm(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")
    return padded

def predict_text(text):
    # Logistic Regression
    lr_input = preprocess_text_for_lr(text)
    lr_pred = log_reg_model.predict(lr_input)[0]
    lr_label = "Real" if lr_pred == 1 else "Fake"

    # LSTM
    lstm_input = preprocess_text_for_lstm(text)
    lstm_pred = (lstm_model.predict(lstm_input)[0][0] > 0.5).astype("int")
    lstm_label = "Real" if lstm_pred == 1 else "Fake"

    return lr_label, lstm_label

# Streamlit UI
st.title("Fake News Detection App (Text Only)")

user_input = st.text_area("Enter news text here:")

if st.button("Classify"):
    if user_input.strip():
        lr_label, lstm_label = predict_text(user_input)
        st.subheader("Results")
        st.write(f"**Logistic Regression Model:** {lr_label}")
        st.write(f"**LSTM Model:** {lstm_label}")
    else:
        st.warning("Please enter some text.")
