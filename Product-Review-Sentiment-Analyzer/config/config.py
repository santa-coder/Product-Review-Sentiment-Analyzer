# config/config.py

import os

# -----------------------------
# Base Directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# Dataset Path
# -----------------------------
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "amazon_reviews.csv")

# -----------------------------
# Model Save Paths
# -----------------------------
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")

VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# -----------------------------
# Random State
# -----------------------------
RANDOM_STATE = 42

# -----------------------------
# Train-Test Split
# -----------------------------
TEST_SIZE = 0.20

# -----------------------------
# TF-IDF Parameters
# -----------------------------
MAX_FEATURES = 5000

NGRAM_RANGE = (1, 2)

MIN_DF = 2

MAX_DF = 0.95

# -----------------------------
# Logistic Regression Parameters
# -----------------------------
MODEL_PARAMS = {
    "C": 1.0,
    "solver": "liblinear",
    "max_iter": 1000,
    "random_state": RANDOM_STATE
}

# -----------------------------
# Label Mapping
# -----------------------------
LABEL_MAPPING = {
    "positive": 1,
    "negative": 0
}

# Reverse Mapping
REVERSE_LABEL_MAPPING = {
    1: "Positive 😊",
    0: "Negative 😞"
}

# -----------------------------
# Streamlit Settings
# -----------------------------
APP_TITLE = "Product Review Sentiment Analyzer"

APP_ICON = "🛒"

# -----------------------------
# Create Model Directory
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)