"""
utils.py
----------------------------------
Utility functions for the Product
Review Sentiment Analyzer
----------------------------------
"""

import os
import re
import string
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (first run only)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    Clean and preprocess review text.
    """

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


def save_model(model, path):
    """
    Save model/vectorizer using Joblib.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """
    Load a saved model/vectorizer.
    """

    return joblib.load(path)


def create_directory(path):
    """
    Create directory if it doesn't exist.
    """

    os.makedirs(path, exist_ok=True)


def print_separator():
    """
    Print a separator line.
    """

    print("=" * 60)


def encode_label(label):
    """
    Convert label text to integer.
    """

    mapping = {
        "positive": 1,
        "negative": 0
    }

    return mapping.get(label.lower())


def decode_label(label):
    """
    Convert integer label to text.
    """

    mapping = {
        1: "Positive 😊",
        0: "Negative 😞"
    }

    return mapping.get(label)