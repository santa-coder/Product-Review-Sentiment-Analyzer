"""
predict.py
-------------------------------
Predict sentiment for a user review
-------------------------------
"""

import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.future_engineering import FeatureEngineer
from src.preprocess import TextPreprocessor

feature_engineer = FeatureEngineer()

feature_engineer.load(
    "models/tfidf_vectorizer.pkl"
)

review = [
    "This phone is amazing."
]

review_vector = feature_engineer.transform(review)

preprocessor = TextPreprocessor()

review =  [
    "Excellent Product!!",
    "Worst Purchase Ever!!",
    "Battery is very good."
]

clean_review = preprocessor.preprocess_single_review(review)

print(clean_review)

# Download NLTK resources (only first time)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Load saved model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Clean the input review
    """

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove Numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Remove stopwords and Lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


def predict_sentiment(review):
    """
    Predict sentiment of a review
    """

    review = preprocess_text(review)

    vector = vectorizer.transform([review])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    confidence = max(probability) * 100

    if prediction == 1:
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative 😞"

    return sentiment, confidence


if __name__ == "__main__":

    print("=" * 50)
    print("Product Review Sentiment Analyzer")
    print("=" * 50)

    review = input("\nEnter Product Review:\n")

    sentiment, confidence = predict_sentiment(review)

    print("\nPrediction :", sentiment)
    print(f"Confidence : {confidence:.2f}%")