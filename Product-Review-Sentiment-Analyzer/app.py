import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------
# MUST BE FIRST STREAMLIT COMMAND
# ---------------------------------------------------
st.set_page_config(
    page_title="Product Review Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

# ---------------------------------------------------
# Load CSS
# ---------------------------------------------------
def load_css():
    with open("static/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# ---------------------------------------------------
# Download NLTK Data
# ---------------------------------------------------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Product Review Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

# ---------------------------------------------------
# NLP Tools
# ---------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ---------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------
def preprocess_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------
def predict(review):

    review = preprocess_text(review)

    vector = vectorizer.transform([review])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    confidence = round(max(probability) * 100, 2)

    return prediction, confidence

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("🛒 Product Review Sentiment Analyzer")

st.write(
    "Predict whether a customer review is **Positive** 😊 "
    "or **Negative** 😞 using Machine Learning."
)

review = st.text_area(
    "Enter Product Review",
    height=150,
    placeholder="Example: This product is amazing. I really loved it!"
)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        prediction, confidence = predict(review)

        if prediction == 1:

            st.success("😊 Positive Review")

            st.metric(
                label="Confidence",
                value=f"{confidence}%"
            )

        else:

            st.error("😞 Negative Review")

            st.metric(
                label="Confidence",
                value=f"{confidence}%"
            )

        st.divider()

        st.subheader("Review")

        st.write(review)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")
st.caption("Developed using Python • Scikit-learn • NLP • Streamlit")