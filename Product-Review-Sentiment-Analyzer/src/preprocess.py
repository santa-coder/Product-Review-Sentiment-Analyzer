"""
preprocess.py
----------------------------------------
Text Preprocessing Module
Product Review Sentiment Analyzer
----------------------------------------
"""

import re
import string
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (only first run)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


class TextPreprocessor:
    """
    Text preprocessing class for sentiment analysis.
    """

    def __init__(self):
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer

    def clean_text(self, text):
        """
        Clean a single review.
        """

        if pd.isna(text):
            return ""

        text = str(text).lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove numbers
        text = re.sub(r"\d+", "", text)

        # Remove punctuation
        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenization
        words = text.split()

        # Stopword removal + Lemmatization
        words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stop_words
        ]

        return " ".join(words)

    def preprocess_dataframe(self, dataframe, column_name):
        """
        Apply preprocessing to an entire dataframe column.
        """

        dataframe = dataframe.copy()

        dataframe[column_name] = dataframe[column_name].apply(
            self.clean_text
        )

        return dataframe

    def preprocess_list(self, reviews):
        """
        Preprocess a list of reviews.
        """

        return [self.clean_text(review) for review in reviews]

    def preprocess_single_review(self, review):
        """
        Preprocess a single review.
        """

        return self.clean_text(review)