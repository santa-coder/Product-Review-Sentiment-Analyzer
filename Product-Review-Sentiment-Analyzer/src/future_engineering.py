"""
feature_engineering.py
------------------------------------------
Feature Engineering Module
Product Review Sentiment Analyzer
------------------------------------------
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureEngineer:
    """
    Convert text into numerical features using TF-IDF.
    """

    def __init__(
        self,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    ):

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words="english"
        )

    def fit_transform(self, X_train):
        """
        Fit TF-IDF on training data and transform it.
        """

        return self.vectorizer.fit_transform(X_train)

    def transform(self, X_test):
        """
        Transform test/new data using existing TF-IDF.
        """

        return self.vectorizer.transform(X_test)

    def fit(self, X_train):
        """
        Fit TF-IDF only.
        """

        self.vectorizer.fit(X_train)

    def save(self, path):
        """
        Save trained TF-IDF vectorizer.
        """

        joblib.dump(self.vectorizer, path)

    def load(self, path):
        """
        Load saved TF-IDF vectorizer.
        """

        self.vectorizer = joblib.load(path)

    def vocabulary_size(self):
        """
        Return vocabulary size.
        """

        return len(self.vectorizer.vocabulary_)

    def feature_names(self):
        """
        Return feature names.
        """

        return self.vectorizer.get_feature_names_out()

    def get_vectorizer(self):
        """
        Return TF-IDF object.
        """

        return self.vectorizer