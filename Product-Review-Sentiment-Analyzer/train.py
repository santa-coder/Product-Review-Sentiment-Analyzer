"""
train.py
-------------------------------------
Train Product Review Sentiment Analyzer
-------------------------------------
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from config.config import (
    DATASET_PATH,
    MODEL_PATH,
    VECTORIZER_PATH,
    TEST_SIZE,
    RANDOM_STATE
)

from src.future_engineering import FeatureEngineer
from src.preprocess import TextPreprocessor
from src.train_model import ModelTrainer

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

# -------------------------------------------------
# Preprocess Dataset
# -------------------------------------------------

print("Cleaning reviews...")

preprocessor = TextPreprocessor()

df = preprocessor.preprocess_dataframe(
    df,
    "review"
)

# -------------------------------------------------
# Features and Labels
# -------------------------------------------------

X = df["review"]

y = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# -------------------------------------------------
# Train Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# -------------------------------------------------
# TF-IDF Feature Engineering
# -------------------------------------------------

print("Extracting TF-IDF features...")

feature_engineer = FeatureEngineer()

X_train = feature_engineer.fit_transform(X_train)

X_test = feature_engineer.transform(X_test)

# -------------------------------------------------
# Train Models
# -------------------------------------------------

trainer = ModelTrainer()

trained_models = trainer.train(
    X_train,
    y_train
)

# -------------------------------------------------
# Evaluate Models
# -------------------------------------------------

results = trainer.evaluate(
    trained_models,
    X_test,
    y_test
)

# -------------------------------------------------
# Select Best Model
# -------------------------------------------------

best_model, model_name, accuracy = trainer.best_model(
    trained_models,
    results
)

# -------------------------------------------------
# Save Model
# -------------------------------------------------

joblib.dump(
    best_model,
    MODEL_PATH
)

feature_engineer.save(
    VECTORIZER_PATH
)

# -------------------------------------------------
# Finish
# -------------------------------------------------

print("\nTraining Completed Successfully!")
print(f"Best Model : {model_name}")
print(f"Accuracy   : {accuracy:.4f}")
print(f"Model Saved : {MODEL_PATH}")
print(f"Vectorizer Saved : {VECTORIZER_PATH}")