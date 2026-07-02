"""
evaluate.py
----------------------------------------
Model Evaluation Module
Product Review Sentiment Analyzer
----------------------------------------
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class ModelEvaluator:
    """
    Evaluate Classification Models
    """

    def __init__(self):
        pass

    def evaluate(self, model, X_test, y_test):
        """
        Evaluate a trained model.
        """

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        cm = confusion_matrix(
            y_test,
            predictions
        )

        report = classification_report(
            y_test,
            predictions
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": report
        }

    def print_results(self, results):
        """
        Print evaluation results.
        """

        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        print(f"Accuracy  : {results['accuracy']:.4f}")
        print(f"Precision : {results['precision']:.4f}")
        print(f"Recall    : {results['recall']:.4f}")
        print(f"F1 Score  : {results['f1_score']:.4f}")

        print("\nConfusion Matrix")
        print(results["confusion_matrix"])

        print("\nClassification Report")
        print(results["classification_report"])