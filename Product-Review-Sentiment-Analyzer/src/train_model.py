from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


class ModelTrainer:

    def __init__(self):

        self.models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ),

            "Naive Bayes":
                MultinomialNB(),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                ),

            "Linear SVM":
                LinearSVC(
                    random_state=42
                )
        }

    def train(self, X_train, y_train):

        trained_models = {}

        for name, model in self.models.items():

            print(f"Training {name}...")

            model.fit(X_train, y_train)

            trained_models[name] = model

        return trained_models

    def evaluate(self, trained_models, X_test, y_test):

        results = {}

        for name, model in trained_models.items():

            prediction = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                prediction
            )

            results[name] = accuracy

        return results

    def best_model(self, trained_models, results):

        best_name = max(
            results,
            key=results.get
        )

        best_model = trained_models[best_name]

        best_accuracy = results[best_name]

        return best_model, best_name, best_accuracy