# === model_evaluation/modelli.py ===
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb 

# 🔁 Importa i dati già processati da data_cleaning
def get_data():
    from data_cleaning.data_cleaning import get_processed_data
    return get_processed_data()


class ModelEvaluator:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.results = {}

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

    def evaluate_models(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }

        for name, model in self.models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring="accuracy")
            self.results[name] = {
                "Accuracy Mean": np.mean(scores),
                "Accuracy Std": np.std(scores)
            }

    def get_results_df(self):
        return pd.DataFrame(self.results).T.sort_values(by="Accuracy Mean", ascending=False)

    def train_best_model(self):
        best_model_name = max(self.results, key=lambda k: self.results[k]["Accuracy Mean"])
        self.best_model = self.models[best_model_name]
        self.best_model.fit(self.X_train, self.y_train)
        print(f"\n✅ Miglior modello selezionato: {best_model_name}")
        return self.best_model

    def test_performance(self):
        y_pred = self.best_model.predict(self.X_test)
        return {
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "Precision": precision_score(self.y_test, y_pred),
            "Recall": recall_score(self.y_test, y_pred),
            "F1-Score": f1_score(self.y_test, y_pred)
        }


if __name__ == "__main__":
    X, y = get_data()

    evaluator = ModelEvaluator(X, y)
    evaluator.split_data()
    evaluator.evaluate_models()

    print("\n📊 Risultati Cross-Validation:")
    print(evaluator.get_results_df())

    evaluator.train_best_model()
    print("\n📈 Performance sul test set:")
    for k, v in evaluator.test_performance().items():
        print(f"{k}: {v:.4f}")




