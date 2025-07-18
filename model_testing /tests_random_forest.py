# === model_testing/test_random_forest.py ===
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_cleaning.data_cleaning import get_processed_data
from model_evaluation.advanced_analysis import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    print_classification_metrics
)

# Carica dati
X, y = get_processed_data()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Modello
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Risultati test
print("=== Random Forest ===")
print_classification_metrics(y_test, y_pred)
plot_confusion_matrix(model, X_test, y_test)
plot_roc_curve(model, X_test, y_test)
plot_feature_importance(model, X.columns)

