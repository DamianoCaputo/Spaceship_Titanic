# === model_testing/test_gradient_boosting.py ===
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from data_cleaning.data_cleaning import get_processed_data
from model_evaluation.advanced_analysis import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    print_classification_metrics
)

X, y = get_processed_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("=== Gradient Boosting ===")
print_classification_metrics(y_test, y_pred)
plot_confusion_matrix(model, X_test, y_test)
plot_roc_curve(model, X_test, y_test)
plot_feature_importance(model, X.columns)