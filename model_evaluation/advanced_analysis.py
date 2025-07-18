# === model_evaluation/advanced_analysis.py ===
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, roc_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score
)
import pandas as pd
import numpy as np


def plot_confusion_matrix(model, X_test, y_test):
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
    disp.ax_.set_title("Matrice di Confusione")
    plt.show()


def plot_roc_curve(model, X_test, y_test):
    probas = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probas)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Curva ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()


def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sns.barplot(x=importances, y=feature_names)
        plt.title("Feature Importance")
        plt.xlabel("Importanza")
        plt.ylabel("Feature")
        plt.show()
    else:
        print("❌ Questo modello non supporta la feature importance.")


def print_classification_metrics(y_true, y_pred):
    print("\n📈 Metriche sul test set:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")