# evaluation/plots.py
from __future__ import annotations
from typing import Dict, Iterable, Optional
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import CalibrationDisplay


# --- Palette fissa per coerenza visiva tra tutti i grafici ---
MODEL_COLORS = {
    "Logistic":     "#1f77b4",  # blu
    "SVM":          "#ff7f0e",  # arancione
    "RandomForest": "#2ca02c",  # verde
    "XGBoost":      "#d62728",  # rosso
}


# --- Utility ---
def _ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _get_scores(model, X):
    """Vettore di score per ROC/Calibrazione:
    - preferisce predict_proba[:,1]
    - altrimenti decision_function
    - altrimenti predizioni (fallback)"""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


# 0) Distribuzione target (torta + barre)
def plot_target_distribution(y, title_prefix: str = "Distribuzione target", save_to: Optional[str] = None):
    values, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    perc = counts / total * 100.0
    labels = [str(v) for v in values]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].pie(counts, labels=[f"{l} ({p:.1f}%)" for l, p in zip(labels, perc)], autopct="%1.1f%%")
    axes[0].set_title(f"{title_prefix} – Torta")

    axes[1].bar(labels, counts)
    axes[1].set_title(f"{title_prefix} – Barre")
    axes[1].set_xlabel("Classe")
    axes[1].set_ylabel("Numero campioni")
    fig.tight_layout()

    if save_to:
        _ensure_dir(os.path.dirname(save_to))
        fig.savefig(save_to, bbox_inches="tight")
    return fig


# 1) Matrici di confusione per più modelli in un’unica figura
def plot_confusion_matrices(models: Dict[str, object], X, y, save_to: Optional[str] = None):
    n = len(models)
    cols = min(2, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"Matrice di confusione — {name}", color=MODEL_COLORS.get(name, "black"))

    for j in range(len(models), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    if save_to:
        _ensure_dir(os.path.dirname(save_to))
        fig.savefig(save_to, bbox_inches="tight")
    return fig


# 2) ROC curves per tutti i modelli
def plot_roc_curves(models: Dict[str, object], X, y, save_to: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        scores = _get_scores(model, X)
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=MODEL_COLORS.get(name, None))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_to:
        _ensure_dir(os.path.dirname(save_to))
        fig.savefig(save_to, bbox_inches="tight")
    return fig

# 4) Confronto modelli — griglia 2x2 con metriche multiple
def plot_model_comparison_grid(
    models: Dict[str, object], X, y,
    metrics: Iterable[str] = ("accuracy", "precision", "recall", "f1"),
    save_to: Optional[str] = None
):
    metrics = [m.lower() for m in metrics]
    allowed = {"accuracy", "precision", "recall", "f1"}
    for m in metrics:
        if m not in allowed:
            raise ValueError(f"Metrica '{m}' non valida. Scegli tra {allowed}.")

    k = len(metrics)
    rows = 2 if k > 2 else 1
    cols = 2 if k > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = np.atleast_1d(axes).ravel()

    preds = {name: m.predict(X) for name, m in models.items()}
    names = list(models.keys())
    colors = [MODEL_COLORS.get(n, "gray") for n in names]

    def _score(metric, y_true, y_pred):
        if metric == "accuracy":  return accuracy_score(y_true, y_pred)
        if metric == "precision": return precision_score(y_true, y_pred, pos_label=True)
        if metric == "recall":    return recall_score(y_true, y_pred, pos_label=True)
        return f1_score(y_true, y_pred, pos_label=True)

    for ax, metric in zip(axes, metrics):
        vals = [_score(metric, y, preds[n]) for n in names]
        ax.bar(names, vals, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"Confronto modelli — {metric}")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    if save_to:
        _ensure_dir(os.path.dirname(save_to))
        fig.savefig(save_to, bbox_inches="tight")
    return fig