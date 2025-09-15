# evaluation/metrics.py
from __future__ import annotations
from typing import Dict
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_model(name: str, model, X, y) -> Dict[str, float]:
    """Calcola metriche base sul set passato (tipicamente test)."""
    y_pred = model.predict(X)
    return dict(
        model=name,
        accuracy=accuracy_score(y, y_pred),
        precision=precision_score(y, y_pred, pos_label=True),
        recall=recall_score(y, y_pred, pos_label=True),
        f1=f1_score(y, y_pred, pos_label=True),
        report=classification_report(y, y_pred, digits=4),
    )

def evaluate_models(models: Dict[str, object], X, y) -> pd.DataFrame:
    """Valuta più modelli e restituisce un DataFrame indicizzato per nome modello."""
    rows = []
    for name, m in models.items():
        rows.append(evaluate_model(name, m, X, y))
    df = pd.DataFrame(rows).set_index("model")
    return df