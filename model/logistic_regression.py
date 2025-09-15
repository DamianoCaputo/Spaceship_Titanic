# model/logistic_regression.py
from __future__ import annotations
from typing import Any, Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from .base_model import BaseClassifier


class LogisticRegressionModel(BaseClassifier):
    """
     Implementazione del modello di Regressione Logistica
     incapsulato nella classe BaseClassifier.
    
     Questo modello sfrutta un preprocessor (ColumnTransformer)
     già costruito per trasformare i dati in ingresso.
     Viene poi applicata la regressione logistica con iperparametri
     di default ragionevoli per dataset piccoli/medi.
    
     Parametri principali:
     - penalty: tipo di regolarizzazione (default: "l2")
     - C: forza della regolarizzazione (valori piccoli = reg. forte)
     - solver: algoritmo di ottimizzazione (es. "liblinear")
     - max_iter: numero massimo di iterazioni
     - class_weight: gestisce sbilanciamento delle classi
     - random_state: per rendere i risultati riproducibili
    
     Nota: gli iperparametri possono essere modificati
     passando un dizionario model_params al costruttore.
    """

    def __init__(self, preprocessor, model_params: Optional[Dict[str, Any]] = None):
        if model_params is None:
            model_params = dict(
                penalty="l2",
                C=1.0,
                solver="liblinear",  # adatto per dataset piccoli / binari
                max_iter=200,
                class_weight=None,     # usare 'balanced' se il target è sbilanciato
                n_jobs=None,
                random_state=42,
            )
        super().__init__(preprocessor=preprocessor, model_params=model_params)

    def _build_estimator(self) -> BaseEstimator:    # Costruisce e restituisce il classificatore LogisticRegression con gli iperparametri definiti nel dizionario model_params.
        return LogisticRegression(**self.model_params)
