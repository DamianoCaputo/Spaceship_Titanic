# model/svm.py
from __future__ import annotations
from typing import Any, Dict, Optional
from sklearn.base import BaseEstimator
from sklearn.svm import SVC, LinearSVC
from .base_model import BaseClassifier


class SVMModel(BaseClassifier):
    """
   Implementazione del modello Support Vector Machine (SVM)
     incapsulato nella classe BaseClassifier.
    
     La SVM è un classificatore che cerca l’iperpiano che separa
     al meglio le classi massimizzando il margine. Con l’uso dei kernel
     può gestire anche problemi non lineari (es. kernel RBF).
    
     Parametri principali:
     - use_linear: se True usa LinearSVC (più veloce con tante feature sparse,
                   ma NON supporta predict_proba)
     - kernel: tipo di kernel da usare (default "rbf")
     - C: parametro di regolarizzazione (alto = meno regolarizzazione)
     - gamma: parametro del kernel RBF
     - probability: se True abilita predict_proba (più lento)
     - class_weight: utile se le classi sono sbilanciate
    
     Nota: se si imposta use_linear=True, il modello diventa LinearSVC.
     Questo è più rapido, ma non fornisce probabilità (solo classi predette)
    """

    def __init__(self, preprocessor, model_params: Optional[Dict[str, Any]] = None):
        if model_params is None:
            model_params = dict(
                use_linear=False,   # True => LinearSVC (niente predict_proba)
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,   # # abilita predict_proba (più lento ma utile per metriche ROC/PR)
                class_weight=None,
            )
        super().__init__(preprocessor=preprocessor, model_params=model_params)

    def _build_estimator(self) -> BaseEstimator:    ## Costruisce e restituisce il classificatore SVM.
        params = dict(self.model_params)  # copia per sicurezza
        use_linear = params.pop("use_linear", False)
        if use_linear:
            # LinearSVC ignora kernel/gamma/probability ed è spesso più rapido
            # NB: niente predict_proba con LinearSVC
            C = params.pop("C", 1.0)
            return LinearSVC(C=C)
        else:
            # SVC classico (kernel RBF di default)
            return SVC(**params)