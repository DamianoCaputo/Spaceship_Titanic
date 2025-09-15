# model/xgboost_model.py
from __future__ import annotations
from typing import Any, Dict, Optional
from sklearn.base import BaseEstimator
from .base_model import BaseClassifier

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        "XGBoost non è installato. Installa con: pip install xgboost"
    ) from e


class XGBoostModel(BaseClassifier):
    """
   Implementazione del modello XGBoost incapsulato
     nella classe BaseClassifier.
    
     XGBoost è un algoritmo di boosting basato su alberi:
     - addestra sequenzialmente più alberi di decisione,
       correggendo gli errori commessi dagli alberi precedenti;
     - utilizza tecniche di regolarizzazione e ottimizzazione
       per migliorare generalizzazione ed efficienza.
    
     Parametri principali:
     - n_estimators: numero di alberi (round di boosting)
     - learning_rate: passo di apprendimento (shrinkage)
     - max_depth: profondità massima di ciascun albero
     - subsample: frazione di campioni usati per ogni albero
     - colsample_bytree: frazione di feature usate per ogni albero
     - min_child_weight: peso minimo richiesto in una foglia
     - reg_lambda: regolarizzazione L2
     - reg_alpha: regolarizzazione L1
     - objective: funzione obiettivo (es. "binary:logistic")
     - eval_metric: metrica di valutazione (es. logloss, auc)
     - tree_method: metodo di costruzione degli alberi
                    ("hist" veloce su CPU, "gpu_hist" se disponibile GPU)
     - random_state: per rendere i risultati riproducibili
     - n_jobs: numero di core da utilizzare (-1 = tutti)
    
     Nota: se il dataset è sbilanciato, è utile impostare anche
     scale_pos_weight = (numero negativi / numero positivi).
     Questo modello supporta nativamente predict_proba.
    """

    def __init__(self, preprocessor, model_params: Optional[Dict[str, Any]] = None):
        if model_params is None:
            model_params = dict(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1.0,
                reg_lambda=1.0,
                reg_alpha=0.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",     
                random_state=42,
                n_jobs=-1,
            )
        super().__init__(preprocessor=preprocessor, model_params=model_params)

    def _build_estimator(self) -> BaseEstimator:        # Costruisce e restituisce il classificatore XGBClassifier con gli iperparametri definiti nel dizionario model_params.
        return XGBClassifier(**self.model_params)