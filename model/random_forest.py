# model/random_forest.py
from __future__ import annotations
from typing import Any, Dict, Optional
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseClassifier


class RandomForestModel(BaseClassifier):
    """
   Implementazione del modello Random Forest incapsulato
    nella classe BaseClassifier.
    
     La Random Forest è un insieme (ensemble) di alberi di decisione:
     - Ogni albero viene addestrato su un sottoinsieme casuale
       di dati e di feature (bagging).
     - La previsione finale si ottiene aggregando le predizioni
       dei singoli alberi (maggioranza per classificazione).
    
     Parametri principali:
     - n_estimators: numero di alberi nella foresta
     - max_depth: profondità massima degli alberi
     - min_samples_split: numero minimo di campioni per dividere un nodo
     - min_samples_leaf: numero minimo di campioni in una foglia
     - max_features: numero di feature considerate a ogni split
     - bootstrap: se True usa campionamento con rimpiazzo
     - class_weight: utile in caso di target sbilanciato
     - random_state: garantisce riproducibilità
     - n_jobs: numero di core usati (-1 = tutti i core disponibili)
    
     Nota: supporta predict_proba di default, quindi restituisce
     anche le probabilità delle classi.
    """

    def __init__(self, preprocessor, model_params: Optional[Dict[str, Any]] = None):
        if model_params is None:
            model_params = dict(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",   # valore tipico per classificazione
                bootstrap=True,
                class_weight=None,     # usare 'balanced' se le classi sono sbilanciat
                random_state=42,
                n_jobs=-1,
            )
        super().__init__(preprocessor=preprocessor, model_params=model_params)

    def _build_estimator(self) -> BaseEstimator:        # Costruisce e restituisce il classificatore RandomForestClassifier con gli iperparametri definiti nel dizionario model_params. 
        return RandomForestClassifier(**self.model_params)