# model/base_model.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class BaseClassifier:
    """    Classe base per tutti i classificatori del progetto.

    Questa classe fornisce una struttura comune per i modelli di machine learning:
    - gestisce la costruzione di una pipeline di scikit-learn che unisce il
      preprocessing dei dati e il modello vero e proprio;
    - centralizza metodi di addestramento, predizione e salvataggio;
    - permette di riutilizzare la stessa interfaccia con modelli diversi
      (Logistic Regression, SVM, Random Forest, XGBoost, ecc.).

    Parametri
    ---------
    preprocessor : ColumnTransformer
        Oggetto di scikit-learn che si occupa del preprocessamento dei dati
        (ad esempio imputazione dei valori mancanti, scaling delle variabili numeriche,
        encoding delle variabili categoriche). Deve essere già costruito prima
        di passarlo al modello.
    model_params : dict
        Dizionario contenente gli iperparametri specifici del modello scelto.
        Ogni sottoclasse (Logistic, SVM, ecc.) definisce quali iperparametri
        sono disponibili.
    """
    preprocessor: ColumnTransformer
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Fitted artifacts
    estimator_: Optional[BaseEstimator] = field(init=False, default=None)
    pipeline_: Optional[Pipeline] = field(init=False, default=None)

    def build_pipeline(self) -> Pipeline:                       #Costruisce la pipeline di scikit-learn che concatenail preprocessor e il modello.
        from sklearn.pipeline import Pipeline  # importazione locale per evitare la dipendenza globale al momento dell'importazione
        estimator = self._build_estimator()
        self.estimator_ = estimator
        self.pipeline_ = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("model", estimator),
        ])
        return self.pipeline_

    # --- Abstracts ---
    def _build_estimator(self) -> BaseEstimator:        #deve essere implementato da ogni sottoclasse.
        raise NotImplementedError("Le sottoclassi devono implementare _build_estimator()")

    # --- Convenience ---
    def fit(self, X, y):                            #Addestra la pipeline sul dataset fornito.
        pipe = self.pipeline_ or self.build_pipeline()
        return pipe.fit(X, y)

    def predict(self, X):                   #Restituisce le predizioni della pipeline sui dati di input
        if self.pipeline_ is None:
            raise RuntimeError("La pipeline non è stata costruita o addestrata. Chiama prima fit()")
        return self.pipeline_.predict(X)

    def predict_proba(self, X):             #Restituisce le probabilità di appartenenza alle classi.
        if self.pipeline_ is None:
            raise RuntimeError("La pipeline non è stata costruita o addestrata. Chiama prima fit()")
        if hasattr(self.pipeline_, "predict_proba"):
            return self.pipeline_.predict_proba(X)
        model = self.pipeline_.named_steps.get("model")
        if hasattr(model, "predict_proba"):
            Xt = self.pipeline_.named_steps["preprocessor"].transform(X)
            return model.predict_proba(Xt)
        raise AttributeError("Questo modello non implementa predict_proba()")

    def save(self, path: str):              #Salva la pipeline addestrata su file
        import joblib
        if self.pipeline_ is None:
            raise RuntimeError("Nessun modello da salvare. Esegui fit() prima")
        joblib.dump(self.pipeline_, path)

    @classmethod                        #Carica una pipeline precedentemente salvata.
    def load(cls, path: str) -> Pipeline:
        import joblib
        return joblib.load(path)
