from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split

class SimpleSplitter:
    @staticmethod
    def _extract_xy(data: pd.DataFrame,target: str = "Transported") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Si aspetta un DataFrame già pulito/featurizzato.
        Estrae X,y e normalizza y in {0,1} se è bool o string ("true"/"false").
        Droppa righe con y mancante.
        """
        if target not in data.columns:
            raise ValueError(f"Target '{target}' non trovato.")

        y = data[target]
        # bool -> int
        if y.dtype == bool:
            y = y.astype(int)
        # object/string -> mappa true/false
        elif y.dtype.kind in "OUS":
            y = (
                y.astype(str).str.lower()
                 .map({"true": 1, "false": 0})
                 .astype("Int64")
            )

        X = data.drop(columns=[target])

        # rimuovi righe con y NA
        mask = y.notna()
        if mask.sum() != len(y):
            X = X.loc[mask].reset_index(drop=True)
            y = y.loc[mask].reset_index(drop=True)

        # porta a int (se Int64 nullable)
        if getattr(y.dtype, "name", "") == "Int64":
            y = y.astype(int)

        return X, y

    @staticmethod
    def split(
        data: pd.DataFrame,
        target: str = "Transported",
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Ritorna: X_train, X_test, y_train, y_test
        """
        X, y = SimpleSplitter._extract_xy(data, target=target)
        strat = y if stratify else None
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat
        )

    @staticmethod
    def split_with_report(
        data: pd.DataFrame,
        target: str = "Transported",
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        """
        Come split(), ma ritorna anche un piccolo report dict.
        """
        X_train, X_test, y_train, y_test = SimpleSplitter.split(
            data, target, test_size, random_state, stratify
        )
        report = {
            "n_rows": len(data),
            "n_features": data.drop(columns=[target]).shape[1],
            "target": target,
            "test_size": test_size,
            "stratify": stratify,
            "train_shape": (X_train.shape[0], X_train.shape[1]),
            "test_shape": (X_test.shape[0], X_test.shape[1]),
            "pos_rate_full": float(
                data[target].astype(str).str.lower().map({"true":1,"false":0}).mean()
                if target in data.columns else float("nan")
            ) if data[target].dtype.kind in "OUS" else (
                float(data[target].mean()) if set(data[target].unique()) <= {0,1,True,False} else None
            ),
        }
        return X_train, X_test, y_train, y_test, report
