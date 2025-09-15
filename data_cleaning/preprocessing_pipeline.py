import pandas as pd
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline

class PreprocessingBuilder:
    """
    Classe con utility per ricavare liste di feature
    e due metodi distinti per costruire il preprocessore:
      - build_with_simple_imputer
      - build_with_knn_imputer
    """

    @staticmethod
    def get_feature_lists(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Restituisce (num_features, cat_features) a partire da X.
        La logica filtra le colonne note, ma aggiunge anche quelle
        numeriche/categoriche non previste.
        """
        # liste "consigliate" per questo dataset
        num_pref = [
            "Age","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck",
            "CabinNum","SpeseTot","GroupSize","CabinSharedCount"
        ]
        cat_pref = [
            "HomePlanet","Destination","Deck","Side","CryoSleep","VIP","AgeBin"
        ]

        # selezione reale in base alle colonne presenti in X
        num_features = [c for c in num_pref if c in X.columns]
        cat_features = [c for c in cat_pref if c in X.columns]

        # aggiungi eventuali numeriche rimaste fuori
        extra_num = [
            c for c in X.select_dtypes(include=["int64","float64","Int64"]).columns
            if c not in num_features and c not in cat_features
        ]
        num_features += extra_num

        # blacklist (non devono finire nel modello)
        blacklist = {"PassengerId","Name","Ticket","GroupId","Cabin"}
        num_features = [c for c in num_features if c not in blacklist]
        cat_features = [c for c in cat_features if c not in blacklist]

        return num_features, cat_features

    @staticmethod
    def build_with_simple_imputer(
        num_features: List[str],
        cat_features: List[str],
        scale_numeric: bool = True
    ) -> ColumnTransformer:
        # numeriche
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        num_pipeline = Pipeline(steps=num_steps)

        # categoriche
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        return ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_features),
                ("cat", cat_pipeline, cat_features),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )

    @staticmethod
    def build_with_knn_imputer(
        num_features: List[str],
        cat_features: List[str],
        knn_neighbors: int = 5,
        scale_numeric: bool = True
    ) -> ColumnTransformer:
        # numeriche
        num_steps = [("imputer", KNNImputer(n_neighbors=knn_neighbors))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        num_pipeline = Pipeline(steps=num_steps)

        # categoriche
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        return ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_features),
                ("cat", cat_pipeline, cat_features),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )

class PreprocessingPipeline:
    """
    Step 3: chiede all'utente quale imputer usare ("simple" o "knn")
    e costruisce solo il preprocessore (ColumnTransformer).
    Nessun modello viene incluso: lo aggiungi tu nel main.
    """

    @staticmethod
    def make_preprocessor_for_X(
        X: pd.DataFrame,
        *,
        scale_numeric: bool = True,
        knn_neighbors: int = 5
    ) -> tuple[ColumnTransformer, list[str], list[str], str]:
        """
        Ritorna: (preprocessor, num_features, cat_features, scelta_effettiva)
        """
        # 1) Input utente
        choice = input("Quale imputer vuoi usare? [simple/knn]: ").strip().lower()
        while choice not in {"simple", "knn"}:
            choice = input("Scelta non valida. Inserisci 'simple' o 'knn': ").strip().lower()

        # 2) Liste feature
        num_features, cat_features = PreprocessingBuilder.get_feature_lists(X)

        # 3) Costruzione preprocessore
        if choice == "simple":
            pre = PreprocessingBuilder.build_with_simple_imputer(
                num_features, cat_features, scale_numeric=scale_numeric
            )
        else:
            pre = PreprocessingBuilder.build_with_knn_imputer(
                num_features, cat_features, knn_neighbors=knn_neighbors, scale_numeric=scale_numeric
            )

        print(f"[Preprocessing] Hai scelto: {choice} | "
              f"num_features={len(num_features)} | cat_features={len(cat_features)}")

        return pre, num_features, cat_features, choice
