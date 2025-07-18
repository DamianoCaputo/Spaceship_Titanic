import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier


class DatasetLoader:
    @staticmethod
    def load_dataset(relative_path):
        try:
            base_path = os.path.dirname(__file__)
        except NameError:
            base_path = os.getcwd()
        full_path = os.path.join(base_path, relative_path)
        return pd.read_csv(full_path)


class FeatureEngineer:
    @staticmethod
    def remove_columns(df):
        return df.drop(columns=["PassengerId", "Cabin", "Name"], errors="ignore")

    @staticmethod
    def add_total_spending(df):
        spesa_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        df["SpeseTot"] = df[spesa_cols].fillna(0).sum(axis=1)
        return df.drop(columns=spesa_cols)

    @staticmethod
    def process_booleans(df):
        df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(bool).astype(int)
        df["VIP"] = df["VIP"].fillna(False).astype(bool).astype(int)
        return df

    @staticmethod
    def separate_target(df):
        y = df["Transported"].astype(int) if "Transported" in df.columns else None
        X = df.drop(columns=["Transported"], errors="ignore")
        return X, y


class PreprocessingPipelineBuilder:
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features

    def build(self):
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, self.num_features),
            ("cat", cat_pipeline, self.cat_features)
        ])

        return preprocessor


class ModelTrainer:
    def __init__(self, preprocessor, classifier):
        self.pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", classifier)
        ])

    def train(self, X, y):
        self.pipeline.fit(X, y)
        print("✅ Modello addestrato con successo.")
        print("📦 Classi trovate nel target:", self.pipeline.named_steps["classifier"].classes_)
        return self.pipeline

    def transform(self, X):
        return self.pipeline.named_steps["preprocessing"].transform(X)


# === MAIN EXECUTION (opzionale per test locale) ===
if __name__ == "__main__":
    df = DatasetLoader.load_dataset("/Users/martinabenedetti/Desktop/Spaceship_Titanic/data/train.csv")
    df = FeatureEngineer.remove_columns(df)
    df = FeatureEngineer.add_total_spending(df)
    df = FeatureEngineer.process_booleans(df)
    X, y = FeatureEngineer.separate_target(df)

    num_features = ["Age", "SpeseTot"]
    cat_features = ["HomePlanet", "Destination"]

    preprocessor = PreprocessingPipelineBuilder(num_features, cat_features).build()
    trainer = ModelTrainer(preprocessor, KNeighborsClassifier(n_neighbors=5))
    if y is not None:
        trainer.train(X, y)
    else:
        print("ℹ️ Dataset caricato. Colonna 'Transported' non trovata, nessun training effettuato.")


# === FUNZIONE PER L'USO NEI MODELLI ===
def get_processed_data():
    """
    Prepara i dati usando le classi modulari e restituisce X, y già preprocessati.
    """
    df = DatasetLoader.load_dataset("/Users/martinabenedetti/Desktop/Spaceship_Titanic/data/train.csv")
    df = FeatureEngineer.remove_columns(df)
    df = FeatureEngineer.add_total_spending(df)
    df = FeatureEngineer.process_booleans(df)
    X, y = FeatureEngineer.separate_target(df)

    num_features = ["Age", "SpeseTot"]
    cat_features = ["HomePlanet", "Destination"]

    preprocessor = PreprocessingPipelineBuilder(num_features, cat_features).build()
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y