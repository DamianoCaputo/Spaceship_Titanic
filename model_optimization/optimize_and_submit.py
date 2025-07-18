import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from data_cleaning.data_cleaning import DatasetLoader, FeatureEngineer, PreprocessingPipelineBuilder, ModelTrainer

# === 1. Percorsi dataset ===
train_path = "/Users/martinabenedetti/Desktop/Spaceship_Titanic/data/train.csv"
test_path = "/Users/martinabenedetti/Desktop/Spaceship_Titanic/data/test.csv"
submission_path = "/Users/martinabenedetti/Desktop/Spaceship_Titanic/data/submission.csv"

# === 2. Carica e prepara train set ===
df_train = DatasetLoader.load_dataset(train_path)
df_train = FeatureEngineer.remove_columns(df_train)
df_train = FeatureEngineer.add_total_spending(df_train)
df_train = FeatureEngineer.process_booleans(df_train)
X_train, y_train = FeatureEngineer.separate_target(df_train)

# === 3. Carica e prepara test set ===
df_test = DatasetLoader.load_dataset(test_path)
passenger_ids = df_test["PassengerId"]
df_test = FeatureEngineer.remove_columns(df_test)
df_test = FeatureEngineer.add_total_spending(df_test)
df_test = FeatureEngineer.process_booleans(df_test)

# === 4. Definizione colonne ===
num_features = ["Age", "SpeseTot"]
cat_features = ["HomePlanet", "Destination"]

# === 5. Costruzione preprocessor ===
preprocessor = PreprocessingPipelineBuilder(num_features, cat_features).build()

# === 6. Hyperparameter tuning con GridSearch ===
param_grid = {
    "n_estimators": [100, 150],
    "learning_rate": [0.1, 0.05],
    "max_depth": [3, 5]
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# === 7. Training con miglior modello ===
trainer = ModelTrainer(preprocessor, grid)
model = trainer.train(X_train, y_train)
best_model = trainer.pipeline.named_steps["classifier"].best_estimator_

print("✅ Miglior modello trovato:", best_model)

# === 8. Trasforma il test set ===
X_test = preprocessor.transform(df_test)

# === 9. Previsioni
predictions = best_model.predict(X_test)

# === 10. Prepara file di submission
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Transported": predictions.astype(bool)  # Kaggle vuole True/False, non 1/0
})

submission.to_csv(submission_path, index=False)
print(f"📄 File di submission salvato in: {submission_path}")