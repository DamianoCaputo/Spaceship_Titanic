from __future__ import annotations
import os
import pandas as pd
from data_cleaning import SelectionFile, Cleaning, SimpleSplitter, PreprocessingPipeline
from model import LogisticRegressionModel, SVMModel, RandomForestModel, XGBoostModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from evaluation import (
    plot_target_distribution,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_model_comparison_grid,
)


def main():
    # === 0) Import ===
    data = SelectionFile.import_data()
    if data.empty:
        print("Il dataset è vuoto. Termino.")
        return

    # === 1) Cleaning ===
    cleaned = Cleaning.clean_and_save(data)

    # === 2) Split ===
    X_train, X_test, y_train, y_test = SimpleSplitter.split(cleaned)

    # === 3) Preprocessing ===
    preprocessor, num_f, cat_f, choice = PreprocessingPipeline.make_preprocessor_for_X(
        cleaned, scale_numeric=True, knn_neighbors=5
    )

    # === 4) Modelli ===
    logistic_clf = LogisticRegressionModel(preprocessor=preprocessor)
    logistic_clf.fit(X_train, y_train)

    svm_clf = SVMModel(preprocessor=preprocessor, model_params=dict(use_linear=False, kernel="rbf", C=1.0, gamma="scale", probability=True))
    svm_clf.fit(X_train, y_train)

    rf_clf = RandomForestModel(preprocessor=preprocessor)
    rf_clf.fit(X_train, y_train)

    xgb_clf = XGBoostModel(preprocessor=preprocessor)
    xgb_clf.fit(X_train, y_train)

    models = {
        "Logistic": logistic_clf,
        "SVM": svm_clf,
        "RandomForest": rf_clf,
        "XGBoost": xgb_clf,
    }

   # === 5) Metriche + report (con diagnostica per modello) ===
    print("\n=== METRICHE SU TEST ===")
    header = f"{'Model':<12}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}"
    print(header); print("-"*len(header))

    print("Modelli nel dict:", list(models.keys()))  # debug
    reports_txt = []

    for name, m in models.items():
        try:
            print(f"\n>>> Valutazione {name} ...", flush=True)
            y_pred = m.predict(X_test)

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, pos_label=True)
            rec  = recall_score(y_test, y_pred, pos_label=True)
            f1   = f1_score(y_test, y_pred, pos_label=True)
            print(f"{name:<12}  {acc:6.3f}  {prec:6.3f}  {rec:6.3f}  {f1:6.3f}")

            rep = classification_report(y_test, y_pred, digits=4)
            print(f"\n--- CLASSIFICATION REPORT ({name}) ---\n{rep}")
            reports_txt.append(f"=== {name} ===\n{rep}\n")

        except Exception as e:
            # non bloccare l'intero ciclo: segnala e continua con gli altri
            print(f"[ERRORE in {name}] {type(e).__name__}: {e}")
            continue

    os.makedirs("reports", exist_ok=True)
    with open("reports/classification_reports.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(reports_txt))
    print("\nReport testuali salvati in: reports/classification_reports.txt")

    # === 6) Grafici ===
    plot_target_distribution(y_test, title_prefix="Target Test", save_to="reports/target_distribution.png")
    plot_confusion_matrices(models, X_test, y_test, save_to="reports/confusion_matrices.png")
    plot_roc_curves(models, X_test, y_test, save_to="reports/roc_curves.png")
    plot_model_comparison_grid(models, X_test, y_test, save_to="reports/model_comparison.png")

    print("\nGrafici e report salvati in ./reports")


if __name__ == "__main__":
    main()



