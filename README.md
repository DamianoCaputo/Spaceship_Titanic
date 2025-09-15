#  Spaceship Titanic 

	⁠Progetto universitario/Kaggle sulla competition *[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)*.  
	⁠Obiettivo: prevedere se un passeggero è stato trasportato (Transported) dopo la collisione, a partire da attributi anagrafici, spese a bordo e dettagli della cabina.

---

##  Obiettivi del progetto

•⁠  ⁠Predizione binaria della variabile target Transported.
•⁠  ⁠Implementare una pipeline modulare e completa:  
  import dei dati → cleaning/feature engineering → split → preprocessing → training → valutazione.
•⁠  ⁠Confronto di più modelli (Logistic Regression, SVM, Random Forest, XGBoost) con metriche e grafici standard.


---


## Come avviene l’esecuzione del progetto

Il progetto è pensato per essere eseguito come pipeline end-to-end partendo da main.py.  
Il flusso tipico è il seguente:

1.⁠ ⁠Selezione dataset  
   L’utente inserisce da console il percorso del dataset (train.csv di Kaggle o un file locale).  
   Il modulo file_importer.py gestisce automaticamente il formato (CSV, XLSX, JSON).  

2.⁠ ⁠Data Cleaning & Feature Engineering  
   Viene invocata la classe DataCleaner che:  
   - rimuove duplicati,  
   - normalizza i valori booleani,  
   - estrae feature da Cabin (Deck, CabinNum, Side),  
   - crea feature di gruppo (GroupId, GroupSize, IsSolo),  
   - calcola la spesa totale (SpeseTot),  
   - rimuove colonne irrilevanti (Name, Ticket, ecc.).  

3.⁠ ⁠Split Train/Test  
   Il dataset pulito viene diviso in X (features) e y (target) tramite splitter_Xy.py, con un ulteriore split in train/test (stratificato per Transported).  

4.⁠ ⁠Preprocessing  
   L’utente sceglie l’imputer:  
   - simple: usa SimpleImputer + StandardScaler + OneHotEncoder.  
   - knn: usa KNNImputer + StandardScaler + OneHotEncoder.  
   Il preprocessing viene costruito dinamicamente in un ColumnTransformer (file preprocessing_pipeline.py).  

5.⁠ ⁠Training Modelli  
   Vengono istanziati e addestrati i seguenti modelli, ognuno incapsulato in una classe:  
   - Logistic Regression  
   - Support Vector Machine  
   - Random Forest  
   - XGBoost (se disponibile)  
   Tutti ereditano da BaseClassifier, che gestisce la pipeline (preprocessing + modello).  

6.⁠ ⁠Valutazione  
   Ogni modello viene valutato con metriche standard (Accuracy, Precision, Recall, F1).  
   - Le metriche vengono salvate in formato testo.  
   - I grafici generati includono: distribuzione target, confusion matrix, curve ROC e confronto modelli.  
   I risultati sono salvati nella cartella reports/.  

7.⁠ ⁠Output Finale  
   L’utente ottiene:  
   - Report numerici delle metriche,  
   - Grafici di confronto,   
   - Eventuali dataset intermedi puliti in data/cleaned/.  

---

## Dettaglio del file main.py

Il file main.py è il cuore del progetto e coordina tutte le fasi.  
La sua logica può essere schematizzata così:

1.⁠ ⁠Input utente da console  
   - Richiede il percorso del dataset (train.csv).  
   - Richiede il tipo di imputer da usare (simple o knn).  

2.⁠ ⁠Import dataset  
   - Usa file_importer.py per leggere i dati in un DataFrame.  

3.⁠ ⁠Data Cleaning  
   - Istanzia la classe DataCleaner e applica le regole di pulizia/feature engineering.  
   - Salva il dataset pulito in data/cleaned/.  

4.⁠ ⁠Split X/y  
   - Con splitter_Xy.py divide il dataset in features e target (Transported).  
   - Esegue lo split in train/test con stratificazione.  

5.⁠ ⁠Preprocessing pipeline  
   - Importa da preprocessing_pipeline.py la pipeline in base alla scelta dell’utente (simple o knn).  

6.⁠ ⁠Training e valutazione modelli  
   - Per ogni modello (logistic_regression.py, svm.py, random_forest.py, xgboost_model.py):  
     - Costruisce la pipeline (preprocessing + modello).  
     - Allena il modello (fit).  
     - Calcola predizioni e probabilità (predict, predict_proba).  
     - Valuta le metriche.  
     - Genera e salva grafici in reports/.  

7.⁠ ⁠Confronto finale  
   - Viene generato un grafico riepilogativo (bar plot) con le metriche di tutti i modelli.  
   - Vengono stampati i risultati migliori in console.  

In breve, *main.py* si occupa di orchestrare l’intero flusso di lavoro:  
input →  cleaning →  split →  preprocessing →  training →  valutazione →  salvataggio.  

---


## Struttura delle cartelle

```text
Spaceship_Titanic/
├── main.py                   # Script principale: pipeline end-to-end e confronto modelli
├── data/                     # Dataset ufficiale Kaggle (train/test)
│   ├── train.csv             # Dataset ufficiale Kaggle (train)
│   ├── test.csv              # Dataset ufficiale Kaggle (test)
│   ├── submission.csv        # Esempio submission Kaggle
│   ├── cleaned/              # Dataset pulito (output cleaning)
│   │   └── cleaned_data.csv  
├── data_cleaning/            # DataCleaner + FeatureEngineer + SaveDB + Cleaning
│   ├── __init__.py
│   ├── file_importer.py      # Import CSV/XLSX/JSON (input da console)
│   ├── data_cleaning.py      # Script cleaning
│   ├──splitter_Xy.py        # Estrazione X/y + train/test split (+ report)
|   └──preprocessing_pipeline.py # ColumnTransformer (Simple/KNN Imputer + scaler + encoder)
├── model/                    # Modelli ML (pipeline, fit/predict, save/load)
│   ├── __init__.py
│   ├── base_model.py         # BaseClassifier
│   ├── logistic_regression.py# LogisticRegressionModel
│   ├── svm.py                # SVMModel
│   ├── random_forest.py      # RandomForestModel
│   └── xgboost_model.py      # XGBoostModel
├── evaluation/               # Funzioni metriche
│   ├── __init__.py
│   └── metrics.py
├── plots.py                  # Grafici (target, confusion, ROC, confronto modelli)
├── reports/                  # Output: grafici PNG + report TXT
├── README.md                 # (questo file)
└── .gitignore
