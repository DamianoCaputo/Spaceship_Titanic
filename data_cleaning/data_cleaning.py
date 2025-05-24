import pandas as pd 
import os 

class RimuoviDuplicati:
    """
    Questa Classe permette di rimuovere righe duplicate da un DataFrame
    """
    @staticmethod
    def dup_remove(data: pd.DataFrame) -> pd.DataFrame:
        data_cleaned = data.drop_duplicates()
        return data_cleaned

class GestioneValoriNulli:
    """
    Classe per la gestione dei valori nulli in un DataFrame.
    - Riempie con 0 le colonne di spesa.
    - Riempie con 'False' i booleani.
    - Riempie con la moda le categoriche.
    - Riempie con la media le numeriche continue (es. Age).
    """

    @staticmethod
    def gestisci_nulli(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Colonne di spesa da riempire con 0
        colonne_spesa = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for col in colonne_spesa:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Colonne boolean da riempire con 'False'
        colonne_boolean = ['CryoSleep', 'VIP']
        for col in colonne_boolean:
            if col in df.columns:
                df[col] = df[col].fillna('False')

        # Colonne categoriche da riempire con la moda
        colonne_categoriche = ['HomePlanet', 'Destination']
        for col in colonne_categoriche:
            if col in df.columns:
                moda = df[col].mode()
                if not moda.empty:
                    df[col] = df[col].fillna(moda[0])

        # Colonna numerica continua da riempire con la media
        if 'Age' in df.columns:
            df['Age'] = df['Age'].fillna(df['Age'].mean())

        return df

class CodificaCategoriche:
    """
    Questa Classe permette di codificare le variabili categoriche in un DataFrame
    """
    @staticmethod
    def label_encode(data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for col in data.select_dtypes(include='object').columns:
            unici = data[col].unique()
            mapping = {val: idx for idx, val in enumerate(unici)}
            data[col] = data[col].map(mapping)
        return data

class SaveDB:
    @staticmethod
    def save_dataset(data: pd.DataFrame):
        save_option = input("\nVuoi salvare il dataset pulito? (s/n): ").strip().lower()
        if save_option == 's':
            default_folder = "data/cleaned"  # Cartella di default
            os.makedirs(default_folder, exist_ok=True)  # Crea la cartella se non esiste

            output_filename = input(f"Inserisci il nome del file di output (lascia vuoto per 'cleaned_data.csv'): ").strip()

            # Se non viene specificato un nome, usa "cleaned_data.csv"
            if not output_filename:
                output_filename = "cleaned_data.csv"

            # Costruisci il percorso completo (relativo)
            relative_path = os.path.join(default_folder, output_filename)

            # Converti in percorso assoluto
            absolute_path = os.path.abspath(relative_path)

            # Salva il file
            data.to_csv(absolute_path, index=False)
            print(f"Dataset pulito salvato in: {absolute_path}")
    
class DataCleaner:
    @staticmethod
    def clean_and_save(data):
        data = data.drop(columns=["PassengerId", "Cabin", "Name"])
        data = RimuoviDuplicati.dup_remove(data)
        data = CodificaCategoriche.label_encode(data)

        # Salvataggio del dataset pulito
        SaveDB.save_dataset(data)
        print("Dataset pulito salvato correttamente in 'data/cleaned'.")
        
        y = data["Transported"].astype(int)  # da bool a 0/1
        X = data.drop(columns=["Transported"])
        return X, y
