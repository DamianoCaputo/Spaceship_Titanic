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
        data = GestioneValoriNulli.riempi(data)
        data = CodificaCategoriche.label_encode(data)

        # Salvataggio del dataset pulito
        SaveDB.save_dataset(data)
        print("Dataset pulito salvato correttamente in 'data/cleaned'.")
        
        y = data["Transported"].astype(int)  # da bool a 0/1
        X = data.drop(columns=["Transported"])
        return X, y
