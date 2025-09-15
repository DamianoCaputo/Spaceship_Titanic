# data_cleaning/data_cleaning.py
from __future__ import annotations
import os
from typing import List, Tuple
import numpy as np
import pandas as pd

class DataCleaner:
    @staticmethod
    def dup_remove(data: pd.DataFrame) -> pd.DataFrame:
        """
        Vengono rimossi i duplicati nel dataset.
        """
        data = data.drop_duplicates()
        return data
    
    @staticmethod
    def drop_target_na(data: pd.DataFrame) -> pd.DataFrame:
        """
        Rimuove le righe con valori mancanti nella colonna target.
        """
        data = data.dropna(subset=["Transported"])
        return data

    @staticmethod
    def normalize_strings(data: pd.DataFrame) -> pd.DataFrame:
        """
        Rimuove spazi bianchi iniziali e finali dalle colonne di tipo stringa.
        """
        data = data.copy()
        for col in data.select_dtypes(include=["object"]).columns:
            data[col] = data[col].astype(str).str.strip()
        return data
    
    @staticmethod
    def process_booleans(data: pd.DataFrame) -> pd.DataFrame: #porta le colonne booleane a 0/1
        """
        Converte le colonne booleane: CryoSleep, VIP in 0/1.
        """
        data = data.copy()
        bool_cols = [c for c in ["CryoSleep", "VIP"] if c in data.columns]
        truthy = {"true", "1", "yes", "y", "t"}
        falsy = {"false", "0", "no", "n", "f"}
        for col in bool_cols:
            data[col] = (
                data[col]
                .astype(str)
                .str.lower()
                .map(lambda x: 1 if x in truthy else (0 if x in falsy else np.nan))
            )
        return data

class FeatureEngineer:
    @staticmethod #capire se rimuovere le colonne o no---
    def add_total_spending(data: pd.DataFrame) -> pd.DataFrame:
        """
        Crea SpeseTot come somma delle colonne di spesa.
        - Se le colonne esistono: somma riga per riga (NaN -> 0).
        - Se non esiste nessuna colonna di spesa: SpeseTot = 0.
        """
        data = data.copy()
        spend_cols = [c for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] if c in data.columns]
        if spend_cols:
            data["SpeseTot"] = data[spend_cols].fillna(0).sum(axis=1)
        else:
            data["SpeseTot"] = 0.0
        return data

    @staticmethod
    def parse_cabin(data: pd.DataFrame) -> pd.DataFrame:
        """
        Viene divisa la colonna Cabin in 3 colonne per una maggiore leggibilità: 
            - Deck, 
            - CabinNum, 
            - Side
        """
        data = data.copy()
        if "Cabin" in data.columns:
            parts = data["Cabin"].astype(str).str.split("/", expand=True)
            if parts.shape[1] >= 1:
                data["Deck"] = parts[0].replace("nan", np.nan)
            if parts.shape[1] >= 2:
                data["CabinNum"] = pd.to_numeric(parts[1], errors="coerce").astype('Int64')
            if parts.shape[1] >= 3:
                data["Side"] = parts[2].replace("nan", np.nan)
        return data
    
    @staticmethod
    def add_group_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Dalla colonna PassengerId si possono ricavare informazioni rilevanti sul gruppo di viaggio.
        Prendiamo un valore di esempio: 0001_01:
            - 0001 = GroupId, identificativo del gruppo di viaggio
            - 01   = MemberId, numero del passeggero all'interno del gruppo
        Quindi quello che possiamo ottenere:
          - GroupSize (# passeggeri nel gruppo)
          - IsSolo (1 se gruppo da 1, 0 altrimenti)
        """
        data = data.copy()
        if "PassengerId" in data.columns:
            data["GroupId"] = data["PassengerId"].astype(str).str.split("_").str[0]
            grp_size = data.groupby("GroupId")["PassengerId"].transform("size")
            data["GroupSize"] = grp_size.astype("Int64")
            data["IsSolo"] = (data["GroupSize"] == 1).astype("Int64")
        else:
            data["GroupSize"] = pd.Series([pd.NA]*len(data), dtype="Int64")
            data["IsSolo"]   = pd.Series([pd.NA]*len(data), dtype="Int64")
        return data
    
    @staticmethod
    def remove_irrelevant_columns(data: pd.DataFrame) -> pd.DataFrame:
        """
        Rimuove colonne non utili per il modeling:
            - Name 
            - PassengerId
            - Cabin
        """
        data = data.copy()
        drop_cols = [c for c in ["Name", "PassengerId", "Cabin"] if c in data.columns]
        if drop_cols:
            data = data.drop(columns=drop_cols, errors="ignore")
        return data

# Salvataggio del DB pulito -----------------
class SaveDB:
    @staticmethod
    def save_dataset(data: pd.DataFrame):
        save_option = input("\nVuoi salvare il dataset pulito? (s/n): ").strip().lower()
        if save_option == 's':
            default_folder = "data/cleaned"  # Cartella di default
            os.makedirs(default_folder, exist_ok=True)  # Crea la cartella se non esiste

            output_filename = input(f"Inserisci il nome del file di output compreso di estensione (lascia vuoto per 'cleaned_data.csv'): ").strip()

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

class Cleaning:
    @staticmethod
    def clean_and_save(data: pd.DataFrame):
        data = data.copy()

        # 1) Cleaning base
        print("Eseguo la pulizia dei dati: rimozione duplicati, normalizzazione stringhe, gestione booleani...","\n")
        data = DataCleaner.dup_remove(data)
        data = DataCleaner.normalize_strings(data)
        data = DataCleaner.process_booleans(data)  # CryoSleep/VIP -> Int64; NON tocca Transported

        # 2) Feature Engineering
        print("Eseguo il Feature Engineering...")
        data = FeatureEngineer.add_total_spending(data)
        data = FeatureEngineer.parse_cabin(data)            # Deck, CabinNum(Int64), Side (senza drop delle Cabin NaN)
        data = FeatureEngineer.add_group_features(data)     # GroupId, GroupSize(Int64), IsSolo(Int64)
        data = FeatureEngineer.remove_irrelevant_columns(data)

        print("Dati dopo la pulizia:")
        print(data.head())  # Mostra le prime 5 righe del dataset dopo la gestione
        print(f"\nControllo se il dataset contiene valori nulli:\n{data.isnull().sum()}")

        # Salvataggio del dataset pulito
        SaveDB.save_dataset(data)
        print("Il Dataset pulito viene salvato correttamente in 'data/cleaned'.")

        return data