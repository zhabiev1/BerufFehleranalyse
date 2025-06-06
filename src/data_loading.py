import pandas as pd
import pickle
import os
from typing import Dict, List

def load_esco_data(file_path: str = 'data/occupations_en.csv') -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ESCO-Datei nicht gefunden: {file_path}")

    try:
        esco_data = pd.read_csv(file_path)
        if esco_data.empty:
            raise pd.errors.EmptyDataError("ESCO-Datei ist leer.")
        return esco_data
    except pd.errors.EmptyDataError as e:
        raise
    except Exception as e:
        raise

def load_predictions(prediction_dir: str = 'data/predictions') -> Dict[str, List]:
    prediction_files = [
        'decorte_esco_predictions_linear.pkl',
        'decorte_predictions_linear.pkl',
        'karrierewege_cp_predictions_linear.pkl',
        'karrierewege_occ_predictions_linear.pkl',
        'karrierewege_predictions_linear.pkl'
    ]

    if not os.path.exists(prediction_dir):
        raise FileNotFoundError(f"Vorhersageverzeichnis nicht gefunden: {prediction_dir}")

    predictions_data = {}
    for file_name in prediction_files:
        file_path = os.path.join(prediction_dir, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Vorhersagedatei nicht gefunden: {file_path}")

        try:
            with open(file_path, 'rb') as file:
                predictions_data[file_name] = pickle.load(file)
        except pickle.PickleError as e:
            raise
        except Exception as e:
            raise
    
    if not predictions_data:
        raise ValueError("Keine Vorhersagen für die Analyse verfügbar.")
    
    return predictions_data