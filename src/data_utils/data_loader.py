import pandas as pd
from typing import Optional

def load_and_validate(filepath):
    """
    Charge et valide le jeu de données
    """
    try :
        df = pd.read_csv(filepath)
        print(f"Données chargées avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        print(f"les colonnes:{list[df.columns]}")
        return df
    except FileNotFoundError:
        print(f"Fichier non trouvé:{filepath}")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        return None