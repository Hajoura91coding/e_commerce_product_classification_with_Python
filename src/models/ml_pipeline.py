import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class MLPipeline:
    """
    Pipeline reproductible pour la classification de produits e-commerce
    """
    # Initialisation de l'objet MLPipeline avec ses attributs de classe
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.vectorizer = None
        self.models = {}
        self.results = {}

    def load_processed_data(self, filepath='ecommerce_data_processed.csv'):
        """
        Charge les données préprocessées
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Données chargées : {df.shape[0]} produits, {df.shape[1]} variables")
            return df
        except FileNotFoundError:
            print("Fichier non trouvé. Assurez-vous d'avoir chargé les données nettoyées.")
            return None

    def create_stratified_split(self, df, text_col, target_col,
                              train_size=0.6, val_size=0.2, test_size=0.2):
        """
        Division stratifiée train/validation/test 60/20/20
        """

        # Nettoyage des données manquantes
        df_clean = df.dropna(subset=[text_col, target_col])

        X = df_clean[text_col]
        y = df_clean[target_col]

        # Division train/temp (60/40)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_size + test_size),
            stratify=y, random_state=self.random_state
        )

        # Division temp en validation/test (20/20)
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1-val_ratio),
            stratify=y_temp, random_state=self.random_state
        )

        print("Division des données :")
        print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

        # Vérification de la stratification
        print("\nDistribution des classes :")
        for dataset_name, y_data in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            distribution = y_data.value_counts(normalize=True).round(3)
            print(f"   {dataset_name}: {distribution.to_dict()}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_features(self, X_train, X_val, X_test, method='tfidf',
                        max_features=10000, ngram_range=(1,2)):
        """
        Préparation des features avec vectorisation TF-IDF : TfidfVectorizer est équivalent a CountVectorizer et TfidfTransformer Tfidf va nous permettre de faire ressortir les mots qui caractérisent chacun des catégories de produits et donc on va pouvoir les comparer. on va appliquer par la suite un tsne pour voir les similarités entre classes
        """
        print(f"\nPréparation des features - Méthode: {method}")

        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )

            # Entraînement sur train seulement
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_val_vec = self.vectorizer.transform(X_val)
            X_test_vec = self.vectorizer.transform(X_test)

            print(f"Vocabulaire: {len(self.vectorizer.vocabulary_)} mots")
            print(f"Matrice train: {X_train_vec.shape}")
            print(f"Densité: {X_train_vec.nnz/X_train_vec.size:.4f}")

            return X_train_vec, X_val_vec, X_test_vec

        else:
            raise ValueError(f"Méthode {method} non implémentée")
