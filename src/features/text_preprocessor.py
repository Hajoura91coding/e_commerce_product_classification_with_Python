
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from typing import Literal

class TextPreprocessor:
    """
    Classe pour le préprocessing de texte avec différentes méthodes
    """

    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = SnowballStemmer(language)
        self.lemmatizer = WordNetLemmatizer()

        # Mots spécifiques à l'e-commerce à supprimer
        self.ecommerce_stopwords = {
            'rs', 'product', 'buy', 'products', 'free', 'flipkart',
            'features', 'delivery', 'shipping', 'cash', 'day',
            'genuine', 'price', 'prices', 'package', 'rupee', 'rupees'
        }
        self.stop_words.update(self.ecommerce_stopwords)

    def clean_text_basic(self, text):
        """
        Nettoyage de base du texte
        """
        if pd.isna(text):
            return ""

        # Conversion en minuscules et suppression des caractères non-alphabétiques
        text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())

        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_stopwords(self, text):
        """
        Supprime les mots vides
        """
        words = text.split()
        return ' '.join([word for word in words if word not in self.stop_words])

    def stem_text(self, text):
        """
        Applique le stemming
        """
        words = text.split()
        return ' '.join([self.stemmer.stem(word) for word in words])

    def lemmatize_text(self, text):
        """
        Applique la lemmatisation
        """
        words = text.split()
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])

    def preprocess_text(self, text, method='stem'):
        """
        Pipeline complet de préprocessing
        """
        # Nettoyage de base
        text = self.clean_text_basic(text)

        # Suppression des mots vides
        text = self.remove_stopwords(text)

        # Application de la méthode choisie
        if method == 'stem':
            text = self.stem_text(text)
        elif method == 'lemma':
            text = self.lemmatize_text(text)

        return text

