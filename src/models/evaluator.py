import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Statistiques
from scipy import stats

from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import  StratifiedKFold
import time

class ModelEvaluator:
    """
    Classe pour l'évaluation complète des modèles
    """

    def __init__(self):
        self.results = {}

    def train_and_evaluate(self, model, X_train, X_val, y_train, y_val, model_name):
        """
        Entraîne et évalue un modèle
        """
        print(f"\n Entraînement {model_name}")
        start_time = time.time()

        # Entraînement
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Prédictions
        start_pred = time.time()
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
        pred_time = time.time() - start_pred

        # Métriques
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision_macro': precision_score(y_val, y_pred, average='macro'),
            'precision_weighted': precision_score(y_val, y_pred, average='weighted'),
            'recall_macro': recall_score(y_val, y_pred, average='macro'),
            'recall_weighted': recall_score(y_val, y_pred, average='weighted'),
            'f1_macro': f1_score(y_val, y_pred, average='macro'),
            'f1_weighted': f1_score(y_val, y_pred, average='weighted'),
            'train_time': train_time,
            'pred_time': pred_time
        }

        # Stockage des résultats
        self.results[model_name] = {
            'model': model,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_val, y_pred)
        }

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Temps entraînement: {train_time:.2f}s")
        print(f"Temps prédiction: {pred_time:.2f}s")

        return model

    def compare_models(self):
        """
        Compare tous les modèles entraînés
        """
        if not self.results:
            print("Aucun modèle à comparer")
            return

        # Création du tableau comparatif
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Modèle': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'F1 Macro': f"{metrics['f1_macro']:.4f}",
                'F1 Weighted': f"{metrics['f1_weighted']:.4f}",
                'Temps Train (s)': f"{metrics['train_time']:.2f}",
                'Temps Pred (s)': f"{metrics['pred_time']:.4f}"
            })

        comparison_df = pd.DataFrame(comparison_data)

        print("\n COMPARAISON DES MODÈLES")
        print("=" * 60)
        print(comparison_df.to_string(index=False))

        # Identification du meilleur modèle
        best_model = max(self.results.keys(),
                        key=lambda x: self.results[x]['metrics']['f1_weighted'])

        print(f"\n Meilleur modèle (F1 weighted): {best_model}")
        print(f"   Performance: {self.results[best_model]['metrics']['f1_weighted']:.4f}")

        return comparison_df, best_model

    def plot_confusion_matrices(self, class_names):
        """
        Affiche les matrices de confusion pour tous les modèles
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']

            # Normalisation
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(cm_norm, annot=True, fmt='.2f',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[idx], cmap='Blues')
            axes[idx].set_title(f'{model_name}\nAccuracy: {self.results[model_name]["metrics"]["accuracy"]:.3f}')
            axes[idx].set_xlabel('Prédiction')
            axes[idx].set_ylabel('Vérité')

        plt.tight_layout()
        plt.show()

def robust_cv_evaluation(model, X, y, cv=5, random_state=42):
    """
    Évaluation robuste avec validation croisée et intervalles de confiance
    """
    print(f"\n Validation croisée stratifiée ({cv} folds)")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Métriques à calculer
    metrics = {
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'f1_weighted': []
    }

    fold = 1
    for train_idx, val_idx in skf.split(X, y):
        print(f"   Fold {fold}/{cv}...", end=" ")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Entraînement
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_train_fold, y_train_fold)

        # Prédiction
        y_pred = model_copy.predict(X_val_fold)

        # Calcul des métriques
        metrics['accuracy'].append(accuracy_score(y_val_fold, y_pred))
        metrics['precision_macro'].append(precision_score(y_val_fold, y_pred, average='macro'))
        metrics['recall_macro'].append(recall_score(y_val_fold, y_pred, average='macro'))
        metrics['f1_macro'].append(f1_score(y_val_fold, y_pred, average='macro'))
        metrics['f1_weighted'].append(f1_score(y_val_fold, y_pred, average='weighted'))

        print(f"F1: {metrics['f1_weighted'][-1]:.3f}")
        fold += 1

    # Calcul des statistiques
    stats_summary = {}
    for metric_name, scores in metrics.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Intervalle de confiance à 95%
        confidence_interval = stats.t.interval(
            0.95, len(scores)-1,
            loc=mean_score,
            scale=stats.sem(scores)
        )

        stats_summary[metric_name] = {
            'mean': mean_score,
            'std': std_score,
            'ci_lower': confidence_interval[0],
            'ci_upper': confidence_interval[1],
            'scores': scores
        }

    return stats_summary