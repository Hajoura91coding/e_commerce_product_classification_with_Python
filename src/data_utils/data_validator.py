import pandas as pd

def get_data_summary(df):
    """
    Résumé statistique du jeu de données
    """
    summary1 = {
        'Nombre de lignes': df.shape[0],
        'Nombre de colonnes': df.shape[1],
        '% de valeurs manquantes': round(df.isnull().sum().sum() / df.shape[0] * 100, 2),
    }
    stats = []
    for col in df.columns:
        stats.append((col,
                     df[col].nunique(),
                     df[col].isnull().sum()*100/df.shape[0],
                     df[col].dtype))
    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'type'])
    stats_df.sort_values('Percentage of missing values', ascending=True)

    return pd.DataFrame([summary1]), stats_df

def analyze_missing_values(df):
    """
    Analyse détaillée des valeurs manquantes
    """
    missing_values = df.isnull().sum()
    missing_percent = 100 * missing_values/len(df)

    missing_df = pd.DataFrame({
        'Colonne': missing_values.index,
        'Valeurs manquantes': missing_values.values,
        'Pourcentage': missing_percent.values
    })
    return missing_df[missing_df['Valeurs manquantes'] > 0].sort_values('Pourcentage', ascending=False)