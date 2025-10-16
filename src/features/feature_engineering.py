

def add_text_features(df, text_col):
    """
    Ajoute des caractéristiques textuelles
    """
    df['word_count'] = df[text_col].str.split().str.len()
    df['char_count'] = df[text_col].str.len()
    df['avg_word_length'] = df['char_count'] / df['word_count']

    return df
