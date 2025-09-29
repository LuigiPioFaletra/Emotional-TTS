import re
import pandas as pd

def emofilm_speaker_id_extraction(nome_file):
    """
    Estrae speaker_id dai file EmoFilm.
    """
    match = re.match(r'^([fm])_([a-z]{3})(\d+)[a-z]{3}\.wav$', nome_file)
    return f"{match.group(1)}_{match.group(3)}" if match else "NON_TROVATO"

    
# Esempio di chiamata
df = pd.read_csv("./CSV/EmoFilm_incomplete.csv", sep=';', encoding='utf-8')
df['speaker_id'] = df['file_name'].apply(emofilm_speaker_id_extraction)
df['dataset_name'] = 'EmoFilm'
df.to_csv("./CSV/EmoFilm_complete.csv", sep=';', index=False, encoding='utf-8')