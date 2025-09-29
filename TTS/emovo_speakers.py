import re
import pandas as pd

def emovo_speaker_id_extraction(nome_file):
    """
    Estrae speaker_id dai file EMOVO.
    """
    match = re.match(r'^[a-z]+-([mf]\d)-[a-z]+\d\.wav$', nome_file)
    return match.group(1) if match else "NON TROVATO"

    
# Esempio di chiamata
df = pd.read_csv("./CSV/EMOVO_incomplete.csv", sep=';', encoding='utf-8')
df['speaker_id'] = df['file_name'].apply(emovo_speaker_id_extraction)
df['dataset_name'] = 'EMOVO'
df.to_csv("./CSV/EMOVO_complete.csv", sep=';', index=False, encoding='utf-8')