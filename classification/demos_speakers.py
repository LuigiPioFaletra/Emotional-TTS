import pandas as pd

def demos_speaker_id_extraction(nome_file):
    """
    Estrae speaker_id dai file DEMoS.
    """
    parts = nome_file.split('_')
    if parts[0] in ('NP', 'PR'):
        return f"{parts[1]}_{parts[2]}"
    else:
        return f"{parts[0]}_{parts[1]}"


# Esempio di chiamata
df = pd.read_csv("./CSV/DEMoS_incomplete.csv", sep=';', encoding='utf-8')
df['speaker_id'] = df['file_name'].apply(demos_speaker_id_extraction)
df['dataset_name'] = 'DEMoS'
df.to_csv("./CSV/DEMoS_complete.csv", sep=';', index=False, encoding='utf-8')
print("ID speaker estratti per DEMoS")
