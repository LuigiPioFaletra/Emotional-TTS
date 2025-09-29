import pandas as pd

def add_transcription_with_token(input_csv, output_csv, emotion_token_map, sep=';'):
    """
    Aggiunge una colonna 'transcription_with_token' concatenando il token emozionale all'inizio della trascrizione basata sulla colonna 'emotion'.
    """
    df = pd.read_csv(input_csv, sep=sep, encoding='utf-8')
    transcription_col = None
    for col in df.columns:
        if "text" in col.lower() or "trascr" in col.lower():
            transcription_col = col
            break
    if transcription_col is None:
        raise ValueError("Nessuna colonna trascrizione trovata nel dataset!")
    df["transcription"] = df[transcription_col]
    df["token"] = df["emotion"].map(emotion_token_map).fillna("")
    df["transcription_with_token"] = (df["token"] + " " + df["transcription"]).str.strip()
    df.to_csv(output_csv, sep=sep, index=False, encoding='utf-8')


# Esempio di chiamata
input_files = [
    './CSV/DEMOS_complete.csv',
    './CSV/EmoFilm_complete.csv',
    './CSV/EMOVO_complete.csv',
    './CSV/Emozionalmente_complete.csv'
]
output_files = [
    './CSV/DEMOS.csv',
    './CSV/EmoFilm.csv',
    './CSV/EMOVO.csv',
    './CSV/Emozionalmente.csv'
]
EMOTION_TOKEN_MAP = {
    "gioia": "<gioia>",
    "rabbia": "<rabbia>",
    "disgusto": "<disgusto>",
    "paura": "<paura>",
    "neutrale": "<neutrale>",
    "tristezza": "<tristezza>",
    "sorpresa": "<sorpresa>",
    "colpa": "<colpa>",
}
for input_csv, output_csv in zip(input_files, output_files):
    add_transcription_with_token(
        input_csv=input_csv,
        output_csv=output_csv,
        emotion_token_map=EMOTION_TOKEN_MAP,
        sep=';'
    )