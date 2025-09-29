import pandas as pd

def create_emofilm_metadata(input_csv, output_csv, sep=';'):
    """
    Estrae l'emozione dal nome del file e genera un file CSV con le colonne: file_name, transcription, emotion.
    """
    abbr_map = {
        'ans': 'paura',
        'dis': 'disgusto',
        'gio': 'gioia',
        'rab': 'rabbia',
        'tri': 'tristezza'
    }
    df = pd.read_csv(input_csv, sep=sep, encoding='utf-8')
    df['Emozione'] = df['file_name'].apply(
        lambda x: abbr_map.get(str(x)[2:5].lower(), 'EMOZIONE NON RICONOSCIUTA')
    )
    df_out = pd.DataFrame({
        'file_name': df['file_name'].str.strip(),
        'transcription': df['manual correction from automatic transcription made with Wav2vec2-large-xlsr-53'].str.strip(),
        'emotion': df['Emozione']
    })
    df_out.to_csv(output_csv, sep=sep, index=False, encoding='utf-8')
    print(f"Generato: {output_csv} ({len(df_out)} righe)")


# Esempio di chiamata
create_emofilm_metadata(
    input_csv="./EmoFilm/transcriptions_it.csv",
    output_csv="./CSV/EmoFilm_incomplete.csv"
)