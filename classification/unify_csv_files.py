import os
import pandas as pd

def generate_metadata(audio_folder, csv_files, output_file="metadata.csv"):
    """
    Genera un file 'metadata.csv' unificato a partire da una lista di CSV
    contenenti informazioni sui file audio, verificando la presenza fisica
    degli audio e la presenza di trascrizioni con token emozionali.
    """
    out_lines = []
    missing_audio = []
    missing_transcription = []
    combined_df = pd.DataFrame()
    
    for csv_file in csv_files:
        print(f"Caricamento {csv_file}...")
        try:
            df = pd.read_csv(csv_file, sep=';')
            df = df.drop(columns=["transcription", "emotion"], errors="ignore")
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Errore durante la lettura di {csv_file}: {e}")
            continue

    for idx, row in combined_df.iterrows():
        dataset_name = str(row.get("dataset_name", "")).strip()
        file_name = str(row.get("file_name", "")).strip()
        speaker_id = str(row.get("speaker_id", "")).strip()
        token_text = str(row.get("transcription_with_token", "")).strip()

        if not file_name:
            continue
        if not token_text:
            missing_transcription.append(file_name)
            continue

        full_path = os.path.join(audio_folder, file_name)
        if not os.path.isfile(full_path):
            missing_audio.append(file_name)
            continue

        line = f"{dataset_name};{file_name};{speaker_id};{token_text}"
        out_lines.append(line)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"\n'{output_file}' creato con {len(out_lines)} righe valide.")
    print(f"File audio mancanti: {len(missing_audio)}")
    print(f"Trascrizioni mancanti o vuote: {len(missing_transcription)}")

    if missing_audio:
        with open("missing_audio.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(missing_audio))
        print("Log audio mancante: missing_audio.txt")
    if missing_transcription:
        with open("missing_transcriptions.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(missing_transcription))
        print("Log trascrizioni mancanti: missing_transcriptions.txt")


# Esempio di chiamata
csv_sorgenti = [
    "./CSV/DEMOS.csv",
    "./CSV/EmoFilm.csv",
    "./CSV/EMOVO.csv",
    "./CSV/Emozionalmente.csv"
]

generate_metadata(audio_folder="./Dataset_completo", csv_files=csv_sorgenti)
print("File metadata.csv generato.")