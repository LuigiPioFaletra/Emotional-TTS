import csv

def create_emozionalmente_metadata(input_csv, output_csv, emotion_map):
    """
    Estrae metadati dal dataset Emozionalmente e genera un CSV con le colonne:
    dataset_name (valore fisso: "Emozionalmente"), file_name, transcription,
    speaker_id (ex 'actor'), emotion (in italiano).
    """
    righe = []
    with open(input_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            nome_file = r.get("file_name", "").strip()
            emozione_raw = r.get("emotion_expressed", "").strip().lower()
            frase = r.get("sentence", "").strip()
            speaker_id = r.get("actor", "").strip()
            label_it = emotion_map.get(emozione_raw)
            if not label_it:
                print(f"Emozione non riconosciuta: '{emozione_raw}'")
                continue
            righe.append({
                "dataset_name": "Emozionalmente",
                "file_name": nome_file,
                "transcription": frase,
                "speaker_id": speaker_id,
                "emotion": label_it
            })

    fieldnames = ["dataset_name", "file_name", "transcription", "speaker_id", "emotion"]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(righe)

    print(f"CSV generato: {output_csv} ({len(righe)} righe)")


# Esempio di chiamata
EMO_MAP = {
    "anger": "rabbia",
    "disgust": "disgusto",
    "fear": "paura",
    "joy": "gioia",
    "neutrality": "neutrale",
    "sadness": "tristezza",
    "surprise": "sorpresa"
}

create_emozionalmente_metadata(
    input_csv="./Emozionalmente/metadata/samples.csv",
    output_csv="./CSV/Emozionalmente_complete.csv",
    emotion_map=EMO_MAP
)