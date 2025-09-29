import pandas as pd

def add_emotions_column(csv_input, csv_output):
    """
    Aggiunge la colonna 'emotion' al CSV a partire dai token
    iniziali delle trascrizioni, riorganizza le colonne e salva il nuovo CSV.
    """
    token_to_emotion = {
        "<col>": "colpa",
        "<dis>": "disgusto",
        "<gio>": "gioia",
        "<neu>": "neutro",
        "<pau>": "paura",
        "<rab>": "rabbia",
        "<sor>": "sorpresa",
        "<tri>": "tristezza"
    }

    def extract_emotion(transcription):
        for token, emotion in token_to_emotion.items():
            if transcription.startswith(token):
                return emotion
        return "unknown"

    df = pd.read_csv(csv_input, sep="\t")
    df["emotion"] = df["transcription_with_token"].apply(extract_emotion)
    df = df[["dataset_name", "file_name", "speaker_id", "emotion", "transcription_with_token"]]
    df.to_csv(csv_output, index=False)
    print(f"Colonna 'emotion' aggiunta correttamente e CSV salvato in: {csv_output}")


# Esempio di chiamata
add_emotions_column(
    csv_input="./metadata.csv",
    csv_output="./metadata_new.csv"
)
