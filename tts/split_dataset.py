import pandas as pd
import random

def prepare_dataset_with_split(csv_input, csv_output, val_per_class=220, test_per_class=220, seed=3407):
    """
    Prepara il dataset finale aggiungendo lo split train/validation/test,
    riorganizzando le colonne e verificando i conteggi per split ed emozione.
    """
    df = pd.read_csv(csv_input, sep=",")
    assert "emotion" in df.columns, "Colonna 'emotion' mancante nel CSV."
    assert "speaker_id" in df.columns, "Colonna 'speaker_id' mancante nel CSV."

    random.seed(seed)
    val_idx, test_idx = [], []

    for emo, group in df.groupby("emotion"):
        indices = list(group.index)
        random.shuffle(indices)
        if len(indices) < val_per_class + test_per_class:
            raise ValueError(
                f"La classe '{emo}' ha solo {len(indices)} esempi, servono almeno {val_per_class + test_per_class}."
            )
        val_idx.extend(indices[:val_per_class])
        test_idx.extend(indices[val_per_class:val_per_class + test_per_class])

    df["split"] = "train"
    df.loc[val_idx, "split"] = "validation"
    df.loc[test_idx, "split"] = "test"

    cols = ["dataset_name", "file_name", "speaker_id", "split", "emotion", "transcription_with_token"]
    df = df[cols]
    df.to_csv(csv_output, index=False, sep=",")

    print(f"CSV finale salvato in: {csv_output}")

    def conteggi(df, name):
        print(f"\n=== {name} ===")
        counts = df.groupby(["split", "emotion"]).size().reset_index(name="count")
        pivot = counts.pivot(index="split", columns="emotion", values="count").fillna(0).astype(int)
        print(pivot)
        print("\nTotali per split:")
        print(df["split"].value_counts())

    conteggi(df, "Dataset finale")


# Esempio di chiamata
prepare_dataset_with_split(
    csv_input="./metadata_new.csv",
    csv_output="./metadata_final.csv",
    val_per_class=220,
    test_per_class=220,
    seed=3407
)