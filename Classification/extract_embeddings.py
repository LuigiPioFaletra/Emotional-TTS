import librosa
import numpy as np
import os
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModel

class AudioEmbeddings:
    """
    Classe per l'estrazione di embeddings da file audio tramite Wav2Vec2.
    """
    def __init__(self, model_name='facebook/wav2vec2-large-xlsr-53', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract(self, speech, sampling_rate=16000):
        inputs = self.processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding="longest")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

def extract_audio_embeddings(csv_input, csv_output, npy_output, audio_base_path):
    """
    Estrae embeddings audio per tutti i file del dataset e salva CSV aggiornato e array .npy.
    """
    df = pd.read_csv(csv_input)
    ae = AudioEmbeddings()
    embeddings_list = []
    for idx, row in df.iterrows():
        audio_file = os.path.join(audio_base_path, row['file_name'])
        if not os.path.exists(audio_file):
            print(f"[ATTENZIONE] File non trovato: {audio_file}")
            embeddings_list.append(np.zeros(1024))  # placeholder
            continue
        audio_vec, sr = librosa.load(audio_file, sr=16000)
        embeddings = ae.extract(audio_vec)
        embeddings_list.append(list(embeddings[0]))
        if idx % 10 == 0:
            print(f"File processati: {idx}")
    df['embeddings'] = embeddings_list
    df.to_csv(csv_output, index=False, sep="\t")
    np.save(npy_output, np.array(embeddings_list))
    print(f"CSV salvato: {csv_output}")
    print(f"Embeddings salvati in .npy: {npy_output}")


# Esempio di chiamata
extract_audio_embeddings(
    csv_input="metadata_new.csv",
    csv_output="metadata_sep.csv",
    npy_output="embeddings.npy",
    audio_base_path="./dataset"
)
