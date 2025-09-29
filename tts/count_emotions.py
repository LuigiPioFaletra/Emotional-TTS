import os
from collections import Counter

def emotions_counter(cartella, etichette):
    """
    Conta i file per ciascuna etichetta.
    Restituisce il totale di file etichettati.
    """
    cnt = Counter({e: 0 for e in etichette})
    tot = 0
    for fn in os.listdir(cartella):
        nome = fn.lower()
        found = False
        for emo in etichette:
            if emo in nome:
                cnt[emo] += 1
                found = True
        if found:
            tot += 1
    return cnt, tot


# Esempio di chiamata
etichette = ['col', 'dis', 'gio', 'neu', 'pau', 'rab', 'sor', 'tri']
dataset = ['./DEMoS/CORPUS', './EmoFilm/wav_corpus', './EMOVO/corpus', './Emozionalmente/wav']
for ds in dataset:
    c, t = emotions_counter(ds, etichette)
    print(f'Conteggi emozioni in {ds}:', c)
    print(f'File etichettati in {ds}:', t, end='\n\n')
