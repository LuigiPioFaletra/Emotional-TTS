import os
import csv

def create_emovo_metadata(input_dir, output_csv):
    """
    Estrae metadati dai file audio EMOVO e salva un CSV con le colonne:
    nome del file, trascrizione associata (in base al codice frase), emozione (in italiano,
    es: 'sorpresa', 'rabbia', 'gioia').
    """
    frasi = {
        "b1": "Gli operai si alzano presto.",
        "b2": "I vigili sono muniti di pistola.",
        "b3": "La cascata fa molto rumore.",
        "l1": "L'autunno prossimo Tony partirà per la Spagna: nella prima metà di ottobre.",
        "l2": "Ora prendo la felpa di là ed esco per fare una passeggiata.",
        "l3": "Un attimo dopo s'è incamminato... ed è inciampato.",
        "l4": "Vorrei il numero telefonico del Signor Piatti.",
        "n1": "La casa forte vuole col pane.",
        "n2": "La forza trova il passo e l'aglio rosso.",
        "n3": "Il gatto sta scorrendo nella pera.",
        "n4": "Insalata pastasciutta coscia d'agnello limoncello.",
        "n5": "Uno quarantatré dieci mille cinquantasette venti.",
        "d1": "Sabato sera cosa farà?",
        "d2": "Porti con te quella cosa?"
    }

    abbreviazioni = {
        "dis": "disgusto",
        "gio": "gioia",
        "neu": "neutrale",
        "pau": "paura",
        "rab": "rabbia",
        "sor": "sorpresa",
        "tri": "tristezza"
    }

    dati = []
    for nome_file in os.listdir(input_dir):
        if nome_file.endswith(".wav"):
            parti = nome_file[:-4].split("-")
            if len(parti) == 3:
                abbrev, attore, codice_frase = parti
                frase = frasi.get(codice_frase, "FRASE NON TROVATA")
                emozione = abbreviazioni.get(abbrev.lower(), "EMOZIONE NON RICONOSCIUTA")
                dati.append([nome_file, frase, emozione])
            else:
                print(f"Formato non riconosciuto: {nome_file}")

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["file_name", "transcription", "emotion"])
        writer.writerows(dati)

    print(f"CSV generato: {output_csv} con {len(dati)} righe.")


# Esempio di chiamata
create_emovo_metadata(
    input_dir="./EMOVO/corpus",
    output_csv="./CSV/EMOVO_incomplete.csv"
)
