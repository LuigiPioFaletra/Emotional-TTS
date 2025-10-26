# Emotional-TTS - Emotional Speech Synthesis for the Italian Language

This repository contains the source code, scripts, and resources used for the Master's thesis  
**"Dare voce alle emozioni: TTS emozionale per la lingua italiana"**  
(*Giving Voice to Emotions: Emotional TTS for the Italian Language*)  
by **Luigi Pio Faletra** - Master's Degree in Artificial Intelligence and Security Engineering,  
"Kore" University of Enna, Academic Year 2024/2025.

---

## Project Overview

This project explores **emotional speech synthesis (Emotional Text-to-Speech, TTS)** for the **Italian language**, an area that remains relatively underexplored compared to high-resource languages such as English.  
The main goal is to assess whether **large language model-based TTS systems**, fine-tuned efficiently, can generate **natural and expressive speech** capable of conveying distinct emotional states.

The experimental work focuses on adapting the **Orpheus Multilingual** model through **LoRA fine-tuning** and the **SNAC vocoder**, leveraging multiple Italian emotional speech corpora.

---

## Methodology and Pipeline

The research integrates data preparation, emotion classification, and speech synthesis into a unified neural workflow.  
**The preprocessing, fine-tuning, and inference steps were carried out using a modified Colab notebook**, available [here](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb).

1. **Data Preprocessing**
   - Integration of DEMoS, EMOVO, EmoFilm, and Emozionalmente corpora  
   - Emotion balancing and metadata extraction  
   - Generation of unified CSV datasets with emotional tokens  

2. **Emotion Classification**
   - Training an MLP-based classifier on audio embeddings  
   - Evaluation through confusion matrices and accuracy reports  

3. **Fine-tuning and Inference**
   - Adapting the Orpheus multilingual TTS model to Italian  
   - Conditioning the model on emotion tokens  
   - Employing the SNAC vocoder for waveform generation  

4. **Evaluation**
   - **Quantitative:** training loss, WER, training time  
   - **Qualitative:** human listening tests and emotion recognition performance  

---

## Key Findings

- Strong ability to represent **basic emotions** such as *anger* and *surprise*.  
- Noticeable improvements with extended training sessions.  
- **Orpheus** demonstrates advantages in **prosodic control**, **computational efficiency**, and **multilingual scalability**, though residual limitations persist in **naturalness** and **generalization**.

---

## Repository Structure

```
main_repository/
│
├── classification/
│   ├── extract_embeddings.py
│   ├── mlp_training.py
│
├── tts/
│   ├── add_emotions_column.py
│   ├── add_tokens.py
│   ├── count_emotions.py
│   ├── demos_speakers.py
│   ├── demos_transcriptions_emotions.ipynb
│   ├── emofilm_speakers.py
│   ├── emofilm_transcriptions_emotions.py
│   ├── emovo_speakers.py
│   ├── emovo_transcriptions_emotions.py
│   ├── emozionalmente_csv_file.py
│   ├── split_dataset.py
│   ├── unify_csv_files.py
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## Technologies and Resources

- **Python 3.10+**
- **PyTorch**
- **Hugging Face Transformers**
- **SNAC Vocoder**
- **LoRA Fine-Tuning**
- **Italian Emotional Speech Datasets**: DEMoS, EMOVO, EmoFilm, Emozionalmente  

---

## Academic Reference

If you use this repository or its findings, please cite the following thesis:

> **Faletra, L. P. (2025).**  
> *Dare voce alle emozioni: TTS emozionale per la lingua italiana.*  
> "Kore" University of Enna - Master's Thesis in Artificial Intelligence and Security Engineering.

---

## Future Work

- Expansion of emotional datasets with natural (non-acted) recordings.  
- Integration of **advanced neural vocoders** (HiFi-GAN, BigVGAN).  
- Development of **objective perceptual metrics** for emotion evaluation.  
- Application in **empathetic voice assistants** and **automatic dubbing systems**.  

---

## Contact

**Author:** Luigi Pio Faletra  
**Supervisor:** Prof. Moreno La Quatra  
[luigipio.faletra@unikorestudent.it](mailto:luigipio.faletra@unikorestudent.it)  
[GitHub Repository](https://github.com/LuigiPioFaletra/Emotional-TTS)
