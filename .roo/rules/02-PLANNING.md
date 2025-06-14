# 🧠 Project: Audio Transcription & Speaker Diarization Pipeline

This project is a modular, GPU-accelerated audio processing pipeline that automates the transcription of spoken audio and optionally identifies who said what using speaker diarization. It is designed for flexibility, reproducibility, and easy integration into enterprise or research workflows.

## 🎯 Key Features

- **Automatic Speech Recognition (ASR)** using NVIDIA NeMo’s state-of-the-art conformer model
- **Speaker Diarization** for segmenting and labeling who spoke when, add a command line attribute that defaults to True
- **Custom Speaker Mapping** via a simple YAML config
- **Structured Output** in both human-readable text and machine-parsable JSON
- **Modular Components** for each stage: audio loading, transcription, diarization, formatting, and output writing
- **Optimized for GPU** execution (CUDA 12.8, dual RTX Titan support)
- make sure to cache results to improve performance

## 🧩 Architecture

This pipeline consists of 5 processing stages:

| Stage           | Function                             |
|-----------------|--------------------------------------|
| `load_audio`    | Loads and prepares the audio file    |
| `transcribe`    | Converts speech to text (ASR)        |
| `diarize`       | Segments by speaker (optional)       |
| `format`        | Combines transcript + speaker info   |
| `write_output`  | Saves final results (`.json`, `.txt`) |

## 📁 Inputs and Outputs

- **Input:** a directory with `.wav` audio files
- **Config:** `config.yaml` for diarization and speaker name mapping
- **Output:** per audio file create a directory named accordingly with:
  - `transcript.json` — structured output with speaker labels and timestamps
  - `transcript.txt` — readable transcript with speaker attribution
  - `transcript_attributed.txt` the transcript with speaker attribution, like in a theater play, for example: 
  SPEAKER_00: this is what SPEAKER_00 said
  SPEAKER_01: this is what SPEAKER_01 said
  SPEAKER_00: this is what SPEAKER_00 also said

## 🚀 Use Cases

- Conference recording transcription
- Podcast indexing and search
- Legal depositions or interview processing
- Multilingual voice interface logging (with future ASR model extensions)