# Music Genre Classification

A small Python project to classify music tracks into genres using classical ML (Random Forest) and a CNN (mel-spectrogram based). Includes training scripts and a Streamlit inference app.

## Repo layout
- [app/streamlit_app.py](app/streamlit_app.py) — Streamlit UI and inference; uses the [`predict_genre`](app/streamlit_app.py) function and the loaded [`model`](app/streamlit_app.py).
- [models/genre_cnn_model.h5](models/genre_cnn_model.h5) — example CNN model checkpoint (may be replaced by the serialized model used by the app).
- [src/train_cnn.py](src/train_cnn.py) — CNN training pipeline (defines `genres` as [`genres`](src/train_cnn.py) and the CNN [`model`](src/train_cnn.py)).
- [src/train_rf.py](src/train_rf.py) — Random Forest training pipeline (produces a `model`; training arrays are [`X`](src/train_rf.py) and [`y`](src/train_rf.py)).
- [src/predict.py](src/predict.py) — prediction utilities used by inference (see file for details).
- requirements.txt — Python dependencies.

## Project layout (actual files in this repository)
This repository currently contains the following files and folders under the project root:

- readme.md
- src/
  - predict.py         — utilities for loading audio and running model inference
  - train_cnn.py       — CNN training script (mel-spectrogram based)
  - train_rf.py        — Random Forest training script (MFCC based)

Use the src/ scripts to train models and the predict utilities for inference. If you expect additional files (app/, models/, requirements.txt) add them to the repo or update paths in the scripts.

## Quickstart

1. Create and activate a virtual environment, then install deps:
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

2. Train a model
- Train the CNN (mel-spectrogram based):
```bash
python src/train_cnn.py
```
This script downloads the GTZAN dataset via the helper used in the script and saves a trained model (see [`src/train_cnn.py`](src/train_cnn.py) for details).

- Train the Random Forest (MFCC features):
```bash
python src/train_rf.py
```
The script extracts MFCC features and saves a pickle model (see [`src/train_rf.py`](src/train_rf.py)).

3. Run the Streamlit app for inference:
```bash
streamlit run app/streamlit_app.py
```
The app uses the [`predict_genre`](app/streamlit_app.py) function and the loaded [`model`](app/streamlit_app.py). Upload a WAV/MP3 file to get a predicted genre.

## Dataset

This project uses the GTZAN dataset. The training scripts expect the dataset to be downloaded and the path to the "genres" folder provided as `DATASET_PATH`. Example download snippet used in the project:

```python
path = kagglehub.dataset_download("carlthome/gtzan-genre-collection")
DATASET_PATH = os.path.join(path, 'genres')
```

Ensure `DATASET_PATH` points to the extracted `genres/` directory (it contains one subfolder per genre) before running `src/train_cnn.py` or `src/train_rf.py`.

## Files and important symbols
- [`predict_genre`](app/streamlit_app.py) — preprocessing and model inference for uploaded audio ([app/streamlit_app.py](app/streamlit_app.py)).
- [`genres`](src/train_cnn.py) — genre list used across training scripts ([src/train_cnn.py](src/train_cnn.py)).
- Training artifacts:
  - CNN model saved under [models/genre_cnn_model.h5](models/genre_cnn_model.h5) (or other model files created by training scripts).
  - Random Forest model saved as a pickle by [src/train_rf.py](src/train_rf.py) (variable [`model`](src/train_rf.py)).

## Notes
- Training scripts use `librosa` to load audio and extract features; ensure required audio backends are available.
- Dataset download in training scripts relies on the helper invoked in the script — check the top of [src/train_cnn.py](src/train_cnn.py) and [src/train_rf.py](src/train_rf.py) for how the dataset is fetched.
- If the Streamlit app errors loading a model, verify the model filename and path used in [app/streamlit_app.py](app/streamlit_app.py).

## Troubleshooting
- If audio fails to load or feature shapes mismatch, inspect preprocessing in [`predict_genre`](app/streamlit_app.py) and feature extraction in [src/train_cnn.py](src/train_cnn.py) / [src/train_rf.py](src/train_rf.py).
- For training dataset issues, check the dataset path resolved in [src/train_cnn.py](src/train_cnn.py).

## License & Attribution
See repository root for license information and dataset attribution.



