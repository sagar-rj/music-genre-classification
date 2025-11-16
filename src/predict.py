import librosa
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("../models/genre_cnn_model.h5")
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

def predict_genre(audio_path):
    signal, sr = librosa.load(audio_path, duration=30)
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.shape[1] < 128:
        mel_spec_db = np.pad(mel_spec_db, ((0,0),(0,128-mel_spec_db.shape[1])), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :128]
    mel_spec_db = mel_spec_db[np.newaxis, ..., np.newaxis]
    prediction = model.predict(mel_spec_db)
    return genres[np.argmax(prediction)]
