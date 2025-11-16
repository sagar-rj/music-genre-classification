import kagglehub
import os
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

path = kagglehub.dataset_download("carlthome/gtzan-genre-collection")
DATASET_PATH = os.path.join(path, 'genres')
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
X, y = [], []

for genre in genres:
    genre_path = os.path.join(DATASET_PATH, genre)
    if not os.path.exists(genre_path):
        continue
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        signal, sr = librosa.load(file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        X.append(mfcc_mean)
        y.append(genre)

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

with open("../models/genre_rf_model.pkl", "wb") as f:
    pickle.dump(model, f)