import os
import librosa
import numpy as np
import kagglehub
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

path = kagglehub.dataset_download("carlthome/gtzan-genre-collection")
DATASET_PATH = os.path.join(path, 'genres')
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
X, y = [], []

for genre_idx, genre in enumerate(genres):
    genre_path = os.path.join(DATASET_PATH, genre)
    if not os.path.exists(genre_path):
        continue
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        signal, sr = librosa.load(file_path, duration=30)
        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_spec_db.shape[1] < 128:
            mel_spec_db = np.pad(mel_spec_db, ((0,0),(0,128-mel_spec_db.shape[1])), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :128]
        X.append(mel_spec_db)
        y.append(genre_idx)

X = np.array(X)[..., np.newaxis]
y = to_categorical(y, num_classes=len(genres))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(genres), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
model.save("../models/genre_cnn_model.h5")
with open("../models/genre_cnn_model.pkl", "wb") as f:
    pickle.dump(model, f)
