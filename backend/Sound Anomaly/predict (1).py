import os
import librosa
import numpy as np
import joblib

model = joblib.load("models/audio_model.pkl")

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)


BASE = "data/audio"

for machine in os.listdir(BASE):
    machine_path = os.path.join(BASE, machine)

    if not os.path.isdir(machine_path):
        continue

    print(f"\n🔧 MACHINE: {machine.upper()}")

    for root, dirs, files in os.walk(machine_path):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)

                features = extract_features(path).reshape(1, -1)
                pred = model.predict(features)[0]
                prob = model.predict_proba(features)[0]
                confidence = max(prob) * 100

                if pred == 0:
                    print(f"{file} → ✅ NORMAL ({confidence:.2f}%)")
                else:
                    print(f"{file} → ⚠️ ANOMALY ({confidence:.2f}%)")