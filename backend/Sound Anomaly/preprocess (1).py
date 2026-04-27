import os
import librosa
import numpy as np

BASE_PATH = "data/audio"


def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)


X = []
y = []

machines = os.listdir(BASE_PATH)

for machine in machines:
    machine_path = os.path.join(BASE_PATH, machine)

    if not os.path.isdir(machine_path):
        continue

    print(f"Processing machine: {machine}")

    normal_path = os.path.join(machine_path, "normal")
    abnormal_path = os.path.join(machine_path, "abnormal")

    # ---------------- NORMAL (0) ----------------
    if os.path.exists(normal_path):
        for file in os.listdir(normal_path):
            if file.endswith(".wav"):
                file_path = os.path.join(normal_path, file)

                try:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(0)
                except Exception as e:
                    print(f"Error in {file}: {e}")

    # ---------------- ABNORMAL (1) ----------------
    if os.path.exists(abnormal_path):
        for root, dirs, files in os.walk(abnormal_path):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)

                    try:
                        features = extract_features(file_path)
                        X.append(features)
                        y.append(1)
                    except Exception as e:
                        print(f"Error in {file}: {e}")


X = np.array(X)
y = np.array(y)

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

np.save("data/audio/X.npy", X)
np.save("data/audio/y.npy", y)

print("Normal samples:", np.sum(y == 0))
print("Abnormal samples:", np.sum(y == 1))

if len(set(y)) < 2:
    raise ValueError("❌ Dataset must contain both normal and abnormal samples!")