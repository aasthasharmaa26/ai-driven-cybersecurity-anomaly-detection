import torch
import joblib
import numpy as np

from models.autoencoder import Autoencoder

# Load vectorizer
vectorizer = joblib.load(
    "models/hdfs_vectorizer.pkl"
)

# Load test logs
with open(
    "data/hdfs_test_abnormal.txt",
    "r",
    encoding="utf-8"
) as f:
    logs = f.readlines()

print("Test Logs:", len(logs))

# TF-IDF Features
X = vectorizer.transform(logs)

print("Feature Shape:", X.shape)

X_dense = X.toarray()

X_tensor = torch.FloatTensor(X_dense)

# Load trained Autoencoder
input_dim = X_dense.shape[1]

model = Autoencoder(input_dim)

model.load_state_dict(
    torch.load(
        "models/hdfs_autoencoder.pt",
        map_location="cpu"
    )
)

model.eval()

# Reconstruction
with torch.no_grad():

    reconstructed = model(X_tensor)

# Reconstruction Error
errors = torch.mean(
    (X_tensor - reconstructed) ** 2,
    dim=1
).numpy()

print("\n=== Error Statistics ===")
print("Min Error :", errors.min())
print("Max Error :", errors.max())
print("Mean Error:", errors.mean())

# Threshold from training data
threshold = 0.00020836103

print("\nThreshold:", threshold)

# Prediction
predictions = np.where(
    errors > threshold,
    -1,
    1
)

anomalies = np.sum(
    predictions == -1
)

normals = np.sum(
    predictions == 1
)

print("\n=== Results ===")
print("Anomalies:", anomalies)
print("Normals:", normals)

print("\n=== First 20 Rows ===")

for i in range(min(20, len(predictions))):

    if predictions[i] == -1:

        print(
            f"Row {i}: ANOMALY DETECTED 🚨 | Error: {errors[i]:.8f}"
        )

    else:

        print(
            f"Row {i}: Normal | Error: {errors[i]:.8f}"
        )

# Top 10 Most Suspicious Logs
print("\n=== Top 10 Most Suspicious Rows ===")

sorted_idx = np.argsort(errors)[::-1]

for idx in sorted_idx[:10]:

    status = (
        "ANOMALY"
        if predictions[idx] == -1
        else "Normal"
    )

    print(
        f"Row {idx} | Error: {errors[idx]:.8f} | {status}"
    )