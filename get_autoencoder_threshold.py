import torch
import joblib
import numpy as np

from models.autoencoder import Autoencoder

# Load vectorizer
vectorizer = joblib.load(
    "models/hdfs_vectorizer.pkl"
)

# Load training logs
with open(
    "data/hdfs_train.txt",
    "r",
    encoding="utf-8"
) as f:
    logs = f.readlines()

X = vectorizer.transform(logs)

X_dense = X.toarray()

X_tensor = torch.FloatTensor(X_dense)

# Load model
input_dim = X_dense.shape[1]

model = Autoencoder(input_dim)

model.load_state_dict(
    torch.load(
        "models/hdfs_autoencoder.pt",
        map_location="cpu"
    )
)

model.eval()

with torch.no_grad():
    reconstructed = model(X_tensor)

errors = torch.mean(
    (X_tensor - reconstructed) ** 2,
    dim=1
).numpy()

print("Mean Error:", errors.mean())
print("Std Error :", errors.std())

threshold = errors.mean() + 3 * errors.std()

print("\nRecommended Threshold:", threshold)