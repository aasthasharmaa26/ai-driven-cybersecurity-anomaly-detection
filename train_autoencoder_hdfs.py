import torch
import torch.nn as nn
import torch.optim as optim

import joblib

from models.autoencoder import Autoencoder

# Load HDFS vectorizer
vectorizer = joblib.load(
    "models/hdfs_vectorizer.pkl"
)

# Read HDFS training logs
with open(
    "data/hdfs_train.txt",
    "r",
    encoding="utf-8"
) as f:
    logs = f.readlines()

print("Training Logs:", len(logs))

# TF-IDF
X = vectorizer.transform(logs)

print("Shape:", X.shape)

# Dense matrix
X = X.toarray()

# Tensor
X_tensor = torch.FloatTensor(X)

# Input dimension
input_dim = X.shape[1]

# Model
model = Autoencoder(input_dim)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 50

print("\nTraining Started...\n")

for epoch in range(epochs):

    outputs = model(X_tensor)

    loss = criterion(
        outputs,
        X_tensor
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 5 == 0:

        print(
            f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.6f}"
        )

torch.save(
    model.state_dict(),
    "models/hdfs_autoencoder.pt"
)

print("\nTraining Complete ✅")