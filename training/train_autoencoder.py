import sys
import os

sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import joblib

from preprocessing.parser import clean_log
from models.autoencoder import Autoencoder

# Load vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")

# Read logs
with open("logs/live.log", "r") as f:

    lines = f.readlines()

# Clean logs
cleaned_logs = []

for line in lines:

    cleaned = clean_log(line)

    cleaned_logs.append(cleaned)

# TF-IDF vectors
X = vectorizer.transform(cleaned_logs)

# Convert to dense array
X = X.toarray()

# Convert to tensor
X_tensor = torch.FloatTensor(X)

# Input size
input_dim = X.shape[1]

# Create model
model = Autoencoder(input_dim)

# Loss
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 50

print("\nTraining Autoencoder...\n")

for epoch in range(epochs):

    # Forward
    outputs = model(X_tensor)

    loss = criterion(outputs, X_tensor)

    # Backprop
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 5 == 0:

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}")

# Save model
torch.save(model.state_dict(), "models/autoencoder.pt")

print("\nAutoencoder training complete 😭🔥")