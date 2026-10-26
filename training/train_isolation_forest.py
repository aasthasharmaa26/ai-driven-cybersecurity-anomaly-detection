import sys
import os

sys.path.append(os.path.abspath("."))

import joblib

from sklearn.ensemble import IsolationForest
from preprocessing.parser import clean_log

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

# Convert to TF-IDF vectors
X = vectorizer.transform(cleaned_logs)

# Train Isolation Forest
model = IsolationForest(

    n_estimators=100,
    contamination=0.1,
    random_state=42

)

model.fit(X)

# Save model
joblib.dump(model, "models/isolation_forest.pkl")

print("Isolation Forest trained successfully")

print("Total logs:", len(cleaned_logs))