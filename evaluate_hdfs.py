import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load labels
labels = pd.read_csv("data/anomaly_label.csv")

# Load logs
with open("data/hdfs_test_abnormal.txt", "r", encoding="utf-8") as f:
    logs = f.readlines()

# Load vectorizer and model
vectorizer = joblib.load("models/hdfs_vectorizer.pkl")
model = joblib.load("models/hdfs_isolation_forest.pkl")

# Convert logs
X = vectorizer.transform(logs)

# Predict
pred = model.predict(X)

# Convert prediction
pred = [1 if x == -1 else 0 for x in pred]

# Convert labels
y_true = labels["Label"].map({
    "Anomaly": 1,
    "Normal": 0
})

# Match lengths
n = min(len(y_true), len(pred))

y_true = y_true[:n]
pred = pred[:n]

print("\n=== Evaluation Results ===\n")

print("Accuracy :", accuracy_score(y_true, pred))
print("Precision:", precision_score(y_true, pred))
print("Recall   :", recall_score(y_true, pred))
print("F1 Score :", f1_score(y_true, pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, pred))

print("\nClassification Report:\n")
print(classification_report(y_true, pred))