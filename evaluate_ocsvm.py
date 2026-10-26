import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load model
model = joblib.load("models/hdfs_ocsvm.pkl")

# Load vectorizer
vectorizer = joblib.load("models/hdfs_vectorizer.pkl")

# Load test logs
with open(
    "data/hdfs_test_abnormal.txt",
    "r",
    encoding="utf-8"
) as f:
    logs = f.readlines()

# Convert logs to TF-IDF features
X = vectorizer.transform(logs)

# Predict
pred = model.predict(X)

# Load labels
labels = pd.read_csv("data/anomaly_label.csv")

# Convert labels
y_true = labels["Label"].map({
    "Normal": 1,
    "Anomaly": -1
})

# Same length
n = min(len(y_true), len(pred))

y_true = y_true[:n]
pred = pred[:n]

print("\n===== Evaluation Results =====")

print(
    "Accuracy :",
    round(accuracy_score(y_true, pred) * 100, 2),
    "%"
)

print(
    "Precision:",
    round(
        precision_score(
            y_true,
            pred,
            pos_label=-1,
            zero_division=0
        ) * 100,
        2
    ),
    "%"
)

print(
    "Recall   :",
    round(
        recall_score(
            y_true,
            pred,
            pos_label=-1,
            zero_division=0
        ) * 100,
        2
    ),
    "%"
)

print(
    "F1 Score :",
    round(
        f1_score(
            y_true,
            pred,
            pos_label=-1,
            zero_division=0
        ) * 100,
        2
    ),
    "%"
)

print("\n===== Confusion Matrix =====")

cm = confusion_matrix(y_true, pred)

print(cm)

print("\nTN =", cm[0][0])
print("FP =", cm[0][1])
print("FN =", cm[1][0])
print("TP =", cm[1][1])