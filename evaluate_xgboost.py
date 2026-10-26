import joblib
import numpy as np

from scipy.sparse import vstack

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

# Load vectorizer
vectorizer = joblib.load(
    "models/hdfs_vectorizer.pkl"
)

# Normal logs
with open(
    "data/hdfs_train.txt",
    "r",
    encoding="utf-8"
) as f:
    normal_logs = f.readlines()

# Abnormal logs
with open(
    "data/hdfs_test_abnormal.txt",
    "r",
    encoding="utf-8"
) as f:
    abnormal_logs = f.readlines()

# Features
X_normal = vectorizer.transform(normal_logs)
X_abnormal = vectorizer.transform(abnormal_logs)

X = vstack([X_normal, X_abnormal])

# Labels
y_normal = np.zeros(len(normal_logs))
y_abnormal = np.ones(len(abnormal_logs))

y = np.concatenate([
    y_normal,
    y_abnormal
])

print("Dataset Shape:", X.shape)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# Model
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("\n===== Evaluation Results =====")

print(
    "Accuracy :",
    round(accuracy_score(y_test, y_pred) * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision_score(y_test, y_pred) * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(recall_score(y_test, y_pred) * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(f1_score(y_test, y_pred) * 100, 2),
    "%"
)

print("\n===== Confusion Matrix =====")

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("\n===== Classification Report =====")

print(classification_report(y_test, y_pred))