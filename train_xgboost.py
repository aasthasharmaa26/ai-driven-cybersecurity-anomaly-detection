import joblib
import numpy as np

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

print("Normal Logs:", len(normal_logs))
print("Abnormal Logs:", len(abnormal_logs))

# Features
X_normal = vectorizer.transform(normal_logs)
X_abnormal = vectorizer.transform(abnormal_logs)

# Combine
from scipy.sparse import vstack

X = vstack([X_normal, X_abnormal])

# Labels
y_normal = np.zeros(len(normal_logs))
y_abnormal = np.ones(len(abnormal_logs))

y = np.concatenate([
    y_normal,
    y_abnormal
])

print("Shape:", X.shape)

# Model
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X, y)

joblib.dump(
    model,
    "models/hdfs_xgboost.pkl"
)

print("XGBoost Training Complete ✅")