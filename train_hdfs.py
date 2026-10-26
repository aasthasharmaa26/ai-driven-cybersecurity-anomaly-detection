import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

# Read training logs
with open("data/hdfs_train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

print("Training Logs:", len(lines))

# TF-IDF
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(lines)

print("Shape:", X.shape)

# Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

model.fit(X)

# Save
joblib.dump(vectorizer, "models/hdfs_vectorizer.pkl")
joblib.dump(model, "models/hdfs_isolation_forest.pkl")

print("Training Complete ✅")