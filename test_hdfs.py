import joblib
from preprocessing.parser import clean_log

# Load model
model = joblib.load("models/hdfs_isolation_forest.pkl")

# Load vectorizer
vectorizer = joblib.load("models/hdfs_vectorizer.pkl")

# Read HDFS logs
with open("data/hdfs_test_abnormal.txt", "r", encoding="utf-8") as f:
 lines = f.readlines()

print("Logs Loaded:", len(lines))

# Clean logs
cleaned_logs = []

for line in lines:
    cleaned = clean_log(line)
    cleaned_logs.append(cleaned)

print("Logs Cleaned ✅")

# Convert to TF-IDF
X = vectorizer.transform(cleaned_logs)

print("Vector Shape:", X.shape)
print("Non-zero values:", X.nnz)

# Predict
predictions = model.predict(X)

anomalies = sum(predictions == -1)
normals = sum(predictions == 1)

print("Anomalies:", anomalies)
print("Normals:", normals)

# Anomaly scores
scores = model.decision_function(X)

print("\n=== Score Statistics ===")
print("Min Score:", scores.min())
print("Max Score:", scores.max())
print("Average Score:", scores.mean())

print("\n=== First 20 Predictions ===")

for i in range(20):

    if predictions[i] == -1:
        print(
            f"Row {i}: ANOMALY DETECTED 🚨 | Score: {scores[i]:.4f}"
        )
    else:
        print(
            f"Row {i}: Normal | Score: {scores[i]:.4f}"
        )

# Top 10 most suspicious logs
print("\n=== Top 10 Most Suspicious Rows ===")

sorted_idx = scores.argsort()

for idx in sorted_idx[:10]:

    print(
        f"Row {idx} | Score: {scores[idx]:.4f} | Prediction: {predictions[idx]}"
    )
