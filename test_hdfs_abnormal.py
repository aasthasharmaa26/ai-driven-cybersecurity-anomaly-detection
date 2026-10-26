import joblib

# Load HDFS model
model = joblib.load("models/hdfs_isolation_forest.pkl")

# Load HDFS vectorizer
vectorizer = joblib.load("models/hdfs_vectorizer.pkl")

# Read abnormal logs
with open("data/hdfs_test_abnormal.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

print("Test Logs:", len(lines))

# Convert to TF-IDF
X = vectorizer.transform(lines)

print("Feature Shape:", X.shape)

# Predict
predictions = model.predict(X)

anomalies = sum(predictions == -1)
normals = sum(predictions == 1)

print("\n=== Results ===")
print("Anomalies:", anomalies)
print("Normals:", normals)

# Show first 20 predictions
print("\n=== First 20 Rows ===")

for i in range(min(20, len(predictions))):
    if predictions[i] == -1:
        print(f"Row {i}: ANOMALY DETECTED 🚨")
    else:
        print(f"Row {i}: Normal")