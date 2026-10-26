import joblib

# Load model
model = joblib.load(
    "models/hdfs_ocsvm.pkl"
)

# Load vectorizer
vectorizer = joblib.load(
    "models/hdfs_vectorizer.pkl"
)

# Load test logs
with open(
    "data/hdfs_test_abnormal.txt",
    "r",
    encoding="utf-8"
) as f:
    logs = f.readlines()

X_test = vectorizer.transform(logs)

print("Test Shape:", X_test.shape)

predictions = model.predict(X_test)

anomalies = sum(predictions == -1)
normals = sum(predictions == 1)

print("\n=== Results ===")
print("Anomalies:", anomalies)
print("Normals:", normals)

for i in range(20):
    if predictions[i] == -1:
        print(f"Row {i}: ANOMALY 🚨")
    else:
        print(f"Row {i}: Normal")