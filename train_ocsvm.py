import joblib

from sklearn.svm import OneClassSVM

# Load TF-IDF features from existing HDFS training

vectorizer = joblib.load("models/hdfs_vectorizer.pkl")

with open(
    "data/hdfs_train.txt",
    "r",
    encoding="utf-8"
) as f:
    logs = f.readlines()

X_train = vectorizer.transform(logs)

print("Training Shape:", X_train.shape)

model = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.1
)

model.fit(X_train)

joblib.dump(
    model,
    "models/hdfs_ocsvm.pkl"
)

print("One-Class SVM Training Complete ✅")