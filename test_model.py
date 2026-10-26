import pandas as pd
import joblib

print("Starting...")

# load model
model = joblib.load("models/isolation_forest.pkl")
print("Model Loaded ✅")

# load dataset
df = pd.read_csv(
    r"C:\Users\HP\Desktop\cic_ids2018\Processed Traffic Data for ML Algorithms\Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"
)

print("Dataset Loaded ✅")

# numeric columns only
X = df.select_dtypes(include=['number'])

print("Numeric Columns Selected ✅")

# remove NaN
X = X.fillna(0)

print("NaN Removed ✅")

# only 26 features
X = X.iloc[:, :26]

print("Selected 26 Features ✅")

# take small sample
X_test = X.head(10)

print("Testing Started 🚀")

# prediction
predictions = model.predict(X_test)

print("Predictions Done ✅")

# output
for i, pred in enumerate(predictions):
    print(f"Raw Prediction: {pred}")

    if pred == -1:
        print(f"Row {i}: ANOMALY DETECTED 🚨")
    else:
        print(f"Row {i}: Normal")

print("Finished ✅")