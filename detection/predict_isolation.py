import joblib

model = joblib.load("models/isolation_forest.pkl")

def predict_anomaly(vector):

    prediction = model.predict(vector)

    if prediction[0] == -1:
        return True

    return False