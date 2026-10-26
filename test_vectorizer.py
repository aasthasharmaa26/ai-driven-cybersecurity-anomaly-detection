from preprocessing.parser import clean_log
import joblib

# Load vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")

# Sample log
log = "Failed password for root from 192.168.1.5"

# Clean
cleaned = clean_log(log)

# Transform
vector = vectorizer.transform([cleaned])

print("Cleaned Log:")
print(cleaned)

print("\nVector Shape:")
print(vector.shape)

print("\nVector:")
print(vector.toarray())