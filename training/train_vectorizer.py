import sys
import os

sys.path.append(os.path.abspath("."))

from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing.parser import clean_log
import joblib

# Read logs
with open("logs/live.log", "r") as f:

    lines = f.readlines()

# Clean logs
cleaned_logs = []

for line in lines:

    cleaned = clean_log(line)

    cleaned_logs.append(cleaned)

# TF-IDF
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(cleaned_logs)

# Save vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("TF-IDF training complete")

print("Shape:", X.shape)