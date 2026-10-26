import sys
import os

sys.path.append(os.path.abspath("."))

import time
import joblib
import torch
import torch.nn as nn

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from preprocessing.parser import clean_log
from models.autoencoder import Autoencoder

# Load vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")

# Load Isolation Forest
iso_model = joblib.load("models/isolation_forest.pkl")

# Load Autoencoder
input_dim = len(vectorizer.get_feature_names_out())

autoencoder = Autoencoder(input_dim)

autoencoder.load_state_dict(
    torch.load("models/autoencoder.pt")
)

autoencoder.eval()

# Loss function
criterion = nn.MSELoss()

# Track processed lines
last_position = 0

class LogHandler(FileSystemEventHandler):

    def on_modified(self, event):

        global last_position

        if event.src_path.endswith("live.log"):

            with open("logs/live.log", "r") as f:

                # Move to last read position
                f.seek(last_position)

                # Read new lines
                new_lines = f.readlines()

                # Update position
                last_position = f.tell()

            for line in new_lines:

                process_log(line)

def process_log(log):

    cleaned = clean_log(log)

    vector = vectorizer.transform([cleaned])

    # Isolation Forest
    iso_pred = iso_model.predict(vector)

    # Autoencoder
    tensor = torch.FloatTensor(vector.toarray())

    reconstructed = autoencoder(tensor)

    loss = criterion(reconstructed, tensor)

    error = loss.item()

    print("\n========================")
    print("NEW LOG:", cleaned)

    # Threat logic
    anomaly_detected = False

    if iso_pred[0] == -1:

        anomaly_detected = True

        print("Isolation Forest: 🚨")

    if error > 0.02:

        anomaly_detected = True

        print("Autoencoder: 🚨")

    print("Reconstruction Error:", error)

    if anomaly_detected:

        print("FINAL STATUS: 🚨 ANOMALY DETECTED")

    else:

        print("FINAL STATUS: ✅ NORMAL")

    print("========================")

# Watchdog observer
observer = Observer()

event_handler = LogHandler()

observer.schedule(event_handler, path="logs", recursive=False)

observer.start()

print("\n🚀 Real-time monitoring started...\n")

try:

    while True:

        time.sleep(1)

except KeyboardInterrupt:

    observer.stop()

observer.join()