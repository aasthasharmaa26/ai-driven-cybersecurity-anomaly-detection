# AI-Driven Cybersecurity Log Analysis and Anomaly Detection

A Python-based cybersecurity project that explores machine-learning and deep-learning methods for identifying unusual patterns in system logs. The repository includes HDFS log preprocessing, TF-IDF feature extraction, anomaly-detection experiments, model evaluation scripts, and a FastAPI-based detection prototype.

> **Project status:** Internship prototype / research and learning project. Results should be reported only after running the evaluation scripts on correctly aligned test logs and labels. This project is not a production security product.

## Highlights

- Log text preprocessing and normalization
- TF-IDF feature extraction for log messages
- Isolation Forest anomaly detection
- One-Class SVM anomaly detection
- PyTorch autoencoder experiment using reconstruction error
- XGBoost supervised classification experiment
- Evaluation scripts for accuracy, precision, recall, F1-score, and confusion matrices
- FastAPI prototype for submitting logs and returning alert metadata
- SQLite-backed alert history and WebSocket alert updates in the prototype

## Architecture

```text
HDFS / system log lines
        |
        v
Preprocessing and text normalization
        |
        v
TF-IDF vectorization
        |
        +--> Isolation Forest
        +--> One-Class SVM
        +--> Autoencoder (reconstruction error)
        +--> XGBoost classifier
        |
        v
Predictions + evaluation metrics
        |
        v
Optional FastAPI alert prototype
```

## Repository layout

```text
.
├── backend/                 # FastAPI prototype
├── database/                # SQLAlchemy database models and setup
├── detection/               # Prediction helpers and threat rules
├── models/                  # Model definitions and saved model artifacts
├── preprocessing/           # Log cleaning and template extraction
├── training/                # Reusable training modules
├── data/                    # HDFS log data and labels (review licensing before redistribution)
├── train_hdfs.py
├── train_ocsvm.py
├── train_autoencoder_hdfs.py
├── train_xgboost.py
├── evaluate_hdfs.py
├── evaluate_ocsvm.py
├── evaluate_xgboost.py
└── requirements.txt
```

## Requirements

- Python 3.11 recommended
- Packages listed in `requirements.txt`
- For the PyTorch autoencoder, install a PyTorch build suitable for your system

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Training and evaluation

Run commands from the repository root so relative file paths resolve correctly.

```bash
python train_hdfs.py
python train_ocsvm.py
python train_autoencoder_hdfs.py
python train_xgboost.py
```

Evaluation scripts:

```bash
python evaluate_hdfs.py
python evaluate_ocsvm.py
python evaluate_xgboost.py
```

Some scripts expect the supplied HDFS data and saved vectorizer/model files to be present. Review each script's input paths before running.

## API prototype

The FastAPI app is in `backend/main.py`. Its imports use project-root-relative paths, so the launch command and Python path may need adjustment depending on how the project is run. Confirm the environment and endpoint behavior locally before presenting the API as a verified live demo.

## Important evaluation notes

- Ensure each test log is paired with its correct ground-truth label. The current scripts need a dataset-alignment review before their metrics can be treated as reliable.
- Do not compare model scores unless the same held-out test set and evaluation protocol are used.
- The XGBoost experiment derives labels from normal and abnormal log files; document this construction and avoid data leakage when creating a final benchmark.
- The API's current threat classification uses simple keyword rules. It is not a learned threat classifier and should be described as a rule-based prototype.
- The API currently sets `reconstruction_error` to a placeholder value. Do not present that field as a real autoencoder score until it is connected to actual inference.
- This repository is for research/educational use and does not replace a SIEM or a production incident-response workflow.

## Security and data handling

The GitHub-ready copy excludes local SQLite databases, generated logs, Python bytecode caches, and the `data/raw` path. Before publishing, verify that the remaining dataset files may be redistributed and that no confidential internship logs, credentials, IP addresses, or internal C-DOT information are included.


## Attribution

This project was developed as part of an individual cybersecurity learning/internship project. Update this section with the exact contribution, dataset citation, and any required institutional acknowledgements before public release.
