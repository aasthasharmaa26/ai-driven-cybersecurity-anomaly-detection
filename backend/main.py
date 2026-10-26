from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np

from detection.predict_isolation import predict_anomaly
from detection.threat_engine import classify_threat

from database.db import SessionLocal
from database.db import engine
from database.db import Base

from database.models import Alert

# =========================================
# CREATE DATABASE TABLES
# =========================================

Base.metadata.create_all(bind=engine)

# =========================================
# FASTAPI
# =========================================

app = FastAPI()

# =========================================
# CORS
# =========================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)

# =========================================
# LOAD TFIDF
# =========================================

vectorizer = joblib.load("models/vectorizer.pkl")

# =========================================
# MEMORY ALERTS
# =========================================

alerts = []

# =========================================
# WEBSOCKET CLIENTS
# =========================================

active_connections = []

# =========================================
# DETECT API
# =========================================

@app.post("/detect")
async def detect(log: str):

    # CLEAN LOG
    cleaned = log.lower()

    # TFIDF VECTOR
    vector = vectorizer.transform([cleaned]).toarray()

    # ISOLATION FOREST
    anomaly = predict_anomaly(vector)

    # TEMP ERROR VALUE
    error = 0.0

    # THREAT ENGINE
    threat_info = classify_threat(cleaned)

    # FINAL RESULT
    result = {

        "log": log,

        "cleaned_log": cleaned,

        "reconstruction_error": error,

        "anomaly_detected": bool(anomaly),

        "threat_type": threat_info["threat_type"],

        "severity": threat_info["severity"],

        "risk_score": threat_info["risk_score"],

        "mitre_attack": threat_info["mitre_attack"]

    }

    # SAVE IN MEMORY
    alerts.append(result)

    # =========================================
    # SAVE TO DATABASE
    # =========================================

    db = SessionLocal()

    new_alert = Alert(

        log=log,

        cleaned_log=cleaned,

        reconstruction_error=error,

        anomaly_detected=bool(anomaly),

        threat_type=threat_info["threat_type"],

        severity=threat_info["severity"],

        risk_score=threat_info["risk_score"],

        mitre_attack=threat_info["mitre_attack"]

    )

    db.add(new_alert)

    db.commit()

    db.close()

    # =========================================
    # SEND REALTIME ALERTS
    # =========================================

    disconnected = []

    for connection in active_connections:

        try:

            await connection.send_json(result)

        except:

            disconnected.append(connection)

    for connection in disconnected:

        active_connections.remove(connection)

    return result

# =========================================
# GET ALL ALERTS
# =========================================

@app.get("/alerts")
def get_alerts():

    return alerts

# =========================================
# DATABASE HISTORY
# =========================================

@app.get("/history")
def get_history():

    db = SessionLocal()

    all_alerts = db.query(Alert).all()

    results = []

    for alert in all_alerts:

        results.append({

            "id": alert.id,

            "log": alert.log,

            "severity": alert.severity,

            "threat_type": alert.threat_type,

            "risk_score": alert.risk_score,

            "mitre_attack": alert.mitre_attack

        })

    db.close()

    return results

# =========================================
# SEARCH ALERTS
# =========================================

@app.get("/search")
def search_alerts(severity: str):

    db = SessionLocal()

    filtered = db.query(Alert).filter(

        Alert.severity == severity

    ).all()

    results = []

    for alert in filtered:

        results.append({

            "log": alert.log,

            "severity": alert.severity,

            "threat_type": alert.threat_type

        })

    db.close()

    return results

# =========================================
# WEBSOCKET
# =========================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    active_connections.append(websocket)

    print("Frontend connected 🔥")

    try:

        while True:

            await websocket.receive_text()

    except WebSocketDisconnect:

        active_connections.remove(websocket)

        print("Frontend disconnected")