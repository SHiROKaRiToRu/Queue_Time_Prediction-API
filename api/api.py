from fastapi import FastAPI
import joblib
import os
import numpy as np
from typing import List, Dict

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # points to api/
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "supervised_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "..", "models", "encoders.pkl")

# Load model and preprocessing objects
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)
feature_cols = ["Doctor_ID","Doctor_Age","Doctor_Type","Reason"]

def predict_single(patient: dict):
    # Encode features
    features = []
    for col in feature_cols:
        if col in ["Doctor_ID","Doctor_Type","Reason"]:
            le = encoders[col]
            if patient[col] not in le.classes_:
                raise ValueError(f"Unknown category '{patient[col]}' for column '{col}'")
            features.append(le.transform([patient[col]])[0])
        else:
            features.append(patient[col])

    # Scale features
    features = scaler.transform([features])

    # Predict serve time
    pred_time = model.predict(features)[0]
    return float(pred_time)

@app.post("/predict_queue")
def predict_queue(queue: List[Dict]):
    """
    Input: list of patient dicts
    [
        {"Doctor_ID": "D00001", "Doctor_Age": 45, "Doctor_Type": "Cardiologist", "Reason": "Regular Check-up"},
        {"Doctor_ID": "D00002", "Doctor_Age": 50, "Doctor_Type": "Dermatologist", "Reason": "Consultation"}
    ]
    
    Output: list of patients with predicted serve time and queue wait time
    """
    pred_times = []
    # Predict individual serve times
    for patient in queue:
        try:
            t = predict_single(patient)
        except ValueError as e:
            return {"error": str(e)}
        pred_times.append(t)
    
    # Calculate cumulative queue-aware wait times
    results = []
    cumulative_time = 0
    for patient, serve_time in zip(queue, pred_times):
        results.append({
            "Doctor_ID": patient["Doctor_ID"],
            "Predicted_Serve_Time": serve_time,
            "Queue_Wait_Time": cumulative_time
        })
        cumulative_time += serve_time
    
    return results
