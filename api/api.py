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
    """Predict serve time in seconds for a single patient"""
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

@app.post("/predict")
def predict(doctor_data: dict):
    """Single patient prediction in seconds"""
    try:
        pred_time = predict_single(doctor_data)
    except ValueError as e:
        return {"error": str(e)}
    return {"predicted_serve_time_seconds": float(pred_time)}

@app.post("/predict_queue")
def predict_queue(queue: List[Dict]):
    """
    Predict serve time and queue-aware wait time (in minutes) for a list of patients.
    
    Input example:
    [
        {"Doctor_ID": "D00001", "Doctor_Age": 45, "Doctor_Type": "Cardiologist", "Reason": "Regular Check-up"},
        {"Doctor_ID": "D00002", "Doctor_Age": 50, "Doctor_Type": "Dermatologist", "Reason": "Consultation"}
    ]
    
    Output example:
    [
        {"Doctor_ID": "D00001","Predicted_Serve_Time_Minutes": 8,"Queue_Wait_Time_Minutes":0},
        {"Doctor_ID": "D00002","Predicted_Serve_Time_Minutes": 10,"Queue_Wait_Time_Minutes":8}
    ]
    """
    pred_times = []
    # Predict individual serve times in seconds
    for patient in queue:
        try:
            t = predict_single(patient)
        except ValueError as e:
            return {"error": str(e)}
        pred_times.append(t)
    
    # Calculate cumulative queue-aware wait times in minutes
    results = []
    cumulative_time = 0
    for patient, serve_time in zip(queue, pred_times):
        serve_min = round(serve_time / 60)  # convert seconds to minutes
        wait_min = round(cumulative_time / 60)
        results.append({
            "Doctor_ID": patient["Doctor_ID"],
            "Predicted_Serve_Time_Minutes": serve_min,
            "Queue_Wait_Time_Minutes": wait_min
        })
        cumulative_time += serve_time
    
    return results
