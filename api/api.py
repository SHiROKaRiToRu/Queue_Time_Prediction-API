from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import numpy as np

app = FastAPI()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # points to api/
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "supervised_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "..", "models", "encoders.pkl")

# Load model and preprocessing objects
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)
feature_cols = ["Doctor_ID","Doctor_Age","Doctor_Type","Reason"]

# Pydantic model for request
class PredictRequest(BaseModel):
    Doctor_ID: str
    Doctor_Age: float
    Doctor_Type: str
    Reason: str

@app.post("/predict")
def predict(doctor_data: PredictRequest):
    """
    Example input:
    {
        "Doctor_ID": "D00001",
        "Doctor_Age": 30-65,
        "Doctor_Type": "Cardiologist",
        "Reason": "Follow-up Appointment"
    }
    """
    # Convert Pydantic object to dict
    doctor_data = doctor_data.dict()

    # Encode features
    features = []
    for col in feature_cols:
        if col in ["Doctor_ID","Doctor_Type","Reason"]:
            le = encoders[col]
            if doctor_data[col] not in le.classes_:
                return {"error": f"Unknown category '{doctor_data[col]}' for column '{col}'"}
            features.append(le.transform([doctor_data[col]])[0])
        else:
            features.append(doctor_data[col])

    # Scale features
    features = scaler.transform([features])

    # Predict serve time
    pred_time = model.predict(features)[0]

    return {"predicted_serve_time_seconds": float(pred_time)}
