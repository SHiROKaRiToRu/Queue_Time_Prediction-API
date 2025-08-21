# api/api.py
from fastapi import FastAPI
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import joblib
import numpy as np

# Load environment variables locally
load_dotenv()

app = FastAPI(title="HealthQ Queue Prediction API")

# ==============================
# 1. MongoDB Connection
# ==============================
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
appointments_collection = db[COLLECTION_NAME]

# ==============================
# 2. Load ML Model & Preprocessing
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # points to api/
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "supervised_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "..", "models", "encoders.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)
feature_cols = ["Doctor_ID","Doctors_Age","Doctor_Type","Reason"]

# ==============================
# 3. MongoDB Extractor
# ==============================
def fetch_appointments_for_doctor(doctor_id: str):
    """
    Fetch upcoming or confirmed appointments for a specific doctor,
    sorted by date.
    """
    appointments = list(appointments_collection.find(
        {"doctorId": doctor_id, "status": {"$in": ["upcoming", "confirmed"]}},
        {
            "_id": 1,  # booking ID
            "doctorId": 1,
            "patientName": 1,
            "age": 1,
            "reason": 1,
            "Doctors_Age": 1,
            "specialty": 1,
            "date": 1,
            "timeSlotId": 1
        }
    ).sort("date", 1))
    return appointments

# ==============================
# 4. Prediction Endpoint
# ==============================
@app.post("/predict_queue_time")
def predict_queue(doctor_id: str):
    """
    Returns the upcoming appointments for a doctor with predicted serve time.
    Queue time is cumulative sum of previous patients' serve times.
    """
    appointments = fetch_appointments_for_doctor(doctor_id)
    
    results = []
    cumulative_time = 0
    
    for appt in appointments:
        # Prepare features for ML model
        features = []
        for col in feature_cols:
            if col in ["Doctor_ID","Doctor_Type","Reason"]:
                le = encoders[col]
                if col == "Doctor_ID":
                    value = appt["doctorId"]
                elif col == "Doctor_Type":
                    value = appt["specialty"]
                elif col == "Reason":
                    value = appt["reason"]
                if value not in le.classes_:
                    # fallback to first class if unknown
                    value = le.classes_[0]
                features.append(le.transform([value])[0])
            else:
                # numeric feature
                if col == "Doctors_Age":
                    features.append(appt.get("Doctors_Age", 40))  # default 40
                else:
                    features.append(appt.get(col, 0))
        
        # Scale features
        features_scaled = scaler.transform([features])
        pred_time_sec = model.predict(features_scaled)[0]
        
        # Queue-aware wait time
        cumulative_time += pred_time_sec
        
        results.append({
            "booking_id": str(appt["_id"]),
            "patient_name": appt["patientName"],
            "predicted_serve_time_seconds": float(pred_time_sec),
            "queue_wait_time_seconds": float(cumulative_time)
        })
    
    return {"doctor_id": doctor_id, "appointments": results}
