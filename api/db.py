import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load env vars locally
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
appointments_collection = db[COLLECTION_NAME]

def fetch_appointments_for_doctor(doctor_id: str):
    appointments = list(appointments_collection.find(
        {"doctorId": doctor_id, "status": {"$in": ["upcoming", "confirmed"]}},
        {
            "_id": 1,
            "doctorId": 1,
            "patientName": 1,
            "age": 1,
            "reason": 1,
            "Doctor_Age": 1,
            "specialty": 1,
            "date": 1,
            "timeSlotId": 1
        }
    ).sort("date", 1))
    
    return appointments
