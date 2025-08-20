import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "Doctor_ID": "D00001",
    "Doctor_Age": 30,
    "Doctor_Type": "Cardiologist",
    "Reason": "Follow-up Appointment"
}

response = requests.post(url, json=data)
print(response.json())
