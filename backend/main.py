from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "Cardiovascular_disease_model.joblib")
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class PredictionRequest(BaseModel):
    age: float
    gender: int
    height: float
    weight: float
    ap_hi: float
    ap_lo: float
    cholesterol: int
    gluc: int
    smoke: int = 0
    alco: int = 0
    active: int

@app.get("/")
def read_root():
    return {"message": "CardioShield AI Backend is running"}

@app.post("/predict")
def predict(data: PredictionRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Calculate Derived Features
        # Note: height is in cm, weight in kg. BMI formula: weight / (height/100)^2
        bmi = data.weight / ((data.height / 100) ** 2)
        pp = data.ap_hi - data.ap_lo

        # Feature array: [age, gender, height, weight, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, pp]
        features = np.array([[
            data.age, data.gender, data.height, data.weight, bmi, 
            data.ap_hi, data.ap_lo, data.cholesterol, data.gluc, 
            data.smoke, data.alco, data.active, pp
        ]])

        prob = model.predict_proba(features)[0][1]
        result = 1 if prob > 0.5 else 0

        return {
            "result": int(result),
            "prob": round(float(prob) * 100, 2),
            "bmi": round(bmi, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
