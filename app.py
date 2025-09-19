from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from src.preprocessing import preprocess

# Load pipeline + label encoder
pipeline = joblib.load("models/final_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

app = FastAPI(title="ML API with Scaling & Label Encoding")

# Input schema (adjust to your feature names)
class InputData(BaseModel):
    volume_24h: float
    mkt_cap: float
    Liquidity_Ratio: float
    h1: float
    h24: float
    d7: float

@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame
    df = pd.DataFrame([{
        "24h_volume": data.volume_24h,
        "mkt_cap": data.mkt_cap,
        "Liquidity_Ratio": data.Liquidity_Ratio,
        "1h": data.h1,
        "24h": data.h24,
        "7d": data.d7
    }])

    # Predict
    pred_encoded = pipeline.predict(df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return {"prediction": pred_label}
