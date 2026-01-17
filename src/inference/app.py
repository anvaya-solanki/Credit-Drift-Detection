import pickle
import numpy as np
from scipy.stats import ks_2samp
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Union
from pathlib import Path  
from src.monitoring.store import log_prediction
from src.drift.analyzer import aggregate_drift

MODEL_URI = "runs:/483eb277392341e4b4ded994ab8ff948/model"

app = FastAPI(title="Home Credit Inference API")


BASE_DIR = Path(__file__).resolve().parent  
REF_STATS_PATH = BASE_DIR / "reference_stats.pkl"
try:
    with open(REF_STATS_PATH, "rb") as f:
        reference_stats = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load reference_stats.pkl: {e}")

class CreditApplication(BaseModel):
    data: Dict[str, Union[float, int, str, None]]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    drift: Dict[str, Dict[str, Union[float, bool]]]

def detect_drift(input_data: dict):
    drift_report = {}

    for feature, value in input_data.items():
        ref = reference_stats.get(feature)

        if ref is None or value is None or not isinstance(value, (int, float)):
            continue
        ks_stat, p_value = ks_2samp(
            ref["values"],
            np.array([value, value])
        )

        drift_report[feature] = {
            "p_value": float(p_value),
            "drift_detected": p_value < 0.05
        }
    return drift_report

@app.on_event("startup")
def load_model():
    global model
    model = mlflow.sklearn.load_model(MODEL_URI)

@app.get("/drift/report")
def drift_report(samples: int = 100):
    return aggregate_drift(samples)


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication):
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.feature_names_in_

    input_df = pd.DataFrame([{col: None for col in feature_names}])
    for key, value in application.data.items():
        if key in input_df.columns:
            input_df.at[0, key] = value
    prob = model.predict_proba(input_df)[0, 1]
    prediction = int(prob >= 0.5)

    drift = detect_drift(application.data)
    log_prediction(
        input_data=application.data,
        prediction=prediction,
        probability=float(prob)
    )
    return {
        "prediction": prediction,
        "probability": float(prob),
        "drift": drift
    }
