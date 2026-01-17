import pandas as pd
from pathlib import Path
from datetime import datetime
import json

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "predictions.csv"

def log_prediction(
    input_data: dict,
    prediction: int,
    probability: float
):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "prediction": prediction,
        "probability": probability,
        "features": json.dumps(input_data)
    }

    df = pd.DataFrame([record])

    if LOG_FILE.exists():
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)
