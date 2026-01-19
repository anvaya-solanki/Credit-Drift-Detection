from pathlib import Path
import pandas as pd
import datetime
import mlflow
import subprocess

LOG_FILE = Path("logs/predictions.csv")
RETRAIN_DATA_DIR = Path("data/retraining")
RETRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)

def save_retraining_data(window_size: int = 200):
    df = pd.read_csv(LOG_FILE)
    if len(df) < window_size:
        raise ValueError("Not enough data for retraining")

    retrain_df = df.tail(window_size)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = RETRAIN_DATA_DIR / f"retrain_data_{timestamp}.csv"
    retrain_df.to_csv(output_path, index=False)
    return str(output_path)


def trigger_retraining_job(data_path: str):
    """
    This simulates retraining.
    """
    mlflow.set_experiment("home_credit_retraining")

    with mlflow.start_run(run_name="auto_retraining"):
        mlflow.log_param("retraining_data", data_path)
        subprocess.run(
            ["python", "src/models/train.py", "--data", data_path],
            check=False
        )

        mlflow.set_tag("trigger", "drift_detected")
