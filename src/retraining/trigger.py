from pathlib import Path
import pandas as pd
import datetime
import mlflow
import subprocess

LOG_FILE = Path("logs/predictions.csv")
RETRAIN_DATA_DIR = Path("data/retraining")
RETRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
ACCUMULATED_DATA = Path("data/retraining/retraining.csv")

def save_retraining_data(window_size: int = 200, clear_logs: bool = False):
    """
    Saves retraining data from accumulated predictions.
    
    Args:
        window_size: Number of samples needed for retraining
        clear_logs: Whether to clear prediction logs after saving (only do this after successful retraining)
    """
    if not ACCUMULATED_DATA.exists():
        if not LOG_FILE.exists():
            raise ValueError("No prediction data available")
        df = pd.read_csv(LOG_FILE)
        if len(df) < window_size:
            raise ValueError(f"Not enough data for retraining: {len(df)} < {window_size}")
        retrain_df = df.tail(window_size)
    else:
        df = pd.read_csv(ACCUMULATED_DATA)
        if LOG_FILE.exists():
            new_preds = pd.read_csv(LOG_FILE)
            if len(new_preds) > 0:
                if "TARGET" not in new_preds.columns and "probability" in new_preds.columns:
                    new_preds["TARGET"] = (new_preds["probability"] > 0.5).astype(int)
                drop_cols = ["prediction", "probability", "timestamp"]
                new_preds = new_preds.drop(columns=[c for c in drop_cols if c in new_preds.columns])
                df = pd.concat([df, new_preds], ignore_index=True)
                df.to_csv(ACCUMULATED_DATA, index=False)
        
        if len(df) < window_size:
            raise ValueError(f"Not enough accumulated data for retraining: {len(df)} < {window_size}")
        retrain_df = df.tail(window_size)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = RETRAIN_DATA_DIR / f"retrain_data_{timestamp}.csv"
    retrain_df.to_csv(output_path, index=False)
    
    print(f"Saved {len(retrain_df)} samples to {output_path}")
    if clear_logs and LOG_FILE.exists():
        LOG_FILE.unlink()
        print(f"Cleared {LOG_FILE}")
    
    return str(output_path)

def trigger_retraining_job(data_path: str):
    """
    Simulates retraining.
    """
    mlflow.set_experiment("home_credit_retraining")
    with mlflow.start_run(run_name="auto_retraining"):
        mlflow.log_param("retraining_data", data_path)
        df = pd.read_csv(data_path)
        mlflow.log_param("num_samples", len(df))
        result = subprocess.run(
            ["python", "src/models/train.py", "--data", data_path],
            check=False,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            mlflow.set_tag("training_status", "success")
            print("Retraining completed successfully")
            if LOG_FILE.exists():
                LOG_FILE.unlink()
                print(f"Cleared prediction logs after successful retraining")
        else:
            mlflow.set_tag("training_status", "failed")
            print(f"Retraining failed: {result.stderr}")
        mlflow.set_tag("trigger", "drift_detected")