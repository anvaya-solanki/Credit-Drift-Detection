from datetime import datetime
from src.retraining.trigger import save_retraining_data, trigger_retraining_job

def drift_action_handler(drift_ratio: float, alert_status: str):
    if alert_status != "alert":
        return {
            "action": "no_action",
            "reason": "No active drift alert"
        }
    if drift_ratio < 0.02:
        return {
            "action": "log_and_monitor",
            "reason": "Minor drift detected"
        }
    if drift_ratio < 0.1:
        return {
            "action": "notify_and_prepare_retraining",
            "reason": "Moderate drift detected"
        }
    try:
        retrain_data_path = save_retraining_data(window_size=200, clear_logs=False)
        print(f"Triggering retraining job with data from: {retrain_data_path}")
        trigger_retraining_job(retrain_data_path)
        
        return {
            "action": "retraining_triggered",
            "reason": "Severe drift detected",
            "retrain_data_path": retrain_data_path
        }
        
    except ValueError as e:
        print(f"Retraining skipped: {e}")
        return {
            "action": "retraining_skipped",
            "reason": str(e)
        }
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return {
            "action": "retraining_failed",
            "reason": f"Unexpected error: {str(e)}"
        }