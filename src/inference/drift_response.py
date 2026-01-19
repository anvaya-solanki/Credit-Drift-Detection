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
    retrain_data_path = save_retraining_data(window_size=200)
    trigger_retraining_job(retrain_data_path)

    return {
        "action": "trigger_retraining",
        "reason": "Severe drift detected",
        "retraining_data": retrain_data_path
    }
