from datetime import datetime

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

    return {
        "action": "trigger_retraining",
        "reason": "Severe drift detected"
    }
