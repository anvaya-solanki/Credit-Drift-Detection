from typing import Dict, Any, List
from datetime import datetime

DRIFT_RATIO_THRESHOLD = 0.3
MIN_DRIFTED_FEATURES = 3

def evaluate_alerts(drift_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert drift summary into alerts
    """
    if "status" in drift_summary:
        return {
            "status": "no_data",
            "alerts": []
        }
    alerts: List[Dict[str, Any]] = []
    drift_ratio = drift_summary["drift_ratio"]
    drifted_features = drift_summary["drifted_features"]
    if drift_ratio >= DRIFT_RATIO_THRESHOLD:
        alerts.append({
            "type": "HIGH_DRIFT_RATIO",
            "message": f"Drift ratio {drift_ratio} exceeds threshold",
            "severity": "high"
        })
    if drifted_features >= MIN_DRIFTED_FEATURES:
        alerts.append({
            "type": "MULTIPLE_FEATURE_DRIFT",
            "message": f"{drifted_features} features drifting",
            "severity": "medium"
        })
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "alerts_triggered": len(alerts),
        "alerts": alerts
    }
