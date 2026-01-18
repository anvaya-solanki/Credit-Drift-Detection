from datetime import datetime


def classify_severity(drift_summary: dict) -> str:
    drift_ratio = drift_summary.get("drift_ratio", 0)
    details = drift_summary.get("details", {})

    p_values = [
        v["p_value"]
        for v in details.values()
        if "p_value" in v
    ]

    min_p = min(p_values) if p_values else 1.0

    if drift_ratio >= 0.6 or min_p < 0.001:
        return "HIGH"

    if drift_ratio >= 0.4 or min_p < 0.01:
        return "MEDIUM"

    if drift_ratio >= 0.2 or min_p < 0.05:
        return "LOW"

    return "NONE"


def generate_alert(drift_summary: dict) -> dict:
    severity = classify_severity(drift_summary)

    if severity == "NONE":
        return {
            "status": "no_alert",
            "alerts": []
        }

    return {
        "status": "alert",
        "alerts": [
            {
                "severity": severity,
                "drift_ratio": drift_summary["drift_ratio"],
                "drifted_features": drift_summary["drifted_features"],
                "timestamp": datetime.utcnow().isoformat(),
                "recommended_action": {
                    "LOW": "Monitor closely",
                    "MEDIUM": "Investigate data pipeline",
                    "HIGH": "Retrain model immediately"
                }[severity]
            }
        ]
    }
