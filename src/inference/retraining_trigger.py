from typing import Dict

def should_trigger_retraining(
    drift_summary: Dict,
    min_drifted_features: int = 3,
    min_samples: int = 500,
    avg_pvalue_threshold: float = 0.05
) -> bool:
    """
    Decide whether retraining should be triggered.
    """
    drifted_features = [
        f for f, stats in drift_summary.items()
        if stats["drift_detected"]
    ]
    if len(drifted_features) < min_drifted_features:
        return False
    avg_pvalue = sum(
        drift_summary[f]["p_value"] for f in drifted_features
    ) / len(drifted_features)
    if avg_pvalue > avg_pvalue_threshold:
        return False
    total_samples = drift_summary.get("_meta", {}).get("samples", 0)
    if total_samples < min_samples:
        return False
    return True
