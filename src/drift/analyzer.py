import pandas as pd
import json
import numpy as np
from scipy.stats import ks_2samp
from pathlib import Path
import pickle

LOG_FILE = Path("logs/predictions.csv")
REF_STATS_PATH = Path("src/inference/reference_stats.pkl")


def load_reference_stats():
    with open(REF_STATS_PATH, "rb") as f:
        return pickle.load(f)


def load_recent_predictions(n: int = 100):
    if not LOG_FILE.exists():
        return None

    df = pd.read_csv(LOG_FILE)
    return df.tail(n)


def aggregate_drift(n: int = 100):
    reference_stats = load_reference_stats()
    recent_df = load_recent_predictions(n)

    if recent_df is None or recent_df.empty:
        return {"status": "no_data"}

    feature_values = {}

    for row in recent_df["features"]:
        parsed = json.loads(row)
        for k, v in parsed.items():
            if isinstance(v, (int, float)):
                feature_values.setdefault(k, []).append(v)

    drift_summary = {}

    for feature, values in feature_values.items():
        if feature not in reference_stats:
            continue

        ref_values = reference_stats[feature]["values"]
        cur_values = np.array(values)

        if len(cur_values) < 5:
            continue

        _, p_value = ks_2samp(ref_values, cur_values)

        drift_summary[feature] = {
            "p_value": float(p_value),
            "drift_detected": p_value < 0.05
        }

    drifted_features = sum(
        1 for v in drift_summary.values() if v["drift_detected"]
    )

    return {
        "samples_analyzed": len(recent_df),
        "total_features_checked": len(drift_summary),
        "drifted_features": drifted_features,
        "drift_ratio": round(drifted_features / max(len(drift_summary), 1), 3),
        "details": drift_summary
    }
