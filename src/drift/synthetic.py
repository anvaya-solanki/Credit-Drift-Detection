import pandas as pd
import json
import numpy as np
from pathlib import Path


LOG_FILE = Path("logs/predictions.csv")


def inject_drift(
    feature: str,
    drift_type: str = "shift",
    magnitude: float = 1.5,
    window_size: int = 20
):
    """
    Inject synthetic drift into the last N predictions.

    drift_type:
        - "shift" → mean shift
        - "scale" → variance increase
        - "noise" → random noise
    """

    if not LOG_FILE.exists():
        raise FileNotFoundError("predictions.csv not found")

    df = pd.read_csv(LOG_FILE)

    if len(df) < window_size:
        raise ValueError("Not enough samples to inject drift")

    recent_idx = df.tail(window_size).index

    for idx in recent_idx:
        features = json.loads(df.at[idx, "features"])

        if feature not in features:
            continue

        value = features[feature]

        if not isinstance(value, (int, float)):
            continue

        if drift_type == "shift":
            features[feature] = value * magnitude

        elif drift_type == "scale":
            features[feature] = value + np.random.normal(
                0, abs(value) * (magnitude - 1)
            )

        elif drift_type == "noise":
            features[feature] = value + np.random.normal(0, magnitude)

        df.at[idx, "features"] = json.dumps(features)

    df.to_csv(LOG_FILE, index=False)

    return {
        "status": "drift_injected",
        "feature": feature,
        "drift_type": drift_type,
        "magnitude": magnitude,
        "samples_modified": window_size
    }
