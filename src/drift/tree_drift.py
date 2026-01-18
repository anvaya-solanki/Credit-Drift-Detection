import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import json
from pathlib import Path
import pickle

LOG_FILE = Path("logs/predictions.csv")
REF_STATS_PATH = Path("src/inference/reference_stats.pkl")

def load_reference_data(max_samples: int = 5000):
    with open(REF_STATS_PATH, "rb") as f:
        ref_stats = pickle.load(f)
    min_len = min(len(stats["values"]) for stats in ref_stats.values())
    rows = []
    for i in range(min(min_len, max_samples)):
        row = {}
        for feature, stats in ref_stats.items():
            row[feature] = stats["values"][i]
        rows.append(row)
    return pd.DataFrame(rows)



def load_recent_data(n: int = 200):
    if not LOG_FILE.exists():
        return None
    df = pd.read_csv(LOG_FILE).tail(n)
    rows = []
    for row in df["features"]:
        parsed = json.loads(row)
        rows.append(parsed)
    return pd.DataFrame(rows)


def tree_based_drift(window_size: int = 200):
    ref_df = load_reference_data()
    cur_df = load_recent_data(window_size)

    if cur_df is None or cur_df.empty:
        return {"status": "no_data"}
    common_cols = list(set(ref_df.columns) & set(cur_df.columns))
    ref_df = ref_df[common_cols].dropna()
    cur_df = cur_df[common_cols].dropna()
    min_size = min(len(ref_df), len(cur_df))
    ref_df = ref_df.sample(min_size, random_state=42)
    cur_df = cur_df.sample(min_size, random_state=42)
    X = pd.concat([ref_df, cur_df])
    y = np.array([0] * min_size + [1] * min_size)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    return {
        "status": "ok",
        "drift_classifier_auc": round(float(auc), 4),
        "drift_detected": bool(auc > 0.75),
        "top_drift_features": importances.head(5).to_dict()
    }
