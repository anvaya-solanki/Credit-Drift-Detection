import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score

DATA_PATH = Path("data/processed")
TARGET = "TARGET"


def load_validation_data():
    df = pd.read_csv(DATA_PATH / "val.csv")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def evaluate(model, X, y):
    probs = model.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(y, probs)

    precision, recall, _ = precision_recall_curve(y, probs)
    pr_auc = auc(recall, precision)

    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(y, preds)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1
    }


def select_best_model(prod_model_uri: str, new_model_uri: str):
    X_val, y_val = load_validation_data()

    prod_model = mlflow.sklearn.load_model(prod_model_uri)
    new_model = mlflow.sklearn.load_model(new_model_uri)

    prod_metrics = evaluate(prod_model, X_val, y_val)
    new_metrics = evaluate(new_model, X_val, y_val)

    wins = sum(
        new_metrics[m] > prod_metrics[m]
        for m in prod_metrics
    )

    decision = "promote" if wins >= 2 else "reject"

    return {
        "decision": decision,
        "production_metrics": prod_metrics,
        "new_model_metrics": new_metrics,
        "wins": wins
    }
