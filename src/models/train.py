import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score
)

from src.data.preprocess_features import build_preprocessor

DATA_PATH = Path("data/processed")
ARTIFACT_PATH = Path("artifacts/models")
ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)

TARGET = "TARGET"

def load_split(split_name: str):
    df = pd.read_csv(DATA_PATH / f"{split_name}.csv")
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

def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    preprocessor, num_features, cat_features = build_preprocessor(
        pd.concat([X_train, y_train], axis=1)
    )
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )
    mlflow.set_experiment("home_credit_baseline")

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        pipeline.fit(X_train, y_train)
        train_metrics = evaluate(pipeline, X_train, y_train)
        val_metrics = evaluate(pipeline, X_val, y_val)
        test_metrics = evaluate(pipeline, X_test, y_test)
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        mlflow.log_param("num_features", len(num_features))
        mlflow.log_param("cat_features", len(cat_features))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="credit_default_model"
        )

    print("Training complete.")
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
