import json
import streamlit as st
import pandas as pd
import requests
from pathlib import Path
import mlflow
import os
import pickle

API_BASE = "http://127.0.0.1:8000"
LOG_PATH = Path(r"D:\College\Data Drift Detection\logs\predictions.csv")

st.set_page_config(page_title="MLOps Drift Dashboard", layout="wide")

st.title("Model Drift & Retraining Dashboard")

st.header("Drift Monitoring")
window_size = st.slider("Drift window size", 0, 500, 100, step=50)
if st.button("Run Drift Check"):
    try:
        resp = requests.get(f"{API_BASE}/drift/alerts", params={"window_size": window_size})
        data = resp.json()
        st.subheader("Drift Summary")
        st.json(data["drift_summary"])
        st.subheader("Alert Status")
        st.json(data["alert_report"])
        st.subheader("System Action")
        st.json(data["response_action"])
    except Exception as e:
        st.error(f"Drift check failed: {e}")

st.header("Feature-level Drift")
if LOG_PATH.exists():
    df = pd.read_csv(LOG_PATH)
    if df.empty:
        st.warning("Prediction log exists but is empty.")
    else:
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        try:
            df["parsed_features"] = df["features"].apply(json.loads)
        except Exception as e:
            st.error(f"Failed to parse feature JSON: {e}")
            st.stop()
        features_df = pd.json_normalize(df["parsed_features"])
        st.subheader("Recent Feature Values (last 20 samples)")
        st.dataframe(features_df.tail(20))
        if "probability" in df.columns:
            st.subheader("Prediction Confidence Over Time")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            st.line_chart(
                df.set_index("timestamp")["probability"]
            )
        if "prediction" in df.columns:
            st.subheader("Prediction Distribution")
            st.bar_chart(
                df["prediction"].value_counts()
            )
        st.subheader("Feature Snapshot")
        sample_features = df["features"].tail(50).apply(lambda x: pd.Series(eval(x)))
        st.dataframe(sample_features.describe())
else:
    st.warning("No prediction logs found.")

st.header("Model Registry")
try:
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='credit_default_model'")
    model_rows = []
    for v in versions:
        model_rows.append({
            "Version": v.version,
            "Run ID": v.run_id,
            "Source": v.source
        })
    st.dataframe(pd.DataFrame(model_rows))
except Exception as e:
    st.warning(f"Could not load model registry: {e}")

st.header("Retraining Status")
RETRAIN_PATH = Path(r"D:\College\Data Drift Detection\data\retraining\retraining.csv")
MIN_RETRAIN_ROWS = 500
if RETRAIN_PATH.exists():
    retrain_df = pd.read_csv(RETRAIN_PATH)
    current_rows = len(retrain_df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Labeled Samples Available", current_rows)
    col2.metric("Minimum Required", MIN_RETRAIN_ROWS)
    if current_rows >= MIN_RETRAIN_ROWS:
        col3.success("READY")
        st.success("Model is eligible for retraining if drift is detected.")
    else:
        col3.warning("WAITING")
        st.warning("Collect more labeled data before retraining.")

st.header("Model Performance Comparison")
try:
    experiment = mlflow.get_experiment_by_name("home_credit_baseline")
    if experiment is None:
        st.warning("MLflow experiment 'home_credit_baseline' not found.")
    else:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_roc_auc DESC"]
        )
        if runs.empty:
            st.warning("No MLflow runs found for this experiment.")
        else:
            comparison_df = runs[
                [
                    "run_id",
                    "metrics.val_roc_auc",
                    "metrics.val_pr_auc",
                    "metrics.val_f1",
                    "metrics.test_roc_auc",
                    "tags.retraining"
                ]
            ].copy()
            comparison_df.rename(
                columns={
                    "run_id": "Run ID",
                    "metrics.val_roc_auc": "Val ROC-AUC",
                    "metrics.val_pr_auc": "Val PR-AUC",
                    "metrics.val_f1": "Val F1",
                    "metrics.test_roc_auc": "Test ROC-AUC",
                    "tags.retraining": "Retrained?"
                },
                inplace=True
            )
            st.dataframe(comparison_df, use_container_width=True)
            best_model = comparison_df.iloc[0]
            st.success(
                f"Best Model: Run {best_model['Run ID']} "
                f"(Val ROC-AUC = {best_model['Val ROC-AUC']:.4f})"
            )
except Exception as e:
    st.error(f"Model comparison failed: {e}")

st.markdown("---")
st.caption("Monitoring Dashboard")