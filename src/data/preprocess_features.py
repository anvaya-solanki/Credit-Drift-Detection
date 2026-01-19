import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(df: pd.DataFrame):
    """
    Builds a preprocessing pipeline for numerical and categorical features.
    """
    DROP_COLS = ["TARGET", "SK_ID_CURR"]
    feature_df = df.drop(
        columns=[c for c in DROP_COLS if c in df.columns]
    )
    numeric_features = feature_df.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = feature_df.select_dtypes(include=["object"]).columns
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor, numeric_features, categorical_features
