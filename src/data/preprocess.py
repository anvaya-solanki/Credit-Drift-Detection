import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/application_train.csv")
PROCESSED_DATA_PATH = Path("data/processed")

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TIME_COLUMN = "DAYS_EMPLOYED"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace special value 365243 with NaN
    """
    df[TIME_COLUMN] = df[TIME_COLUMN].replace(365243, np.nan)
    return df

def time_based_split(df: pd.DataFrame) -> dict:
    """
    Sort by DAYS_EMPLOYED to simulate time
    More negative -> older data
    """
    df_sorted = df.sort_values(by=TIME_COLUMN, ascending=True)

    n = len(df_sorted)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

def save_splits(splits: dict):
    for split_name, split_df in splits.items():
        split_df.to_csv(
            PROCESSED_DATA_PATH / f"{split_name}.csv",
            index=False
        )

def main():
    df = load_data(RAW_DATA_PATH)
    df = clean_days_employed(df)

    splits = time_based_split(df)
    save_splits(splits)

    for name, split in splits.items():
        print(f"{name.upper()} size: {len(split)}")


if __name__ == "__main__":
    main()