import pickle
import pandas as pd
from pathlib import Path

df = pd.read_csv(r"D:\College\Data Drift Detection\data\processed\train.csv")  

numeric_df = df.select_dtypes(include=["int64", "float64"])

reference_stats = {}

for col in numeric_df.columns:
    values = numeric_df[col].dropna().values

    if len(values) == 0:
        continue
    reference_stats[col] = {
        "values": numeric_df[col].dropna().values
    }

BASE_DIR = Path(__file__).resolve().parent
output_path = BASE_DIR / "reference_stats.pkl"

with open(output_path, "wb") as f:
    pickle.dump(reference_stats, f)

print("reference_stats.pkl created successfully")
print(f"Total features stored: {len(reference_stats)}")