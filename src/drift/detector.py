import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict


class DriftDetector:
    """
    Pure stateless statistical drift detector using the Kolmogorov-Smirnov test.
    """

    def __init__(self, reference_df: pd.DataFrame):
        self.reference_df = reference_df.select_dtypes(include=[np.number])

    def detect(self, current_df: pd.DataFrame) -> Dict[str, Dict]:
        drift_report = {}

        current_df = current_df.select_dtypes(include=[np.number])

        for col in self.reference_df.columns:
            if col not in current_df.columns:
                continue

            ref_values = self.reference_df[col].dropna().values
            cur_values = current_df[col].dropna().values

            if len(cur_values) < 10:
                continue

            _, p_value = ks_2samp(ref_values, cur_values)

            drift_report[col] = {
                "p_value": float(p_value),
                "drift_detected": bool(p_value < 0.05)
            }

        return drift_report
