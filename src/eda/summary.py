import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

def summarize_numerical_features(df: pd.DataFrame, output_dir: str, low_variance_thresh: float = 0.01):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        logger.info("No numerical columns found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    summary_data = []

    for col in num_cols:
        logger.info(f"Summarizing numerical column: {col}")
        series = df[col].dropna()
        if series.empty:
            logger.warning(f"Column {col} has only NaNs, skipping.")
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        mode_val = series.mode().iloc[0] if not series.mode().empty else np.nan

        summary_data.append({
            "feature": col,
            "count": len(series),
            "missing": df[col].isna().sum(),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "q1": q1,
            "median": series.median(),
            "q3": q3,
            "max": series.max(),
            "iqr": iqr,
            "skewness": skew(series),
            "kurtosis": kurtosis(series),
            "mode": mode_val,
            "n_unique": df[col].nunique(),
            "pct_unique": df[col].nunique() / len(df),
            "pct_zeros": (series == 0).mean(),
            "is_constant": df[col].nunique() == 1,
            "is_low_variance": df[col].nunique() / len(df) < low_variance_thresh
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "numerical_summary.csv"), index=False)
    logger.info(f"Saved numerical summary to {output_dir}/numerical_summary.csv")
