import pandas as pd
import numpy as np

def detect_outliers(df: pd.DataFrame, output_dir: str):
    numeric_cols = df.select_dtypes(include=["number"]).columns
    outlier_info = []

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
        outlier_info.append({
            "feature": col,
            "outlier_count": int(outliers.sum()),
            "outlier_pct": float(100 * outliers.mean())
        })

    pd.DataFrame(outlier_info).to_csv(f"{output_dir}/outliers.csv", index=False)
