import pandas as pd
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

def target_mean_by_category(df: pd.DataFrame, target: str, output_dir: str, min_samples: int = 10):
    os.makedirs(output_dir, exist_ok=True)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    result = []

    for col in cat_cols:
        logger.info(f"Processing categorical feature: {col}")
        group = df.groupby(col)[target].agg(["count", "mean"]).reset_index()
        group = group[group["count"] >= min_samples]
        group.columns = [col, "count", f"{target}_mean"]
        group["feature"] = col
        result.append(group)

    if result:
        final_df = pd.concat(result)
        final_df.to_csv(os.path.join(output_dir, "target_mean_by_category.csv"), index=False)
        logger.info(f"Saved target mean by category to {output_dir}/target_mean_by_category.csv")
    else:
        logger.warning("No categorical features met the minimum sample size.")
