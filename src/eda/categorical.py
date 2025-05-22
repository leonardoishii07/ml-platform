import pandas as pd
import numpy as np
import plotly.express as px
import os
from scipy.stats import entropy
from src.utils.logger import get_logger

logger = get_logger(__name__)

def analyze_categorical_features(df: pd.DataFrame, output_dir: str, top_n: int = 10):
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not cat_cols:
        logger.info("No categorical columns found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for col in cat_cols:
        logger.info(f"Analyzing categorical column: {col}")
        value_counts = df[col].value_counts(dropna=False)
        total = value_counts.sum()

        top_categories = value_counts.head(top_n)
        proportions = (top_categories / total).round(4)

        col_entropy = entropy(value_counts, base=2)  # Shannon entropy
        cardinality = df[col].nunique(dropna=False)
        relative_cardinality = cardinality / df.shape[0]

        result = {
            "feature": col,
            "cardinality": cardinality,
            "relative_cardinality": round(relative_cardinality, 4),
            "entropy": round(col_entropy, 4),
            "top_categories": proportions.to_dict()
        }

        results.append(result)

        # Generate and save countplot
        try:
            fig = px.bar(
                x=top_categories.index.astype(str),
                y=top_categories.values,
                labels={"x": col, "y": "Count"},
                title=f"Top {top_n} categories - {col}"
            )
            fig.write_html(os.path.join(output_dir, f"{col}_countplot.html"))
        except Exception as e:
            logger.warning(f"Failed to generate plot for {col}: {e}")

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "categorical_summary.csv"), index=False)
    logger.info(f"Saved categorical summary to {output_dir}/categorical_summary.csv")
