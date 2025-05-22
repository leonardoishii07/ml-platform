import pandas as pd
import plotly.express as px
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

def plot_target_mean_bar(df: pd.DataFrame, target: str, output_dir: str, max_categories: int = 10):
    os.makedirs(output_dir, exist_ok=True)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        logger.info(f"Plotting barplot for {col}")
        try:
            group = df.groupby(col)[target].mean().sort_values(ascending=False).head(max_categories)
            fig = px.bar(group, title=f"{target} mean by {col}", labels={col: col, "value": f"{target} mean"})
            fig.write_html(os.path.join(output_dir, f"{col}_barplot.html"))
        except Exception as e:
            logger.warning(f"Failed to plot {col}: {e}")
