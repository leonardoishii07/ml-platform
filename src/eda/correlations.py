import pandas as pd
import plotly.express as px
import os
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.logger import get_logger

logger = get_logger(__name__)

def plot_correlation_matrix(df: pd.DataFrame, output_dir: str):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    logger.info(f"Saved correlation matrix to {output_dir}/correlation_matrix.png")

def scatter_matrix(df: pd.DataFrame, output_dir: str, max_vars: int = 10):
    os.makedirs(output_dir, exist_ok=True)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(num_cols) < 2:
        logger.warning("Not enough numerical columns for scatter matrix.")
        return

    cols_to_plot = num_cols[:max_vars]
    try:
        fig = px.scatter_matrix(df[cols_to_plot])
        fig.update_traces(diagonal_visible=False)
        fig.write_html(os.path.join(output_dir, f"scatter_matrix.html"))
        logger.info("Saved scatter matrix.")
    except Exception as e:
        logger.warning(f"Failed to generate scatter matrix: {e}")