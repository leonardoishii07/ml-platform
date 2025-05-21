import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_correlation_matrix(df: pd.DataFrame, output_dir: str):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
