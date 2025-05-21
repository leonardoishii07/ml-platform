import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_missing_values(df: pd.DataFrame, output_dir: str):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if not missing.empty:
        plt.figure(figsize=(8, 6))
        missing.plot(kind="bar")
        plt.title("Missing Values per Column")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/missing_values.png")
        plt.close()
