import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_distributions(df: pd.DataFrame, output_dir: str):
    for col in df.select_dtypes(include=["number"]).columns:
        plt.figure(figsize=(6, 4))
        df[col].hist(bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_distribution.png"))
        plt.close()
