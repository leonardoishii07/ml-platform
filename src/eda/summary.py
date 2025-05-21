import pandas as pd

def generate_summary(df: pd.DataFrame, output_dir: str):
    summary = df.describe(include="all").transpose()
    summary.to_csv(f"{output_dir}/summary.csv")
