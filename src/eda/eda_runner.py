from src.eda import summary, distributions, correlations, missing, outliers, categorical, target_mean_by_category, category_plots
import pandas as pd
import os


def run_eda(df: pd.DataFrame, config: dict) -> None:
    """
    Run exploratory data analysis (EDA) on the given DataFrame.
    """
    eda_config = config.get("eda", {})
    target = config.get("experiment", {}).get("target")
    output_dir = eda_config.get("output_dir", "outputs/eda")
    os.makedirs(output_dir, exist_ok=True)

    if not eda_config['enabled']:
        print("EDA is disabled in the configuration.")
        return
    print("Running EDA...")
    steps = eda_config.get("steps", {})

    if steps.get("summary", False):
        summary.summarize_numerical_features(df, output_dir)

    if steps.get("categorical", False):
        categorical.analyze_categorical_features(df, output_dir)

    if steps.get("distributions", False):
        distributions.plot_distributions(df, output_dir)

    if steps.get("correlations", False):
        correlations.plot_correlation_matrix(df, output_dir)
        correlations.scatter_matrix(df, output_dir)

    if steps.get("missing", False):
        missing.plot_missing_values(df, output_dir)

    if steps.get("outliers", False):
        outliers.detect_outliers(df, output_dir)

    if steps.get("target_mean_by_category", False):
        if target is None:
            print("Target variable is not specified in the configuration.")
        else:
            target_mean_by_category.target_mean_by_category(df, target, output_dir)
    if steps.get("plot_target_mean_bar", False):
        if target is None:
            print("Target variable is not specified in the configuration.")
        else:
            category_plots.plot_target_mean_bar(df, target, output_dir)

    print("EDA completed.")

