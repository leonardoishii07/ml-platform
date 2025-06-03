import yaml
from src.ingest.loader_factory import get_loader
from src.eda.eda_runner import run_eda
from src.preprocessing.preprocessing_runner import PreprocessingRunner
from src.feature_selection import FeatureSelectionRunner
from src.utils.logger import get_logger

logger = get_logger()

logger.info("Starting ML Experiment")

# Load config
logger.info("Loading configuration")
with open("configs/dataset3.yaml") as f:
    config = yaml.safe_load(f)
logger.info("Configuration loaded successfully.")

# Load data
loader = get_loader(config)
df = loader.load()

# Run EDA
run_eda(df, config)

# Run preprocessing
df = PreprocessingRunner(config["preprocessing"]).run(df)

# Run feature selection
if "feature_selection" in config:
    df = FeatureSelectionRunner(config["feature_selection"]).run(df, config["experiment"]["target"])

# Save processed data
df.to_csv("data/processed/preprocessed_data.csv", index=False)
logger.info("Data processing completed and saved.")