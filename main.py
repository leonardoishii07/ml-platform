import yaml
from src.ingest.loader_factory import get_loader
from src.eda.eda_runner import run_eda
from src.preprocessing.preprocessing_runner import PreprocessingRunner
from src.utils.logger import get_logger

logger = get_logger()

logger.info("Starting ML Experiment")

# Load config
logger.info("Loading configuration")
# with open("configs/dataset1.yaml") as f:
#     config = yaml.safe_load(f)
with open("configs/dataset3.yaml") as f:
    config = yaml.safe_load(f)
logger.info("Configuration loaded successfully.")

# Load data
loader = get_loader(config)
df = loader.load()
run_eda(df, config)
df = PreprocessingRunner(config["preprocessing"]).run(df)
df.to_csv("data/processed/preprocessed_data.csv", index=False)