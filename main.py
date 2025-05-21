import yaml
from src.ingest.loader_factory import get_loader
from src.eda.eda_runner import run_eda
from src.utils.logger import get_logger

logger = get_logger()

logger.info("Starting ML Experiment")

# Load config
logger.info("Loading configuration")
with open("configs/dataset1.yaml") as f:
    config = yaml.safe_load(f)
logger.info("Configuration loaded successfully.")

# Load data
loader = get_loader(config)
df = loader.load()
run_eda(df, config)
