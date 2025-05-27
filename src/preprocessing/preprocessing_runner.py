from src.preprocessing.missing_handler import MissingValueHandler
from src.preprocessing.scaler_handler import ScalerHandler
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PreprocessingRunner:
    def __init__(self, config: dict):
        self.config = config

    def run(self, df):
        logger.info("Starting preprocessing")

        # Handle missing values
        missing_cfg = self.config.get("missing_values", {})
        if missing_cfg.get("enabled", True):
            logger.info("Running MissingValueHandler")
            mv_handler = MissingValueHandler(**missing_cfg)
            mv_handler.fit(df)
            df = mv_handler.transform(df)

        # Scaling
        scaler_cfg = self.config.get("scaler", {})
        if scaler_cfg.get("enabled", False):
            logger.info("Running ScalerHandler")
            scaler = ScalerHandler(**scaler_cfg)
            scaler.fit(df)
            df = scaler.transform(df)

        # Encoding (futuro)
        # ...

        logger.info("Finished preprocessing")
        return df
