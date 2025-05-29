from src.preprocessing.missing_handler import MissingValueHandler
from src.preprocessing.scaler_handler import ScalerHandler
from src.preprocessing.encoder_handler import EncoderHandler
from src.preprocessing.auto_feature_generator import AutoFeatureGenerator
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
        encoder_cfg = self.config.get("encoder", {})
        if encoder_cfg.get("enabled", False):
            logger.info("Running EncoderHandler")
            encoder = EncoderHandler(**encoder_cfg)
            encoder.fit(df)
            df = encoder.transform(df)

        # Feature generation
        feature_gen_cfg = self.config.get("auto_feature_generation", {})
        if feature_gen_cfg.get("enabled", False):
            logger.info("Running AutoFeatureGenerator")
            feature_gen = AutoFeatureGenerator(**feature_gen_cfg)
            df = feature_gen.generate_features(df)

        logger.info("Finished preprocessing")
        return df
