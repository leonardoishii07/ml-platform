import logging
import os

def get_logger(name: str = "ml-platform", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
