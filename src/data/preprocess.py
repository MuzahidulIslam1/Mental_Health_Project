import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("preprocess")

def load_raw(path: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.exception(f"Failed to load data from {path}: {e}")
        raise

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting basic cleaning...")
    df = df.copy()

    df = df.dropna(subset=["Age", "Gender", "treatment"])
    df["Age"] = df["Age"].clip(lower=18, upper=100)

    # Normalize gender values
    df["Gender"] = df["Gender"].str.strip().str.lower()
    df["Gender"] = df["Gender"].replace({
        "male": "male",
        "m": "male",
        "female": "female",
        "f": "female"
    })

    logger.info(f"Cleaned data shape: {df.shape}")
    return df

def save_processed(df: pd.DataFrame, path: str):
    logger.info(f"Saving processed data to {path}")
    try:
        df.to_csv(path, index=False)
        logger.info("Processed data saved successfully.")
    except Exception as e:
        logger.exception(f"Error saving processed data: {e}")
        raise
