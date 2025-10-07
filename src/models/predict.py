import joblib
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("predict")

def predict_from_model(input_data: pd.DataFrame):
    logger.info("Loading trained model for prediction...")
    model_path = "models/trained_model.pkl"

    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        raise

    try:
        preds = model.predict(input_data)
        logger.info("Prediction successful.")
        return preds
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise
