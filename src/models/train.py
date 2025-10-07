import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.data.preprocess import load_raw, basic_clean, save_processed
from src.models.model import get_models, build_pipeline
from src.utils.logger import get_logger

logger = get_logger("train")

def main():
    logger.info("=== Starting Model Training Pipeline ===")

    raw = Path("data/raw/survey.csv")
    if not raw.exists():
        logger.error(f"File not found: {raw}")
        raise FileNotFoundError(raw)

    logger.info("Loading and cleaning data...")
    df = load_raw(str(raw))
    df_clean = basic_clean(df)
    save_processed(df_clean, "data/processed/processed_survey.csv")

    X = df_clean.drop(columns=["treatment"])
    y = (df_clean["treatment"] == "Yes").astype(int)

    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = get_models()
    best_model_name, best_score, best_pipeline = None, 0, None

    logger.info("Training models and evaluating performance...")
    for name, model in models.items():
        try:
            pipe = build_pipeline(model)
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = accuracy_score(y_test, preds)
            logger.info(f"{name} | accuracy = {acc:.3f}")
            if acc > best_score:
                best_score = acc
                best_model_name = name
                best_pipeline = pipe
        except Exception as e:
            logger.exception(f"Training failed for {name}: {e}")

    if not best_pipeline:
        logger.error("No successful model training.")
        return

    out = Path("models/trained_model.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, out)
    logger.info(f"Best model: {best_model_name} (accuracy={best_score:.3f})")
    logger.info(f"Model saved to {out}")
    logger.info("=== Training Complete ===")

if __name__ == "__main__":
    main()
