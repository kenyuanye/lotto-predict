import os
import joblib
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def predict_with_ml(models_dir: str, features_df: pd.DataFrame) -> tuple[list[int], int]:
    """
    Predicts 6 main lotto numbers and 1 Powerball using pre-trained models.

    Args:
        models_dir (str): Path to directory containing model .pkl files.
        features_df (pd.DataFrame): Input features for prediction (single row).

    Returns:
        tuple: (sorted list of 6 main numbers, Powerball number)
    """
    logger.info("🚀 Starting ML prediction")
    if not isinstance(features_df, pd.DataFrame):
        logger.error("Expected a pandas DataFrame for features_df, got: %s", type(features_df))
        raise ValueError("features_df must be a pandas DataFrame")

    if features_df.empty:
        logger.error("Provided features DataFrame is empty.")
        raise ValueError("features_df is empty.")

    predictions = []
    for i in range(1, 7):
        model_path = os.path.join(models_dir, f"rf_model_pos{i}.pkl")
        if not os.path.exists(model_path):
            logger.error(f"❌ Missing model file: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        logger.debug(f"🔍 Loading model for position {i} from {model_path}")
        model = joblib.load(model_path)
        pred = model.predict(features_df)
        predictions.append(int(pred[0]))

    # Powerball prediction
    model_path_pb = os.path.join(models_dir, "rf_model_powerball.pkl")
    if not os.path.exists(model_path_pb):
        logger.error(f"❌ Missing Powerball model file: {model_path_pb}")
        raise FileNotFoundError(f"Powerball model file not found: {model_path_pb}")
    logger.debug(f"🔍 Loading Powerball model from {model_path_pb}")
    model_pb = joblib.load(model_path_pb)
    powerball_pred = int(model_pb.predict(features_df)[0])

    logger.info("✅ ML prediction complete")
    return sorted(predictions), powerball_pred
