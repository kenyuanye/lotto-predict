import os
import joblib
import numpy as np
import pandas as pd
from utils.feature_engineering import build_features_for_prediction
from utils import NUMBER_COLUMNS, POWERBALL_COLUMN

MODEL_PATH = "models"

# Expected models
POSITION_MODELS = [
    os.path.join(MODEL_PATH, f"model_pos{i+1}.pkl") for i in range(6)
]
POWERBALL_MODEL = os.path.join(MODEL_PATH, "model_powerball.pkl")

def predict_with_ml(draw_history_df, exclusions=None):
    """
    Predict next draw using ML models with full feature input.
    Returns: [[n1, n2, n3, n4, n5, n6, Powerball]]
    """
    try:
        X = build_features_for_prediction(draw_history_df)

        prediction = []

        # Predict 6 main numbers
        for i, model_path in enumerate(POSITION_MODELS):
            model = joblib.load(model_path)
            raw_pred = model.predict(X)[0]
            clipped = int(np.clip(round(raw_pred), 1, 40))
            prediction.append(clipped)

        # Predict Powerball
        pb_model = joblib.load(POWERBALL_MODEL)
        pb_pred = pb_model.predict(X)[0]
        powerball = int(np.clip(round(pb_pred), 1, 10))
        prediction.append(powerball)

        return [prediction]

    except Exception as e:
        raise RuntimeError(f"❌ ML prediction failed: {e}")
