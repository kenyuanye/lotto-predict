import os
import joblib
import numpy as np
import pandas as pd
from core.feature_engineering import build_features_for_prediction
from core import NUMBER_COLUMNS, POWERBALL_COLUMN

SYMBOLIC_MODEL_PATH = "models/models_symbolic"

POSITION_MODELS = [
    os.path.join(SYMBOLIC_MODEL_PATH, f"symbolic_model_pos{i+1}.pkl") for i in range(6)
]

POWERBALL_MODEL = os.path.join(SYMBOLIC_MODEL_PATH, "symbolic_model_powerball.pkl")

def predict_with_symbolic(draw_history_df, exclusions=None):
    """
    Predict 6 main numbers and Powerball using symbolic regression models.
    Returns: [[n1, n2, n3, n4, n5, n6, Powerball]]
    """
    try:
        X = build_features_for_prediction(draw_history_df)

        prediction = []

        # Predict main numbers
        for model_file in POSITION_MODELS:
            model = joblib.load(model_file)
            raw_pred = model.predict(X)[0]
            clipped = int(np.clip(round(raw_pred), 1, 40))
            prediction.append(clipped)

        # Powerball prediction
        pb_model = joblib.load(POWERBALL_MODEL)
        pb_pred = pb_model.predict(X)[0]
        powerball = int(np.clip(round(pb_pred), 1, 10))
        prediction.append(powerball)

        return [prediction]

    except Exception as e:
        raise RuntimeError(f"❌ Symbolic prediction failed: {e}")
