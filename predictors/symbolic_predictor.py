import joblib
import os
import numpy as np
import pandas as pd
from utils.feature_engineering import build_features_for_prediction

SYMBOLIC_MODEL_PATH = "models/models_symbolic"

POSITION_MODELS = [
    os.path.join(SYMBOLIC_MODEL_PATH, f"symbolic_model_pos{i+1}.pkl") for i in range(6)
]
POWERBALL_MODEL = os.path.join(SYMBOLIC_MODEL_PATH, "symbolic_model_powerball.pkl")


def predict_with_symbolic(draw_history_df, exclusions=None):
    """
    Predict 6 main numbers and Powerball using symbolic regression models.
    Returns a list: [n1, n2, n3, n4, n5, n6, powerball]
    """
    X = build_features_for_prediction(draw_history_df)
    prediction = []

    for model_file in POSITION_MODELS:
        model = joblib.load(model_file)
        pred = model.predict(X)[0]
        prediction.append(int(round(pred)))

    powerball_model = joblib.load(POWERBALL_MODEL)
    powerball = int(round(powerball_model.predict(X)[0]))
    prediction.append(powerball)

    return prediction