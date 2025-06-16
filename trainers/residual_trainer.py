# trainers/residual_trainer.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

from utils.feature_engineering import build_features_for_prediction
from utils import NUMBER_COLUMNS, POWERBALL_COLUMN
from utils.model_utils import load_models

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RESIDUAL_MODEL_FILE = os.path.join(MODEL_DIR, "residual_model.pkl")
RESIDUAL_MODEL_PB_FILE = os.path.join(MODEL_DIR, "residual_model_powerball.pkl")


def train_residual_models(draw_df: pd.DataFrame, logger=None):
    log = logger.info if logger else print
    err = logger.error if logger else lambda msg, **kwargs: print(msg)

    try:
        df = draw_df.copy().reset_index(drop=True)
        df["DrawIndex"] = df.index
        df["Draw Number"] = df.index

        X = build_features_for_prediction(df)
        models = load_models()

        # Check model presence
        missing = [str(i) for i in range(1, 7) if models.get(str(i)) is None]
        if missing:
            raise ValueError(f"Missing base models for positions: {missing}")
        if "full" not in models:
            raise ValueError("Missing 'full' base model.")
        if POWERBALL_COLUMN not in models:
            raise ValueError("Missing PowerBall base model.")

        # Predict main numbers
        base_preds = np.round(models["full"].predict(X)).astype(int)
        base_preds = np.clip(base_preds, 1, 40)
        actual_main = df[NUMBER_COLUMNS].values
        residuals_main = actual_main - base_preds

        residual_model = MultiOutputRegressor(Ridge(alpha=1.0))
        residual_model.fit(X, residuals_main)
        joblib.dump(residual_model, RESIDUAL_MODEL_FILE)
        log("✅ Trained and saved residual model for main numbers.")

        # Predict Powerball
        pb_preds = models[POWERBALL_COLUMN].predict(X).astype(int)
        pb_preds = np.clip(pb_preds, 1, 10)
        actual_pb = df[POWERBALL_COLUMN].values
        residuals_pb = actual_pb - pb_preds

        pb_residual_model = GradientBoostingRegressor(n_estimators=150, random_state=42)
        pb_residual_model.fit(X, residuals_pb)
        joblib.dump(pb_residual_model, RESIDUAL_MODEL_PB_FILE)
        log("✅ Trained and saved residual model for PowerBall.")

    except Exception as e:
        err(f"❌ Failed training residual models: {e}", exc_info=True)


# CLI usage
if __name__ == "__main__":
    df = pd.read_excel("data/draw_history.xlsx", parse_dates=["Draw Date"])
    df.columns = df.columns.astype(str).str.strip()
    train_residual_models(df)
