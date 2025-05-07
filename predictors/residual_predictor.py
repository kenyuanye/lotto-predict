# utils/residual_predictor.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]
POWERBALL_COLUMN = "Power Ball"

RESIDUAL_MODEL_FILE = os.path.join(MODEL_DIR, "residual_model.pkl")
RESIDUAL_MODEL_PB_FILE = os.path.join(MODEL_DIR, "residual_model_powerball.pkl")


def train_residual_models(draw_df, models, logger=None):
    """Train residual models based on prediction errors."""
    try:
        draw_df = draw_df.copy().reset_index(drop=True)
        X = draw_df[["DrawIndex"]]

        # Predict first pass
        preds_full = np.round(models["full"].predict(X)).astype(int)
        preds_full = np.clip(preds_full, 1, 40)  # clamp between 1–40

        # Calculate residuals
        actual_full = draw_df[NUMBER_COLUMNS].values
        residuals = actual_full - preds_full

        # Train residual model (multioutput regressor)
        residual_model = MultiOutputRegressor(Ridge())
        residual_model.fit(X, residuals)
        joblib.dump(residual_model, RESIDUAL_MODEL_FILE)

        # Powerball residual
        preds_pb = models[POWERBALL_COLUMN].predict(X).astype(int)
        preds_pb = np.clip(preds_pb, 1, 10)
        actual_pb = draw_df[POWERBALL_COLUMN].values
        residuals_pb = actual_pb - preds_pb

        residual_model_pb = Ridge()
        residual_model_pb.fit(X, residuals_pb)
        joblib.dump(residual_model_pb, RESIDUAL_MODEL_PB_FILE)

        if logger:
            logger.info("✅ Residual models trained and saved successfully.")

    except Exception as e:
        if logger:
            logger.error(f"❌ Failed to train residual models: {e}", exc_info=True)
        else:
            print(f"❌ Failed to train residual models: {e}")

def predict_with_residuals(draw_df, models):
    """Predict next draw numbers using base + residual models."""
    try:
        residual_model = joblib.load(RESIDUAL_MODEL_FILE)
        residual_model_pb = joblib.load(RESIDUAL_MODEL_PB_FILE)

        latest_index = draw_df["DrawIndex"].max() + 1
        X_pred = pd.DataFrame({"DrawIndex": [latest_index]})

        # Predict base
        base_pred = np.round(models["full"].predict(X_pred)).astype(int).flatten()
        base_pred = np.clip(base_pred, 1, 40)

        # Predict correction
        residual_correction = residual_model.predict(X_pred).flatten()
        corrected_nums = np.round(base_pred + residual_correction).astype(int)
        corrected_nums = np.clip(corrected_nums, 1, 40)

        # Predict Powerball base
        base_pb = int(models[POWERBALL_COLUMN].predict(X_pred)[0])
        base_pb = np.clip(base_pb, 1, 10)

        # Predict Powerball correction
        residual_pb = int(residual_model_pb.predict(X_pred)[0])
        corrected_pb = np.clip(base_pb + residual_pb, 1, 10)

        return list(corrected_nums) + [corrected_pb]

    except Exception as e:
        print(f"❌ Failed to predict with residuals: {e}")
        return []

