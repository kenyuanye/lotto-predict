# legacy_prediction/residual_predictor.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from utils.feature_engineering import build_features_for_prediction
from utils import NUMBER_COLUMNS, POWERBALL_COLUMN

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

RESIDUAL_MODEL_FILE = os.path.join(MODEL_DIR, "residual_model.pkl")
RESIDUAL_MODEL_PB_FILE = os.path.join(MODEL_DIR, "residual_model_powerball.pkl")


def train_residual_models(draw_df, models, logger=None):
    """Train residual models based on prediction errors from the base models."""
    try:
        if "full" not in models or POWERBALL_COLUMN not in models:
            raise KeyError("Missing 'full' or Power Ball model in provided models.")

        draw_df = draw_df.copy().reset_index(drop=True)
        draw_df["DrawIndex"] = draw_df.index
        draw_df["Draw Number"] = draw_df.index

        X = build_features_for_prediction(draw_df)

        # Train main number residuals
        preds_full = np.round(models["full"].predict(X)).astype(int)
        preds_full = np.clip(preds_full, 1, 40)
        actual_full = draw_df[NUMBER_COLUMNS].values
        residuals = actual_full - preds_full

        residual_model = MultiOutputRegressor(Ridge())
        residual_model.fit(X, residuals)
        joblib.dump(residual_model, RESIDUAL_MODEL_FILE)

        # Train PowerBall residuals
        preds_pb = models[POWERBALL_COLUMN].predict(X).astype(int)
        preds_pb = np.clip(preds_pb, 1, 10)
        actual_pb = draw_df[POWERBALL_COLUMN].values
        residuals_pb = actual_pb - preds_pb

        residual_model_pb = Ridge()
        residual_model_pb.fit(X, residuals_pb)
        joblib.dump(residual_model_pb, RESIDUAL_MODEL_PB_FILE)

        if logger:
            logger.info("✅ Residual models trained and saved successfully.")
        else:
            print("✅ Residual models trained and saved successfully.")

    except Exception as e:
        msg = f"❌ Failed to train residual models: {e}"
        if logger:
            logger.error(msg, exc_info=True)
        else:
            print(msg)


def predict_with_residuals(draw_df, models):
    """Predict next draw numbers using base + residual models."""
    try:
        if "full" not in models or POWERBALL_COLUMN not in models:
            raise KeyError("Missing required base model: 'full' or 'Power Ball'")

        if not os.path.exists(RESIDUAL_MODEL_FILE) or not os.path.exists(RESIDUAL_MODEL_PB_FILE):
            raise FileNotFoundError("Residual model files not found.")

        residual_model = joblib.load(RESIDUAL_MODEL_FILE)
        residual_model_pb = joblib.load(RESIDUAL_MODEL_PB_FILE)

        # Build prediction features
        latest_df = draw_df.copy()
        latest_df["DrawIndex"] = latest_df.index
        latest_df["Draw Number"] = latest_df.index
        X_pred = build_features_for_prediction(latest_df).iloc[[-1]]

        # Main numbers: base + residual correction
        base_pred = np.round(models["full"].predict(X_pred)).astype(int).flatten()
        base_pred = np.clip(base_pred, 1, 40)

        correction = residual_model.predict(X_pred).flatten()
        corrected_main = np.clip(np.round(base_pred + correction).astype(int), 1, 40)

        # Ensure 6 unique main numbers
        unique_main = sorted(set(corrected_main))
        while len(unique_main) < 6:
            candidate = np.random.randint(1, 41)
            if candidate not in unique_main:
                unique_main.append(candidate)
        final_main = sorted(unique_main[:6])

        # Powerball prediction
        base_pb = int(models[POWERBALL_COLUMN].predict(X_pred)[0])
        base_pb = np.clip(base_pb, 1, 10)

        residual_pb = int(residual_model_pb.predict(X_pred)[0])
        corrected_pb = int(np.clip(base_pb + residual_pb, 1, 10))

        final_set = final_main + [corrected_pb]

        if len(final_set) != 7 or any(not isinstance(n, int) for n in final_set):
            print(f"[residual] Invalid prediction: {final_set}")
            return []

        return [final_set]

    except Exception as e:
        print(f"❌ Failed to predict with residuals: {e}")
        return []
