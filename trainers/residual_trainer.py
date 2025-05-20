import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from core.feature_engineering import build_features_for_prediction
from core import NUMBER_COLUMNS, POWERBALL_COLUMN
from core.model_utils import load_models

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_residual_models(draw_df: pd.DataFrame, logger=None):
    log = logger.info if logger else print
    err = logger.error if logger else (lambda msg, **kwargs: print(msg))

    try:
        df = draw_df.copy().reset_index(drop=True)
        df["DrawIndex"] = range(len(df))
        df["Draw Number"] = range(len(df))

        X = build_features_for_prediction(df)
        models = load_models()

        # Validate base model availability
        missing = [str(i) for i in range(1, 7) if models.get(str(i)) is None]
        if missing:
            raise ValueError(f"Missing base models for positions: {missing}")
        if POWERBALL_COLUMN not in models:
            raise ValueError("Missing base PowerBall model.")

        # Predict base and calculate residuals for main numbers
        base_preds = np.column_stack([models[str(i)].predict(X) for i in range(1, 7)])
        residuals = df[NUMBER_COLUMNS].values - base_preds

        residual_model = MultiOutputRegressor(Ridge(alpha=1.0))
        residual_model.fit(X, residuals)
        joblib.dump(residual_model, os.path.join(MODEL_DIR, "residual_model.pkl"))
        log("✅ Trained and saved residual model for main numbers")

        # Powerball residual
        y_pb_actual = df[POWERBALL_COLUMN].values
        y_pb_pred = models[POWERBALL_COLUMN].predict(X)
        residual_pb = y_pb_actual - y_pb_pred

        pb_residual_model = GradientBoostingRegressor(n_estimators=150, random_state=42)
        pb_residual_model.fit(X, residual_pb)
        joblib.dump(pb_residual_model, os.path.join(MODEL_DIR, "residual_model_powerball.pkl"))
        log("✅ Trained and saved residual model for PowerBall")

    except Exception as e:
        err(f"❌ Failed training residual models: {e}", exc_info=True)

# CLI entry
if __name__ == "__main__":
    df = pd.read_excel("data/draw_history.xlsx", parse_dates=["Draw Date"])
    df.columns = df.columns.astype(str).str.strip()
    train_residual_models(df)
