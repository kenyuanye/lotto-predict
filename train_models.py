# train_models.py

import os
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from utils.feature_engineering import build_features_for_prediction

logger = logging.getLogger("train_models")
logging.basicConfig(level=logging.INFO)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]
POWERBALL_COLUMN = "Power Ball"

MODEL_FILE_NAMES = {
    **{col: f"model_pos{i+1}.pkl" for i, col in enumerate(NUMBER_COLUMNS)},
    "full": "model_fullset.pkl",
    POWERBALL_COLUMN: "model_powerball.pkl",
}

def train_models(draw_df: pd.DataFrame, logger=None):
    log = logger.info if logger else print
    err = logger.error if logger else print

    df = draw_df.copy().sort_values("Draw Number").reset_index(drop=True)
    df["DrawIndex"] = df.index

    missing_before = df[NUMBER_COLUMNS + [POWERBALL_COLUMN]].isnull().sum().sum()
    df = df.dropna(subset=NUMBER_COLUMNS + [POWERBALL_COLUMN])
    missing_after = df[NUMBER_COLUMNS + [POWERBALL_COLUMN]].isnull().sum().sum()

    log(f"⚠️ Dropped {missing_before - missing_after} missing values from training data")

    try:
        X = build_features_for_prediction(df)
        Y = df[NUMBER_COLUMNS].astype(int).values
        Y_pb = df[POWERBALL_COLUMN].astype(int).values

        # Train individual models
        for i, col in enumerate(NUMBER_COLUMNS):
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X, Y[:, i])
            joblib.dump(model, os.path.join(MODEL_DIR, MODEL_FILE_NAMES[col]))
            log(f"✅ Trained model for number position {col}")

        # Train MultiOutput full-set model
        full_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42))
        full_model.fit(X, Y)
        joblib.dump(full_model, os.path.join(MODEL_DIR, MODEL_FILE_NAMES["full"]))
        log("✅ Trained full-set model")

        # Train PowerBall model
        pb_model = RandomForestClassifier(n_estimators=200, random_state=42)
        pb_model.fit(X, Y_pb)
        joblib.dump(pb_model, os.path.join(MODEL_DIR, MODEL_FILE_NAMES[POWERBALL_COLUMN]))
        log("✅ Trained PowerBall model")

    except Exception as e:
        err(f"❌ Model training failed: {e}", exc_info=True)

if __name__ == "__main__":
    df = pd.read_excel("data/draw_history.xlsx", parse_dates=["Draw Date"])
    train_models(df)

