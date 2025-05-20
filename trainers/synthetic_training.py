# trainers/synthetic_training.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from core.feature_engineering import build_features_for_prediction
from core import NUMBER_COLUMNS, POWERBALL_COLUMN

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_reverse_sets(filepath=os.path.join(DATA_DIR, "reverse_engineered_sets.csv")):
    """Load reverse engineered sets for synthetic training."""
    if not os.path.exists(filepath):
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
        df["DrawIndex"] = range(len(df))
        return df
    except Exception as e:
        print(f"❌ Failed to load reverse sets: {e}")
        return pd.DataFrame()

def train_on_synthetic_data(logger=None):
    log = logger.info if logger else print
    err = logger.error if logger else (lambda msg, **kwargs: print(msg))

    df = load_reverse_sets()
    if df.empty:
        err("❌ No synthetic data found to train on.")
        return

    try:
        df = df.dropna(subset=["Numbers", "Powerball"])
        df[NUMBER_COLUMNS] = df["Numbers"].apply(lambda x: pd.Series(list(map(int, str(x).split(", ")))))
        df[POWERBALL_COLUMN] = df["Powerball"].astype(int)

        # Add required indexing columns for feature generation
        df["DrawIndex"] = range(len(df))
        df["Draw Number"] = range(len(df))  # Needed by feature generator

        X = build_features_for_prediction(df)
        Y_numbers = df[NUMBER_COLUMNS].values
        Y_powerball = df[POWERBALL_COLUMN].values

        # Train full-set model
        model_full = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
        model_full.fit(X, Y_numbers)
        joblib.dump(model_full, os.path.join(MODEL_DIR, "model_fullset_synthetic.pkl"))
        log("✅ Trained and saved full set model (synthetic)")

        # Train PowerBall model
        model_pb = RandomForestRegressor(n_estimators=100, random_state=42)
        model_pb.fit(X, Y_powerball)
        joblib.dump(model_pb, os.path.join(MODEL_DIR, "model_powerball_synthetic.pkl"))
        log("✅ Trained and saved PowerBall model (synthetic)")

    except Exception as e:
        err(f"❌ Failed to train on synthetic data: {e}", exc_info=True)

# --- Optional CLI entry point ---
if __name__ == "__main__":
    train_on_synthetic_data()
