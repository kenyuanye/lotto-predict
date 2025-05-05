# utils/synthetic_training.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# --- Correct Base Path ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Go up from utils/
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODEL_DIR, exist_ok=True)

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]
POWERBALL_COLUMN = "Power Ball"

def load_reverse_sets(filepath=os.path.join(DATA_DIR, "reverse_engineered_sets.csv")):
    """Load reverse engineered sets for synthetic training."""
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"❌ Failed to load reverse sets: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def prepare_synthetic_data(df):
    """Prepare X (indexes) and Y (numbers) for training."""
    if df.empty:
        return None, None

    numbers = df["Numbers"].apply(lambda x: list(map(int, str(x).split(", "))))
    powerballs = df["Powerball"]

    X = np.arange(len(df)).reshape(-1, 1)
    Y_numbers = np.vstack(numbers.values)
    Y_powerball = powerballs.values

    return X, Y_numbers, Y_powerball

def train_on_synthetic_data(logger=None):
    """Train models based purely on reverse engineered sets."""
    log = logger.info if logger else print
    err = logger.error if logger else print

    synthetic_df = load_reverse_sets()

    if synthetic_df.empty:
        err("❌ No synthetic data found to train on.")
        return

    X, Y_numbers, Y_powerball = prepare_synthetic_data(synthetic_df)

    try:
        # Train full set model
        full_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
        full_model.fit(X, Y_numbers)
        joblib.dump(full_model, os.path.join(MODEL_DIR, "model_fullset_synthetic.pkl"))
        log("✅ Trained and saved full set model (synthetic)")

        # Train powerball model separately
        pb_model = RandomForestRegressor(n_estimators=100, random_state=42)
        pb_model.fit(X, Y_powerball)
        joblib.dump(pb_model, os.path.join(MODEL_DIR, "model_powerball_synthetic.pkl"))
        log("✅ Trained and saved PowerBall model (synthetic)")

    except Exception as e:
        err(f"❌ Failed to train on synthetic data: {e}", exc_info=True)

# --- Entry Point ---
if __name__ == "__main__":
    train_on_synthetic_data()
