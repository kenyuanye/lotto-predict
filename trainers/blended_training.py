# trainers/blended_training.py

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.feature_engineering import build_features_for_prediction
from core import NUMBER_COLUMNS, POWERBALL_COLUMN

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_real_data():
    """Load historical draw data from Excel."""
    path = os.path.join(DATA_DIR, "draw_history.xlsx")
    try:
        df = pd.read_excel(path, parse_dates=["Draw Date"])
        df.columns = df.columns.astype(str).str.strip()
        df = df.dropna(subset=NUMBER_COLUMNS + [POWERBALL_COLUMN])
        df["DrawIndex"] = range(len(df))
        df["Draw Number"] = range(len(df))
        return df
    except Exception as e:
        print(f"❌ Failed to load real draw history: {e}")
        return pd.DataFrame()

def load_synthetic_data():
    """Load reverse engineered sets from CSV."""
    path = os.path.join(DATA_DIR, "reverse_engineered_sets.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        df = df.dropna(subset=["Numbers", "Powerball"])
        df[NUMBER_COLUMNS] = df["Numbers"].apply(lambda x: pd.Series(list(map(int, str(x).split(", ")))))
        df[POWERBALL_COLUMN] = df["Powerball"].astype(int)
        df["DrawIndex"] = range(len(df))
        df["Draw Number"] = range(len(df))
        return df
    except Exception as e:
        print(f"❌ Failed to load synthetic sets: {e}")
        return pd.DataFrame()

def prepare_blended_features(real_df, synthetic_df, real_weight=0.7):
    """Blend real and synthetic datasets, build prediction features."""
    real_n = int(real_weight * 1000)
    syn_n = 1000 - real_n

    real_sample = real_df.sample(n=min(real_n, len(real_df)), random_state=42)
    syn_sample = synthetic_df.sample(n=min(syn_n, len(synthetic_df)), random_state=42)

    blended = pd.concat([real_sample, syn_sample]).reset_index(drop=True)
    blended["DrawIndex"] = range(len(blended))
    blended["Draw Number"] = range(len(blended))

    X = build_features_for_prediction(blended)
    Y_numbers = blended[NUMBER_COLUMNS].values
    Y_powerball = blended[POWERBALL_COLUMN].values

    return X, Y_numbers, Y_powerball

def train_blended_model(real_weight=0.7, logger=None):
    """Train and save models using blended data."""
    log = logger.info if logger else print
    err = logger.error if logger else lambda msg, **kwargs: print(msg)

    real_df = load_real_data()
    synthetic_df = load_synthetic_data()

    if real_df.empty or synthetic_df.empty:
        err("❌ Real or Synthetic data missing. Cannot train blended model.")
        return

    X, Y_numbers, Y_powerball = prepare_blended_features(real_df, synthetic_df, real_weight=real_weight)

    try:
        # Train main number set model
        full_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42))
        full_model.fit(X, Y_numbers)
        joblib.dump(full_model, os.path.join(MODEL_DIR, "model_fullset_blended.pkl"))
        log(f"✅ Trained and saved blended full set model (real_weight={real_weight})")

        # Train PowerBall model
        pb_model = RandomForestRegressor(n_estimators=200, random_state=42)
        pb_model.fit(X, Y_powerball)
        joblib.dump(pb_model, os.path.join(MODEL_DIR, "model_powerball_blended.pkl"))
        log(f"✅ Trained and saved blended PowerBall model (real_weight={real_weight})")

    except Exception as e:
        err(f"❌ Failed to train blended model: {e}", exc_info=True)

# --- Optional CLI entry ---
if __name__ == "__main__":
    train_blended_model()
