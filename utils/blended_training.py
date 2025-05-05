# utils/blended_training.py

import os
import pandas as pd
import numpy as np
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

def load_real_data(filepath=os.path.join(DATA_DIR, "draw_history.xlsx")):
    """Load real historical draw data."""
    try:
        df = pd.read_excel(filepath, parse_dates=["Draw Date"])
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        print(f"❌ Failed to load real draw history: {e}")
        return pd.DataFrame()

def load_synthetic_data(filepath=os.path.join(DATA_DIR, "reverse_engineered_sets.csv")):
    """Load reverse engineered synthetic sets."""
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"❌ Failed to load synthetic sets: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def prepare_combined_dataset(real_df, synthetic_df, real_weight=0.7):
    """Blend real and synthetic datasets."""
    real_df = real_df.copy().reset_index(drop=True)
    real_df["DrawIndex"] = real_df.index
    real_X = real_df[["DrawIndex"]].values
    real_Y_numbers = real_df[NUMBER_COLUMNS].values
    real_Y_powerball = real_df[POWERBALL_COLUMN].values

    numbers = synthetic_df["Numbers"].apply(lambda x: list(map(int, str(x).split(", "))))
    pb = synthetic_df["Powerball"]
    syn_X = np.arange(len(synthetic_df)).reshape(-1, 1)
    syn_Y_numbers = np.vstack(numbers.values)
    syn_Y_powerball = pb.values

    # Determine blend sizes
    total_size = len(real_X) + len(syn_X)
    real_size = int(real_weight * total_size)
    synthetic_size = total_size - real_size

    # Handle case when not enough data
    real_X = real_X[:real_size] if len(real_X) >= real_size else real_X
    real_Y_numbers = real_Y_numbers[:real_size] if len(real_Y_numbers) >= real_size else real_Y_numbers
    real_Y_powerball = real_Y_powerball[:real_size] if len(real_Y_powerball) >= real_size else real_Y_powerball

    syn_X = syn_X[:synthetic_size] if len(syn_X) >= synthetic_size else syn_X
    syn_Y_numbers = syn_Y_numbers[:synthetic_size] if len(syn_Y_numbers) >= synthetic_size else syn_Y_numbers
    syn_Y_powerball = syn_Y_powerball[:synthetic_size] if len(syn_Y_powerball) >= synthetic_size else syn_Y_powerball

    combined_X = np.vstack([real_X, syn_X])
    combined_Y_numbers = np.vstack([real_Y_numbers, syn_Y_numbers])
    combined_Y_powerball = np.concatenate([real_Y_powerball, syn_Y_powerball])

    return combined_X, combined_Y_numbers, combined_Y_powerball

def train_blended_model(real_weight=0.7, logger=None):
    """Train a model using blended real + synthetic datasets."""
    log = logger.info if logger else print
    err = logger.error if logger else print

    real_df = load_real_data()
    synthetic_df = load_synthetic_data()

    if real_df.empty or synthetic_df.empty:
        err("❌ Real or Synthetic data missing. Cannot train blended model.")
        return

    X, Y_numbers, Y_powerball = prepare_combined_dataset(real_df, synthetic_df, real_weight=real_weight)

    try:
        # Train full number set model
        full_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42))
        full_model.fit(X, Y_numbers)
        joblib.dump(full_model, os.path.join(MODEL_DIR, "model_fullset_blended.pkl"))
        log(f"✅ Trained and saved blended full set model with real_weight={real_weight}")

        # Train Powerball separately
        pb_model = RandomForestRegressor(n_estimators=200, random_state=42)
        pb_model.fit(X, Y_powerball)
        joblib.dump(pb_model, os.path.join(MODEL_DIR, "model_powerball_blended.pkl"))
        log(f"✅ Trained and saved blended PowerBall model with real_weight={real_weight}")

    except Exception as e:
        err(f"❌ Failed to train blended model: {e}", exc_info=True)

# --- Entry Point ---
if __name__ == "__main__":
    train_blended_model()
