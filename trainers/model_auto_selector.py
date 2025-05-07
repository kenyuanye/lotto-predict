# utils/model_auto_selector.py

import os
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error

MODEL_DIR = "models"

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]
POWERBALL_COLUMN = "Power Ball"

def load_model(model_name):
    try:
        return joblib.load(os.path.join(MODEL_DIR, model_name))
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None

def evaluate_model(model, X, y_true):
    try:
        preds = model.predict(X)
        if preds.ndim == 2:
            errors = [mean_absolute_error(y_true[:, i], preds[:, i]) for i in range(preds.shape[1])]
            return np.mean(errors)
        else:
            return mean_absolute_error(y_true, preds)
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        return np.inf

def auto_select_best_model(draw_df, for_powerball=False):
    """Auto-select the best model: real, synthetic, blended based on validation MAE."""
    X = np.arange(len(draw_df)).reshape(-1, 1)

    if for_powerball:
        y = draw_df[POWERBALL_COLUMN].values
        model_files = {
            "Real": "model_powerball.pkl",
            "Synthetic": "model_powerball_synthetic.pkl",
            "Blended": "model_powerball_blended.pkl"
        }
    else:
        y = draw_df[NUMBER_COLUMNS].values
        model_files = {
            "Real": "model_fullset.pkl",
            "Synthetic": "model_fullset_synthetic.pkl",
            "Blended": "model_fullset_blended.pkl"
        }

    results = {}
    for name, file in model_files.items():
        model = load_model(file)
        if model:
            mae = evaluate_model(model, X, y)
            results[name] = mae

    if not results:
        return None, {}

    best_model_type = min(results, key=results.get)
    best_model_path = model_files[best_model_type]
    best_model = load_model(best_model_path)

    return best_model, results
