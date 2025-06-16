# trainers/model_auto_selector.py

import os
import sys
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error

# Ensure proper imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.feature_engineering import build_features_for_prediction
from utils import NUMBER_COLUMNS, POWERBALL_COLUMN

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

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
    """
    Auto-select the best model (real, synthetic, blended) based on lowest validation MAE.
    Returns the best model and a dict of MAEs.
    """
    features = build_features_for_prediction(draw_df)

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
    for label, filename in model_files.items():
        model = load_model(filename)
        if model:
            mae = evaluate_model(model, features, y)
            results[label] = mae

    if not results:
        return None, {}

    best_model_label = min(results, key=results.get)
    best_model = load_model(model_files[best_model_label])

    return best_model, results
