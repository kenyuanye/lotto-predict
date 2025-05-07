# utils/symbolic_vs_rf_comparison.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score
import logging

MODEL_DIR_SYMBOLIC = "models_symbolic"
MODEL_DIR_RF = "models"

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]
POWERBALL_COLUMN = "Power Ball"

def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        logging.error(f"❌ Failed to load model {path}: {e}", exc_info=True)
        return None

def compare_model_accuracy(draw_df: pd.DataFrame):
    """Compare Symbolic vs RandomForest model accuracy side-by-side."""
    results = []

    df = draw_df.copy().reset_index(drop=True)
    df["DrawIndex"] = df.index
    X = df[["DrawIndex"]]

    for col in NUMBER_COLUMNS + [POWERBALL_COLUMN]:
        symbolic_model_path = os.path.join(MODEL_DIR_SYMBOLIC, f"symbolic_model_pos{col}.pkl") if col != "Power Ball" else os.path.join(MODEL_DIR_SYMBOLIC, "symbolic_model_powerball.pkl")
        rf_model_path = os.path.join(MODEL_DIR_RF, f"model_pos{col}.pkl") if col != "Power Ball" else os.path.join(MODEL_DIR_RF, "model_powerball.pkl")

        symbolic_model = load_model(symbolic_model_path)
        rf_model = load_model(rf_model_path)

        if symbolic_model is None or rf_model is None:
            logging.warning(f"⚠️ Skipping {col} due to missing model(s)")
            continue

        y_true = df[col]

        try:
            symbolic_preds = np.round(symbolic_model.predict(X)).astype(int)
            rf_preds = np.round(rf_model.predict(X)).astype(int)

            if col == "Power Ball":
                symbolic_preds = np.clip(symbolic_preds, 1, 10)
                rf_preds = np.clip(rf_preds, 1, 10)
            else:
                symbolic_preds = np.clip(symbolic_preds, 1, 40)
                rf_preds = np.clip(rf_preds, 1, 40)

            sym_mae = mean_absolute_error(y_true, symbolic_preds)
            rf_mae = mean_absolute_error(y_true, rf_preds)

            sym_acc = accuracy_score(y_true, symbolic_preds)
            rf_acc = accuracy_score(y_true, rf_preds)

            # 🔥 Determine better model
            if sym_acc > rf_acc:
                better = "Symbolic"
            elif rf_acc > sym_acc:
                better = "RF"
            else:
                better = "Tie"

            results.append({
                "Position": col,
                "Symbolic MAE": round(sym_mae, 3),
                "RF MAE": round(rf_mae, 3),
                "Symbolic Accuracy %": round(sym_acc * 100, 2),
                "RF Accuracy %": round(rf_acc * 100, 2),
                "Better Model": better  # ✅ Always include Better Model
            })
        except Exception as e:
            logging.error(f"❌ Failed to compare for {col}: {e}", exc_info=True)

    if not results:
        logging.warning("⚠️ No model comparison results available.")
        return pd.DataFrame(columns=["Position", "Symbolic MAE", "RF MAE", "Symbolic Accuracy %", "RF Accuracy %", "Better Model"])

    return pd.DataFrame(results)
