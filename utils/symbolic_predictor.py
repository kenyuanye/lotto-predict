# utils/symbolic_predictor.py

import os
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator
from utils.custom_rules import powerball_bias_weight, calculate_powerball_gaps

MODEL_DIR_SYMBOLIC = "models_symbolic"
MODEL_DIR_RF = "models"

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]
POWERBALL_COLUMN = "Power Ball"

def load_model_with_fallback(position: str) -> tuple:
    """Load symbolic model if available, fallback to RF model if missing."""
    symbolic_path = os.path.join(MODEL_DIR_SYMBOLIC, f"symbolic_model_pos{position}.pkl")
    rf_path = os.path.join(MODEL_DIR_RF, f"model_pos{position}.pkl")

    if os.path.exists(symbolic_path):
        model = joblib.load(symbolic_path)
        logging.info(f"✅ Loaded symbolic model for position {position}")
        return model, 'Symbolic'
    elif os.path.exists(rf_path):
        model = joblib.load(rf_path)
        logging.warning(f"⚠️ Symbolic model missing for {position}, fallback to RF model.")
        return model, 'RF'
    else:
        logging.error(f"❌ No model found for position {position}")
        return None, None

def load_powerball_model_with_fallback() -> tuple:
    """Load symbolic PowerBall model or fallback to RF model."""
    symbolic_path = os.path.join(MODEL_DIR_SYMBOLIC, "symbolic_model_powerball.pkl")
    rf_path = os.path.join(MODEL_DIR_RF, "model_powerball.pkl")

    if os.path.exists(symbolic_path):
        model = joblib.load(symbolic_path)
        logging.info("✅ Loaded symbolic PowerBall model")
        return model, 'Symbolic'
    elif os.path.exists(rf_path):
        model = joblib.load(rf_path)
        logging.warning("⚠️ Symbolic PowerBall model missing, fallback to RF model.")
        return model, 'RF'
    else:
        logging.error("❌ No PowerBall model found")
        return None, None

def predict_symbolic(draw_df: pd.DataFrame, top_n: int = 10, powerball_gaps: dict = None) -> dict:
    """Predict using symbolic (or fallback RF) models."""
    features = draw_df[["DrawIndex"]]
    predictions = {}
    models = {}
    model_types = {}

    # Load models for all positions
    for col in NUMBER_COLUMNS:
        model, model_type = load_model_with_fallback(col)
        if model:
            models[col] = model
            model_types[col] = model_type

    pb_model, pb_model_type = load_powerball_model_with_fallback()
    if pb_model:
        models["Power Ball"] = pb_model
        model_types["Power Ball"] = pb_model_type

    logging.info(f"🔍 Loaded models: {model_types}")

    # Make predictions
    for col in models:
        try:
            preds = models[col].predict(features)
            preds = np.round(preds).astype(int)
            if col == "Power Ball":
                preds = np.clip(preds, 1, 10)
                # 🔥 Adjust PowerBall predictions if powerball_gaps provided
                if powerball_gaps:
                    adjusted_preds = []
                    for pred in preds:
                        weight = powerball_bias_weight(pred, powerball_gaps)
                        if np.random.rand() < weight:  # Higher gap = higher chance
                            adjusted_preds.append(pred)
                        else:
                            adjusted_preds.append(np.random.randint(1, 11))
                    preds = np.array(adjusted_preds)
            else:
                preds = np.clip(preds, 1, 40)
            predictions[col] = preds.tolist()
            logging.info(f"✅ Predictions done for {col}")
        except Exception as e:
            logging.error(f"❌ Prediction failed for {col}: {e}", exc_info=True)

    # --- De-duplicate main numbers for each prediction set ---
    sets = []
    for i in range(len(draw_df)):
        try:
            main_nums = [
                predictions["1"][i],
                predictions["2"][i],
                predictions["3"][i],
                predictions["4"][i],
                predictions["5"][i],
                predictions["6"][i],
            ]
            main_nums = sorted(set(main_nums))  # De-duplicate
            while len(main_nums) < 6:
                new_num = np.random.randint(1, 41)
                if new_num not in main_nums:
                    main_nums.append(new_num)
                main_nums = sorted(main_nums)  # Re-sort
            pb = int(predictions["Power Ball"][i])
            full_set = main_nums + [pb]
            sets.append(full_set)
        except Exception as e:
            logging.warning(f"⚠️ Failed to create set at index {i}: {e}", exc_info=True)

    return {
        "predictions": sets,
        "models": models,
        "model_types": model_types
    }

def extract_symbolic_formulas(models: dict) -> dict:
    """Extract discovered symbolic formulas."""
    formulas = {}
    for col, model in models.items():
        try:
            if hasattr(model, "_program"):
                formulas[col] = str(model._program)
                logging.info(f"📜 Extracted formula for {col}")
            else:
                formulas[col] = "⚠️ No symbolic program available."
        except Exception as e:
            formulas[col] = f"❌ Error extracting: {e}"
            logging.warning(f"⚠️ Failed to extract formula for {col}: {e}", exc_info=True)
    return formulas
