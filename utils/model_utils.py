import os
import joblib
import logging

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
logging.basicConfig(level=logging.INFO)

def get_available_models():
    available = {}

    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        available["RandomForest"] = RandomForestRegressor
        available["GradientBoosting"] = GradientBoostingRegressor
    except ImportError:
        pass

    try:
        from sklearn.neural_network import MLPRegressor
        available["MLPRegressor"] = MLPRegressor
    except ImportError:
        pass

    try:
        from xgboost import XGBRegressor
        available["XGBoost"] = XGBRegressor
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        available["LightGBM"] = LGBMRegressor
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor
        available["CatBoost"] = CatBoostRegressor
    except ImportError:
        pass

    return available


def get_missing_model_libraries():
    missing = []

    try:
        from xgboost import XGBRegressor
    except ImportError:
        missing.append(("XGBoost", "pip install xgboost"))

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        missing.append(("LightGBM", "pip install lightgbm"))

    try:
        from catboost import CatBoostRegressor
    except ImportError:
        missing.append(("CatBoost", "pip install catboost"))

    return missing


def load_models():
    models = {}
    try:
        # Load Full-Set Model
        full_model_path = os.path.join(MODEL_DIR, "model_fullset.pkl")
        if os.path.exists(full_model_path):
            models["full"] = joblib.load(full_model_path)
            logging.info("✅ Loaded model_fullset.pkl (full)")
        else:
            logging.warning("⚠️ model_fullset.pkl (full) not found.")

        # Load PowerBall Model
        pb_path = os.path.join(MODEL_DIR, "model_powerball.pkl")
        if os.path.exists(pb_path):
            models["Power Ball"] = joblib.load(pb_path)
            logging.info("✅ Loaded model_powerball.pkl")
        else:
            logging.warning("⚠️ model_powerball.pkl not found.")

        # Load Individual Position Models
        for i in range(1, 7):
            pos_path = os.path.join(MODEL_DIR, f"model_pos{i}.pkl")
            if os.path.exists(pos_path):
                models[str(i)] = joblib.load(pos_path)
                logging.info(f"✅ Loaded model_pos{i}.pkl")
            else:
                logging.warning(f"⚠️ model_pos{i}.pkl not found.")

        # Final Check for Critical Dependencies
        if "full" not in models:
            logging.error("❌ 'full' model missing from loaded models. Residual prediction will fail.")
        if "Power Ball" not in models:
            logging.error("❌ 'Power Ball' model missing. PowerBall prediction will fail.")

        return models
    except Exception as e:
        logging.error(f"❌ Failed to load models: {e}", exc_info=True)
        return {}
