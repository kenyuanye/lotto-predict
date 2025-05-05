# utils/model_utils.py

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
