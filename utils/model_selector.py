# utils/model_selector.py

import os
import joblib
import numpy as np
import pandas as pd
import logging

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from collections import Counter

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]
POWERBALL_COLUMN = "Power Ball"

def prepare_features(df):
    """Prepare features for training."""
    df = df.copy()
    if "DrawIndex" not in df.columns:
        df["DrawIndex"] = df.index
    return df[["DrawIndex"]]

def evaluate_models(df):
    """
    Evaluate different models and return (full evaluation, best model summary).
    """
    df = df.copy().reset_index(drop=True)
    X = prepare_features(df)

    results = []
    best_models = {}
    best_model_names = []

    regressors = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Ridge": Ridge(),
        "LinearRegression": LinearRegression()
    }

    for col in NUMBER_COLUMNS:
        y = df[col]
        model_errors = {}

        for name, model in regressors.items():
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            error = mean_absolute_error(y_test, preds)
            model_errors[name] = error

        best_name = min(model_errors, key=model_errors.get)
        best_models[col] = regressors[best_name]
        best_model_names.append(best_name)

        for name, error in model_errors.items():
            results.append({"Position": col, "Model": name, "MAE": error})

    # Powerball prediction
    y_pb = df[POWERBALL_COLUMN]
    pb_errors = {}
    for name, model in regressors.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y_pb, test_size=0.25, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        error = mean_absolute_error(y_test, preds)
        pb_errors[name] = error

    best_pb = min(pb_errors, key=pb_errors.get)
    best_models[POWERBALL_COLUMN] = regressors[best_pb]
    best_model_names.append(best_pb)

    for name, error in pb_errors.items():
        results.append({"Position": POWERBALL_COLUMN, "Model": name, "MAE": error})

    # Full set model (for information, no alternative models)
    Y_full = df[NUMBER_COLUMNS]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_full, test_size=0.25, random_state=42)
    full_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    full_model.fit(X_train, Y_train)
    preds = full_model.predict(X_test)
    full_error = np.mean([
        mean_absolute_error(Y_test.iloc[:, i], preds[:, i]) for i in range(6)
    ])
    results.append({"Position": "Full Set", "Model": "RandomForest", "MAE": full_error})

    # Summarize best models
    summary_counter = Counter(best_model_names)
    summary_df = pd.DataFrame.from_dict(summary_counter, orient="index", columns=["Best Model Count"]).reset_index()
    summary_df = summary_df.rename(columns={"index": "Model"})

    return pd.DataFrame(results), summary_df

def train_and_save_best_models(df, logger=None):
    """
    Train best models based on evaluation and save to disk.
    """
    try:
        df = df.copy().reset_index(drop=True)
        X = prepare_features(df)
        Y_full = df[NUMBER_COLUMNS]

        # Evaluate to find best
        full_results, _ = evaluate_models(df)
        logger.info("📊 Model evaluation completed for saving.")

        # Re-train full set model
        full_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
        full_model.fit(X, Y_full)
        joblib.dump(full_model, os.path.join(MODEL_DIR, "model_fullset.pkl"))
        logger.info("✅ Saved full set model.")

        # Position-based models
        regressors = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Ridge": Ridge(),
            "LinearRegression": LinearRegression()
        }

        for col in NUMBER_COLUMNS:
            y = df[col]
            best_model_name = full_results.query(f"Position == '{col}'").sort_values("MAE").iloc[0]["Model"]
            model = regressors[best_model_name]
            model.fit(X, y)
            joblib.dump(model, os.path.join(MODEL_DIR, f"model_pos{col}.pkl"))
            logger.info(f"✅ Saved best model for position {col}: {best_model_name}")

        # Powerball model
        y_pb = df[POWERBALL_COLUMN]
        best_pb_model_name = full_results.query(f"Position == '{POWERBALL_COLUMN}'").sort_values("MAE").iloc[0]["Model"]
        pb_model = regressors[best_pb_model_name]
        pb_model.fit(X, y_pb)
        joblib.dump(pb_model, os.path.join(MODEL_DIR, "model_powerball.pkl"))
        logger.info(f"✅ Saved best Powerball model: {best_pb_model_name}")

    except Exception as e:
        if logger:
            logger.error(f"❌ Failed during best model training/saving: {e}", exc_info=True)
        else:
            print(f"❌ Failed during best model training/saving: {e}")
