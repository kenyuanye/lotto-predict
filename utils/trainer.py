# utils/trainer.py

import os
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]
POWERBALL_COLUMN = "Power Ball"

MODEL_FILE_NAMES = {
    **{col: f"model_pos{i+1}.pkl" for i, col in enumerate(NUMBER_COLUMNS)},
    "full": "model_fullset.pkl",
    POWERBALL_COLUMN: "model_powerball.pkl",
}

def train_models(df, logger=None):
    df = df.copy()
    df = df.sort_values("Draw Date").reset_index(drop=True)
    df["DrawIndex"] = df.index

    missing_before = df[NUMBER_COLUMNS + [POWERBALL_COLUMN]].isnull().sum().sum()
    df = df.dropna(subset=NUMBER_COLUMNS + [POWERBALL_COLUMN])
    missing_after = df[NUMBER_COLUMNS + [POWERBALL_COLUMN]].isnull().sum().sum()

    log = logger.info if logger else print
    log(f"⚠️ Dropped {missing_before - missing_after} missing values from training data")

    X = df[["DrawIndex"]]
    Y = df[NUMBER_COLUMNS].values
    Y_pb = df[POWERBALL_COLUMN].values

    try:
        # Train individual number models
        for i, col in enumerate(NUMBER_COLUMNS):
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X, Y[:, i])
            joblib.dump(model, os.path.join(MODEL_DIR, MODEL_FILE_NAMES[col]))
            log(f"✅ Trained and saved model for number position {col}")

        # Full set model
        model_full = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42))
        model_full.fit(X, Y)
        joblib.dump(model_full, os.path.join(MODEL_DIR, MODEL_FILE_NAMES["full"]))
        log("✅ Trained and saved full set model")

        # Powerball model
        model_pb = RandomForestClassifier(n_estimators=200, random_state=42)
        model_pb.fit(X, Y_pb)
        joblib.dump(model_pb, os.path.join(MODEL_DIR, MODEL_FILE_NAMES[POWERBALL_COLUMN]))
        log("✅ Trained and saved PowerBall model")

    except Exception as e:
        err = logger.error if logger else print
        err(f"❌ Model training failed: {e}", exc_info=True)

def load_models():
    models = {}
    try:
        # Always load Full Set model
        full_model_path = os.path.join(MODEL_DIR, "model_fullset.pkl")
        if os.path.exists(full_model_path):
            models["full"] = joblib.load(full_model_path)
            logging.info("✅ Loaded full set model.")
        else:
            logging.warning("⚠️ Full set model (model_fullset.pkl) not found.")

        # Always load PowerBall model
        powerball_model_path = os.path.join(MODEL_DIR, "model_powerball.pkl")
        if os.path.exists(powerball_model_path):
            models["Power Ball"] = joblib.load(powerball_model_path)
            logging.info("✅ Loaded PowerBall model.")
        else:
            logging.warning("⚠️ PowerBall model (model_powerball.pkl) not found.")

        # Load per-position models 1–6 if available
        for i in range(1, 7):
            pos_model_path = os.path.join(MODEL_DIR, f"model_pos{i}.pkl")
            if os.path.exists(pos_model_path):
                models[str(i)] = joblib.load(pos_model_path)
                logging.info(f"✅ Loaded model_pos{i}.pkl for position {i}.")
            else:
                logging.info(f"ℹ️ Position model model_pos{i}.pkl not found. Skipping.")

        # Load optional synthetic models if available
        synthetic_full_path = os.path.join(MODEL_DIR, "model_fullset_synthetic.pkl")
        synthetic_pb_path = os.path.join(MODEL_DIR, "model_powerball_synthetic.pkl")
        if os.path.exists(synthetic_full_path):
            models["full_synthetic"] = joblib.load(synthetic_full_path)
            logging.info("✅ Loaded synthetic full set model.")
        if os.path.exists(synthetic_pb_path):
            models["powerball_synthetic"] = joblib.load(synthetic_pb_path)
            logging.info("✅ Loaded synthetic PowerBall model.")

        # Load optional blended models if available
        blended_full_path = os.path.join(MODEL_DIR, "model_fullset_blended.pkl")
        blended_pb_path = os.path.join(MODEL_DIR, "model_powerball_blended.pkl")
        if os.path.exists(blended_full_path):
            models["full_blended"] = joblib.load(blended_full_path)
            logging.info("✅ Loaded blended full set model.")
        if os.path.exists(blended_pb_path):
            models["powerball_blended"] = joblib.load(blended_pb_path)
            logging.info("✅ Loaded blended PowerBall model.")

        return models

    except Exception as e:
        logging.error(f"❌ Failed to load models: {e}", exc_info=True)
        return {}



def run_sequential_training(df, logger=None):
    df = df.copy().sort_values("Draw Number").reset_index(drop=True)
    df["DrawIndex"] = df.index

    log = logger.info if logger else print
    err = logger.error if logger else print

    total_rows = len(df)
    log(f"🚀 Starting sequential learning from {total_rows} draws")

    correct_counts = {col: 0 for col in NUMBER_COLUMNS + [POWERBALL_COLUMN]}
    prediction_attempts = 0

    for i in range(10, total_rows - 1):
        train_df = df.iloc[:i]
        test_row = df.iloc[i]

        try:
            X_train = train_df[["DrawIndex"]]
            Y_train = train_df[NUMBER_COLUMNS].values
            Y_pb_train = train_df[POWERBALL_COLUMN].values
            X_test = pd.DataFrame({"DrawIndex": [i]})

            preds = {}
            for j, col in enumerate(NUMBER_COLUMNS):
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, Y_train[:, j])
                pred = int(np.clip(np.round(model.predict(X_test))[0], 1, 40))
                preds[col] = pred

            pb_model = RandomForestClassifier(n_estimators=100, random_state=42)
            pb_model.fit(X_train, Y_pb_train)
            preds[POWERBALL_COLUMN] = int(np.clip(pb_model.predict(X_test)[0], 1, 10))

            prediction_attempts += 1

            for col in preds:
                if preds[col] == test_row[col]:
                    correct_counts[col] += 1

        except Exception as e:
            err(f"❌ Error in sequential iteration {i}: {e}", exc_info=True)
            continue

    summary = {
        col: round(100 * correct_counts[col] / prediction_attempts, 2)
        for col in correct_counts
    }

    log("📊 Sequential Training Accuracy (% correct per column):")
    for col, acc in summary.items():
        log(f"  {col}: {acc}%")

    return summary
