# utils/trainer.py

import os
import joblib
import pandas as pd
import logging
from train_models import train_models  # ML trainer
from trainers.symbolic_trainer import train_symbolic_models

from core.feature_engineering import build_features_for_prediction

MODEL_PATH = "models"

logger = logging.getLogger("trainer")

def run_all_trainers(draw_df: pd.DataFrame):
    """
    Run all model training workflows (ML + symbolic).
    """
    logger.info("✅ Starting full training pipeline...")
    draw_df = draw_df.sort_values("Draw Number").reset_index(drop=True)

    # Ensure correct features are built
    _ = build_features_for_prediction(draw_df)

    train_models(draw_df, logger=logger)
    train_symbolic_models(draw_df)

    logger.info("✅ All training complete.")

def run_sequential_training(df: pd.DataFrame, logger=None):
    """
    Run sequential one-step-ahead training for accuracy simulation.
    """
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    import numpy as np

    df = df.copy().sort_values("Draw Number").reset_index(drop=True)
    df["DrawIndex"] = df.index

    NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]
    POWERBALL_COLUMN = "Power Ball"

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


def load_models():
    """
    Load all trained models from the models directory.
    Returns:
        dict: key → model name, value → loaded model
    """
    models = {}
    if not os.path.exists(MODEL_PATH):
        logging.warning("⚠️ Model path does not exist.")
        return models

    for filename in os.listdir(MODEL_PATH):
        if filename.endswith(".joblib"):
            full_path = os.path.join(MODEL_PATH, filename)
            try:
                model = joblib.load(full_path)
                model_name = os.path.splitext(filename)[0]
                models[model_name] = model
            except Exception as e:
                logging.error(f"❌ Failed to load model {filename}: {e}")
    return models


# Optional entrypoint
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    draw_df = pd.read_excel("data/draw_history.xlsx", parse_dates=["Draw Date"])
    run_all_trainers(draw_df)
