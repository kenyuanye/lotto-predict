# utils/symbolic_trainer.py

import os
import joblib
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
import logging

MODEL_DIR = "models_symbolic"
os.makedirs(MODEL_DIR, exist_ok=True)

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]
POWERBALL_COLUMN = "Power Ball"

def train_symbolic_models(draw_df: pd.DataFrame, logger=None):
    """Ultra-High-Accuracy symbolic regression training."""
    log = logger.info if logger else print
    err = logger.error if logger else print

    try:
        df = draw_df.copy().reset_index(drop=True)
        df["DrawIndex"] = df.index
        X = df[["DrawIndex"]]

        for col in NUMBER_COLUMNS:
            y = df[col]
            model = SymbolicRegressor(
                population_size=1500,
                generations=200,
                stopping_criteria=0.001,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                verbose=1,
                parsimony_coefficient=0.0001,
                random_state=42,
                n_jobs=-1  # <-- Use all CPU cores for faster training
            )
            model.fit(X, y)
            model_path = os.path.join(MODEL_DIR, f"symbolic_model_pos{col}.pkl")
            joblib.dump(model, model_path)
            log(f"🧠 Ultra Symbolic model trained for position {col} and saved to {model_path}")

        # Train PowerBall separately
        y_pb = df[POWERBALL_COLUMN]
        model_pb = SymbolicRegressor(
            population_size=1500,
            generations=200,
            stopping_criteria=0.001,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=1,
            parsimony_coefficient=0.0001,
            random_state=42,
            n_jobs=-1
        )
        model_pb.fit(X, y_pb)
        model_path_pb = os.path.join(MODEL_DIR, "symbolic_model_powerball.pkl")
        joblib.dump(model_pb, model_path_pb)
        log(f"🧠 Ultra Symbolic model trained for PowerBall and saved to {model_path_pb}")

    except Exception as e:
        err(f"❌ Symbolic model training failed: {e}", exc_info=True)

