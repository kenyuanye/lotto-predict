# trainers/symbolic_trainer.py

import os
import sys
import joblib
import pandas as pd
from gplearn.genetic import SymbolicRegressor

# Enable parent directory access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.feature_engineering import build_features_for_prediction
from utils import NUMBER_COLUMNS, POWERBALL_COLUMN

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "models_symbolic")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_symbolic_models(draw_df: pd.DataFrame, logger=None):
    """Train SymbolicRegressor models for all positions using full features."""
    log = logger.info if logger else print
    err = logger.error if logger else (lambda msg, **kwargs: print(msg))

    try:
        df = draw_df.copy().reset_index(drop=True)
        df["DrawIndex"] = range(len(df))
        df["Draw Number"] = range(len(df))  # Required for some features
        X = build_features_for_prediction(df)

        for i, col in enumerate(NUMBER_COLUMNS):
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
                n_jobs=-1
            )
            model.fit(X, y)
            model_path = os.path.join(MODEL_DIR, f"symbolic_model_pos{i+1}.pkl")
            joblib.dump(model, model_path)
            log(f"✅ Symbolic model for position {col} saved → {model_path}")

        # Train Powerball model
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
        pb_path = os.path.join(MODEL_DIR, "symbolic_model_powerball.pkl")
        joblib.dump(model_pb, pb_path)
        log(f"✅ Symbolic model for PowerBall saved → {pb_path}")

    except Exception as e:
        err(f"❌ Symbolic training failed: {e}", exc_info=True)

# --- Optional CLI entry ---
if __name__ == "__main__":
    try:
        df = pd.read_excel("data/draw_history.xlsx", parse_dates=["Draw Date"])
        df.columns = df.columns.astype(str).str.strip()
        train_symbolic_models(df)
    except Exception as e:
        print(f"❌ Failed to load data or run training: {e}")
