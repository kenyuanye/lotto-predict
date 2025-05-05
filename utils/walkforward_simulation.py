# utils/walkforward_simulation.py

import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from utils.predictor import generate_candidate_sets

# Paths
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]
POWERBALL_COLUMN = "Power Ball"

def walkforward_simulation(draw_df: pd.DataFrame, top_n: int = 10):
    logger = logging.getLogger("lotto")
    logger.info("🚶 Starting new walkforward simulation (with coverage evaluation)...")

    df = draw_df.copy().sort_values("Draw Number").reset_index(drop=True)
    df["DrawIndex"] = df.index
    results = []

    total_draws = len(df)

    for i in range(2, total_draws - 1):  # Start from index 2 (predicting 3rd draw onward)
        try:
            train_df = df.iloc[:i+1]
            predict_row = df.iloc[i+1]

            # Prepare training data
            X_train = train_df[["DrawIndex"]]
            Y_train_numbers = train_df[NUMBER_COLUMNS].values
            Y_train_pb = train_df[POWERBALL_COLUMN].values
            X_pred = pd.DataFrame({"DrawIndex": [predict_row["DrawIndex"]]})

            # Train fresh models
            full_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
            full_model.fit(X_train, Y_train_numbers)

            pb_model = RandomForestClassifier(n_estimators=200, random_state=42)
            pb_model.fit(X_train, Y_train_pb)

            # Predict candidate sets
            full_preds = full_model.predict(X_pred)
            pb_preds = pb_model.predict_proba(X_pred)

            preds_per_column = {}
            for idx, col in enumerate(NUMBER_COLUMNS):
                preds_per_column[col] = np.clip(np.round(full_preds[:, idx]).astype(int), 1, 40).tolist()

            preds_per_column["Power Ball"] = np.clip(np.argmax(pb_preds, axis=1) + 1, 1, 10).tolist()

            # Generate candidate sets
            candidate_sets = generate_candidate_sets(preds_per_column, top_n=30)

            top_picks = candidate_sets[:10]
            backup_picks = candidate_sets[10:20]
            unlikely_picks = candidate_sets[20:30]

            # Actual numbers
            actual_main = set(predict_row[NUMBER_COLUMNS].values)
            actual_pb = int(predict_row[POWERBALL_COLUMN])

            def calculate_coverage(sets):
                if not sets:
                    return 0
                all_numbers = set()
                for s in sets:
                    all_numbers.update(s[:6])
                matches = len(actual_main.intersection(all_numbers))
                return round(100 * matches / 6, 2)  # % coverage

            top_pick_coverage = calculate_coverage(top_picks)
            backup_pick_coverage = calculate_coverage(backup_picks)
            unlikely_pick_coverage = calculate_coverage(unlikely_picks)

            results.append({
                "Draw Number": predict_row["Draw Number"],
                "Top Picks Coverage %": top_pick_coverage,
                "Backup Picks Coverage %": backup_pick_coverage,
                "Unlikely Picks Coverage %": unlikely_pick_coverage,
            })

            if i % 100 == 0:
                logger.info(f"✅ Simulated up to draw {predict_row['Draw Number']}")

        except Exception as e:
            logger.error(f"❌ Error during walkforward step {i}: {e}", exc_info=True)

    result_df = pd.DataFrame(results)

    # Save to CSV
    output_path = "data/walkforward_simulation_results.csv"
    result_df.to_csv(output_path, index=False)
    logger.info(f"💾 Walkforward simulation results saved to {output_path}")

    return result_df
