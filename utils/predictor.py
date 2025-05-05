import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from typing import List, Dict
import logging

from utils.custom_rules import position1_bias_needed, powerball_repeat_bias

def get_excluded_numbers(exclude_input: str) -> List[int]:
    """Parse exclusion list from string input."""
    try:
        return [int(x.strip()) for x in exclude_input.split(',') if x.strip().isdigit()]
    except Exception as e:
        logging.warning(f"⚠️ Failed to parse excluded numbers: {e}", exc_info=True)
        return []

def predict_column(model: BaseEstimator, X: pd.DataFrame, col_name: str) -> List[int]:
    """Use trained model to predict a single number column."""
    try:
        logging.info(f"🧠 Predicting for column {col_name} with input shape {X.shape}")
        preds = model.predict(X)
        logging.info(f"✅ Raw predictions for {col_name}: {preds[:10]}")  # <--- Detailed raw output
        preds = np.round(preds).astype(int)
        preds = np.clip(preds, 1, 40 if col_name != "Power Ball" else 10)
        logging.info(f"✅ Processed predictions for {col_name} after rounding & clipping: {preds[:10]}")  # <--- After clip
        return preds.tolist()
    except Exception as e:
        logging.error(f"❌ Prediction failed for {col_name}: {e}", exc_info=True)
        return []

def generate_candidate_sets(predictions: Dict[str, List[int]], top_n: int = 10) -> List[List[int]]:
    """Generate unique candidate sets from column predictions."""
    all_combinations = []
    num_predictions = len(predictions.get("1", []))
    logging.info(f"🔢 Number of predictions per column: {num_predictions}")

    attempts = 0
    max_attempts = 5  # Try up to 5 times to generate enough valid sets

    while len(all_combinations) < top_n and attempts < max_attempts:
        for i in range(num_predictions):
            if len(all_combinations) >= top_n:
                break

            try:
                combo = [
                    predictions["1"][i],
                    predictions["2"][i],
                    predictions["3"][i],
                    predictions["4"][i],
                    predictions["5"][i],
                    predictions["6"][i],
                ]

                # Check for uniqueness
                if len(set(combo)) != 6:
                    logging.warning(f"⚠️ Skipping invalid set with duplicates: {combo}")
                    continue

                combo_sorted = sorted(combo)
                powerball = predictions["Power Ball"][i] if "Power Ball" in predictions else np.random.randint(1, 11)

                full_set = combo_sorted + [powerball]
                if full_set not in all_combinations:
                    all_combinations.append(full_set)

            except Exception as e:
                logging.warning(f"⚠️ Failed to generate combination at index {i}: {e}", exc_info=True)
                continue

        attempts += 1
        if len(all_combinations) < top_n:
            logging.info(f"♻️ Resampling attempt {attempts} — currently have {len(all_combinations)} valid sets.")

    if len(all_combinations) < top_n:
        logging.warning(f"⚠️ Only {len(all_combinations)} valid sets generated after {attempts} attempts.")

    logging.info(f"🧩 Generated {len(all_combinations)} candidate sets (before exclusions)")
    logging.info(f"📋 Sample candidate sets: {all_combinations[:5]}")
    return all_combinations


def predict_all(draw_df: pd.DataFrame, models: Dict[str, BaseEstimator], exclude_list: List[int], top_n: int = 10) -> Dict[str, List[List[int]]]:
    """Predict next lotto numbers using all models."""
    features = draw_df[["DrawIndex"]]

    logging.info(f"🛠️ Starting full predictions for models: {list(models.keys())}")
    logging.info(f"🔍 Features shape: {features.shape}")

    predictions = {}

    for col in ["1", "2", "3", "4", "5", "6", "Power Ball"]:
        if col in models:
            predictions[col] = predict_column(models[col], features, col)
        else:
            logging.warning(f"⚠️ Model for {col} not found. Skipping.")

    number_sets = generate_candidate_sets(predictions, top_n=100)
    logging.info(f"🛡️ Total number of sets before exclusions: {len(number_sets)}")

    # Apply manual exclusions
    filtered_sets = []
    for s in number_sets:
        if all(num not in exclude_list for num in s[:6]):
            filtered_sets.append(s)
        if len(filtered_sets) >= top_n:
            break

    logging.info(f"🛡️ Number of sets after manual exclusions: {len(filtered_sets)}")

    # Additional bias handling
    if position1_bias_needed(draw_df):
        logging.info("📈 Applying first number bias: favoring numbers >10 in position 1")
        filtered_sets = [s for s in filtered_sets if s[0] > 10] or filtered_sets

    # Powerball repeat bias
    biased_sets = []
    for s in filtered_sets:
        if powerball_repeat_bias(draw_df, s[6]):
            biased_sets.append(s)

    if biased_sets:
        logging.info(f"🎯 {len(biased_sets)} sets favored after Powerball repeat bias.")
        filtered_sets = biased_sets

    final_results = {
        "Top Picks": filtered_sets[:top_n],
        "Backup Picks": filtered_sets[top_n:top_n*2]
    }

    logging.info(f"📦 Final prediction result sets: { {k: len(v) for k, v in final_results.items()} }")
    return final_results
