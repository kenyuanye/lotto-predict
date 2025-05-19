# utils/ensemble_predictor.py

import numpy as np

def average_predictions(prediction_sets):
    """Average multiple sets of predictions."""
    combined = np.array(prediction_sets)
    avg_set = np.round(np.mean(combined, axis=0)).astype(int)
    return avg_set.tolist()

def majority_vote(prediction_sets):
    """Pick numbers that appear most often across prediction sets."""
    vote_counter = {}

    for pset in prediction_sets:
        for num in pset:
            vote_counter[num] = vote_counter.get(num, 0) + 1

    # Sort by vote count, highest first
    sorted_nums = sorted(vote_counter.items(), key=lambda x: (-x[1], x[0]))
    
    # Pick top 6 numbers (main numbers) + 1 for Powerball
    main_numbers = [num for num, _ in sorted_nums[:6]]
    powerball_candidates = [num for num, _ in sorted_nums[6:] if num <= 10]
    powerball = powerball_candidates[0] if powerball_candidates else np.random.randint(1, 11)

    return sorted(main_numbers) + [powerball]

def ensemble_predict(base_predictions, method="average"):
    """
    Combine multiple sets of predictions.
    - base_predictions: List of prediction sets (each set = list of 7 numbers)
    - method: "average" or "vote"
    """
    if not base_predictions or len(base_predictions) < 2:
        raise ValueError("At least 2 prediction sets are required for ensemble.")

    main_sets = [pset[:6] for pset in base_predictions]
    powerballs = [pset[6] for pset in base_predictions]

    if method == "average":
        averaged_main = average_predictions(main_sets)
        averaged_pb = int(np.round(np.mean(powerballs)))
    elif method == "vote":
        return majority_vote(base_predictions)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    return sorted(averaged_main) + [averaged_pb]
    
def ensure_unique_and_valid(prediction_sets):
    """
    Ensure each prediction set contains 6 unique main numbers (1–40) and a valid Powerball (1–10).
    Filters out invalid sets.
    """
    valid_sets = []
    for s in prediction_sets:
        if not isinstance(s, (list, tuple)) or len(s) != 7:
            continue
        main = s[:6]
        pb = s[6]
        if len(set(main)) == 6 and all(1 <= n <= 40 for n in main) and 1 <= pb <= 10:
            valid_sets.append(sorted(main) + [pb])
    return valid_sets

from legacy_prediction.ml_predictor import predict_with_ml
from legacy_prediction.symbolic_predictor import predict_with_symbolic
from legacy_prediction.walkforward_predictor import predict_with_walkforward

def predict_with_ensemble(draw_df):
    """
    Run ensemble prediction using ML, Symbolic, and Walkforward as base sets.
    Returns a list of 10 identical predictions for compatibility with Level 1.
    """
    base_sets = []
    try:
        base_sets.append(predict_with_ml(draw_df)[0])
    except Exception:
        pass
    try:
        base_sets.append(predict_with_symbolic(draw_df)[0])
    except Exception:
        pass
    try:
        base_sets.append(predict_with_walkforward(draw_df)[0])
    except Exception:
        pass

    if len(base_sets) < 2:
        return []

    ensemble_set = ensemble_predict(base_sets, method="vote")  # or "average"
    return [ensemble_set] * 10
