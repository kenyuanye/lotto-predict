# utils/ensemble_predictor.py

import numpy as np
from collections import Counter
from legacy_prediction.ml_predictor import predict_with_ml
from legacy_prediction.symbolic_predictor import predict_with_symbolic
from legacy_prediction.walkforward_predictor import predict_with_walkforward


def average_predictions(prediction_sets):
    """Average multiple sets of predictions."""
    combined = np.array(prediction_sets)
    avg_set = np.round(np.mean(combined, axis=0)).astype(int)
    return avg_set.tolist()


def majority_vote(prediction_sets):
    """Pick numbers that appear most often across prediction sets."""
    vote_counter = {}

    for pset in prediction_sets:
        if not isinstance(pset, (list, tuple)) or len(pset) < 7:
            continue
        for num in pset[:6]:  # Only vote on main numbers
            vote_counter[num] = vote_counter.get(num, 0) + 1

    # Sort by vote count, highest first
    sorted_nums = sorted(vote_counter.items(), key=lambda x: (-x[1], x[0]))

    # Pick top 6 numbers (main numbers) + 1 for Powerball
    main_numbers = [num for num, _ in sorted_nums[:6]]
    powerball_candidates = [pset[6] for pset in prediction_sets if isinstance(pset, (list, tuple)) and len(pset) >= 7 and 1 <= pset[6] <= 10]
    powerball = Counter(powerball_candidates).most_common(1)[0][0] if powerball_candidates else np.random.randint(1, 11)

    return sorted(main_numbers) + [powerball]


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


def ensemble_predict(base_predictions, method="average"):
    """
    Combine multiple sets of predictions.
    - base_predictions: List of prediction sets (each set = list of 7 numbers)
    - method: "average" or "vote"
    """
    if not base_predictions or not isinstance(base_predictions, list):
        raise ValueError("Prediction list must be non-empty.")

    valid_sets = ensure_unique_and_valid(base_predictions)
    if len(valid_sets) < 2:
        raise ValueError("At least 2 valid prediction sets are required for ensemble.")

    if method == "average":
        main_sets = [s[:6] for s in valid_sets]
        powerballs = [s[6] for s in valid_sets]
        averaged_main = average_predictions(main_sets)
        averaged_pb = int(np.round(np.mean(powerballs)))
        averaged_pb = max(1, min(averaged_pb, 10))
        return sorted(set(averaged_main))[:6] + [averaged_pb]

    elif method == "vote":
        return majority_vote(valid_sets)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def predict_with_ensemble(draw_df):
    """
    Run ensemble prediction using ML, Symbolic, and Walkforward as base sets.
    Returns a list of 10 identical predictions for compatibility with Level 1.
    """
    base_sets = []
    try:
        base = predict_with_ml(draw_df)
        if base and isinstance(base[0], list):
            base_sets.append(base[0])
    except Exception as e:
        print(f"[ensemble] ML failed: {e}")

    try:
        base = predict_with_symbolic(draw_df)
        if base and isinstance(base[0], list):
            base_sets.append(base[0])
    except Exception as e:
        print(f"[ensemble] Symbolic failed: {e}")

    try:
        base = predict_with_walkforward(draw_df)
        if base and isinstance(base, list):
            base_sets.append(base)
    except Exception as e:
        print(f"[ensemble] Walkforward failed: {e}")

    if len(base_sets) < 2:
        print("[ensemble] Not enough valid base sets.")
        return []

    try:
        ensemble_set = ensemble_predict(base_sets, method="vote")  # or "average"
        return [ensemble_set] * 10
    except Exception as e:
        print(f"[ensemble] Ensemble failed: {e}")
        return []
