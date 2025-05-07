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
