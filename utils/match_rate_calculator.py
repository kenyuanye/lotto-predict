# utils/match_rate_calculator.py

import pandas as pd

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]

def calculate_match_rate(predicted_set, actual_set):
    """Calculate the match rate between a predicted set and the actual draw."""
    if not predicted_set or not actual_set:
        return 0.0

    main_predicted = set(predicted_set[:6])
    predicted_pb = predicted_set[6] if len(predicted_set) > 6 else None

    main_actual = set(actual_set[:6])
    actual_pb = actual_set[6] if len(actual_set) > 6 else None

    main_matches = len(main_predicted.intersection(main_actual))
    pb_match = (predicted_pb == actual_pb)

    # Match rate is based on 7 numbers (6 main + 1 Powerball)
    total_parts = 7
    matched_parts = main_matches + (1 if pb_match else 0)

    return round(matched_parts / total_parts * 100, 2)  # Return as percentage


def compare_all_predictions(predictions, actual_draw):
    """
    Compare all predictions against the actual draw.
    Return list of (prediction, match_rate %) sorted best to worst.
    """
    results = []
    for pred in predictions:
        if not isinstance(pred, (list, tuple)) or len(pred) < 7:
            continue  # Skip invalid
        match_rate = calculate_match_rate(pred, actual_draw)
        results.append((pred, match_rate))

    results = sorted(results, key=lambda x: -x[1])  # Highest match rate first
    return results
