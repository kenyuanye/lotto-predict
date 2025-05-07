import numpy as np
import pandas as pd
from utils.custom_rules import (
    number_occurrence_map,
    position1_bias_needed,
    powerball_repeat_bias,
    build_historical_full_sets,
    is_full_set_duplicate,
    is_sequential_set,
    calculate_powerball_gaps,
    powerball_bias_weight
)

def predict_custom_rules(draw_df, top_n=10):
    """
    Predict main numbers and Powerball using defined custom rules.
    Returns a list of sets with 6 main numbers + 1 Powerball.
    """
    results = []
    freq_map = number_occurrence_map(draw_df)
    historical_sets = build_historical_full_sets(draw_df)
    position1_bias = position1_bias_needed(draw_df)
    pb_repeat_bias = powerball_repeat_bias(draw_df)
    pb_gaps = calculate_powerball_gaps(draw_df)

    tries = 0
    while len(results) < top_n and tries < 1000:
        tries += 1
        main_numbers = set()

        # Apply position 1 bias logic
        if position1_bias:
            nums = list(range(11, 41))
        else:
            nums = list(range(1, 41))

        # Prioritize cold numbers (low frequency)
        weighted_pool = [n for n in nums for _ in range(2 if freq_map.get(n, 0) <= 5 else 1)]

        # Sample without replacement
        main_numbers = set(np.random.choice(weighted_pool, 6, replace=False))

        # Powerball prediction
        pb_weights = powerball_bias_weight(pb_gaps)
        powerball = int(np.random.choice(range(1, 11), p=pb_weights))

        candidate = sorted(main_numbers) + [powerball]

        if is_full_set_duplicate(candidate, historical_sets):
            continue
        if is_sequential_set(candidate):
            continue

        results.append(candidate)

    return {
        "predictions": results,
        "rule_meta": {
            "position1_bias": position1_bias,
            "pb_repeat_bias": pb_repeat_bias,
            "pb_gap_bias": pb_gaps
        }
    }

