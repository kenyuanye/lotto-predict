import numpy as np
import pandas as pd
import logging

# --- Config ---
COLD_THRESHOLD = 5  # Appearances below this = cold number


# === Rule Utility Functions ===

def is_cold_number(num, freq_map):
    """Check if a number is considered 'cold' (low frequency)."""
    return freq_map.get(num, 0) <= COLD_THRESHOLD

def number_occurrence_map(draw_df):
    """Build a frequency map of all drawn numbers."""
    try:
        all_numbers = draw_df[["1", "2", "3", "4", "5", "6"]].values.ravel()
        return dict(zip(*np.unique(all_numbers, return_counts=True)))
    except Exception as e:
        logging.error(f"❌ Failed to build occurrence map: {e}", exc_info=True)
        return {}

def position1_bias_needed(draw_df, window=5):
    """
    Check if position 1 numbers are mostly <10 in last `window` draws.
    If so, we should bias toward >10 next draw.
    """
    try:
        last_values = draw_df.sort_values("Draw Number", ascending=False)["1"].head(window)
        return (last_values < 10).sum() >= window
    except Exception as e:
        logging.error(f"❌ Failed position 1 bias check: {e}", exc_info=True)
        return False

def powerball_repeat_bias(draw_df, window=3):
    """
    Check if PowerBall repeated in the last `window` draws.
    If not, it's more likely to repeat soon.
    """
    try:
        recent = draw_df.sort_values("Draw Number", ascending=False)["Power Ball"].head(window).values
        return len(set(recent)) < len(recent)
    except Exception as e:
        logging.error(f"❌ Failed PowerBall repeat bias check: {e}", exc_info=True)
        return False

def build_historical_full_sets(draw_df):
    """
    Build a set of historical full sets (main numbers + powerball) to exclude.
    """
    historical_sets = set()
    try:
        for _, row in draw_df.iterrows():
            main_nums = sorted([row[str(i)] for i in range(1, 7)])
            powerball = row["Power Ball"]
            full_set = tuple(main_nums + [powerball])
            historical_sets.add(full_set)
    except Exception as e:
        logging.warning(f"⚠️ Failed to parse draw record: {e}", exc_info=True)
    return historical_sets

def is_full_set_duplicate(candidate_set, historical_sets):
    """
    Return True if the candidate set matches any historical full draw (main + powerball).
    """
    try:
        main_nums = sorted(candidate_set[:6])
        powerball = candidate_set[6]
        return tuple(main_nums + [powerball]) in historical_sets
    except Exception as e:
        logging.warning(f"⚠️ Failed duplicate full set check: {e}", exc_info=True)
        return False

def is_sequential_set(candidate_set):
    """
    Return True if the main numbers are in complete sequence (e.g., 1,2,3,4,5,6).
    """
    try:
        main_nums = sorted(candidate_set[:6])
        return all((main_nums[i+1] - main_nums[i]) == 1 for i in range(5))
    except Exception as e:
        logging.warning(f"⚠️ Failed sequential check: {e}", exc_info=True)
        return False

def calculate_powerball_gaps(draw_df, window=30):
    """
    Return a dict showing how many draws ago each PowerBall number was drawn.
    More gaps = more bias toward those numbers.
    """
    try:
        recent_pbs = draw_df.sort_values("Draw Number")["Power Ball"].tolist()
        gaps = {i: None for i in range(1, 11)}

        for pb in range(1, 11):
            for j in range(1, len(recent_pbs)+1):
                if recent_pbs[-j] == pb:
                    gaps[pb] = j-1  # 0 if just drawn
                    break
            if gaps[pb] is None:
                gaps[pb] = len(recent_pbs)
        return gaps
    except Exception as e:
        logging.warning(f"⚠️ Failed to calculate PowerBall gaps: {e}", exc_info=True)
        return {}

def powerball_bias_weight(pb_gaps, draw_df, max_gap=30):
    """
    Returns a normalized weight array for all Powerball numbers based on gap size.
    Longer gaps = higher weight.
    """
    try:
        weights = []
        for pb in range(1, 11):
            gap = pb_gaps.get(pb, max_gap)
            weight = min(gap / max_gap, 1.0)
            weights.append(weight)

        weights = np.array(weights)
        norm_weights = weights / weights.sum()
        return norm_weights
    except Exception as e:
        logging.error(f"❌ Failed to calculate PowerBall bias weights: {e}", exc_info=True)
        return np.ones(10) / 10  # Fallback uniform distribution


# === Main Prediction Logic ===

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

        # Bias logic for Position 1
        if position1_bias:
            nums = list(range(11, 41))
        else:
            nums = list(range(1, 41))

        # Weight cold numbers
        weighted_pool = [n for n in nums for _ in range(2 if freq_map.get(n, 0) <= 5 else 1)]

        # Sample without replacement
        main_numbers = set(np.random.choice(weighted_pool, 6, replace=False))

        # Powerball prediction with weights
        pb_weights = powerball_bias_weight(pb_gaps, draw_df)
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

def predict_with_custom_rules(draw_df):
    """
    Wrapper for compatibility with level_predictor.
    Extracts 'predictions' list from the custom rule predictor output.
    """
    result = predict_custom_rules(draw_df)
    return result["predictions"] if isinstance(result, dict) and "predictions" in result else []


# === Compatibility Alias for Multi-Level Predictor ===

def apply_custom_rules(draw_df, base_set=None):
    """
    Alias to maintain compatibility with multi-level predictor logic.
    Ignores base_set and uses internal rules only.
    """
    return predict_with_custom_rules(draw_df)
