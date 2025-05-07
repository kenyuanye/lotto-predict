# utils/custom_rules.py

import numpy as np
import logging

COLD_THRESHOLD = 5  # Appearances below this = cold number

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

def powerball_bias_weight(pb_number, draw_df, max_gap=30):
    """
    Calculate weight for a PowerBall number based on how many draws since it last appeared.
    Returns a float where larger gap = higher weight (towards 1.0 if very overdue).
    """
    try:
        sorted_draws = draw_df.sort_values("Draw Number", ascending=True)
        last_seen_idx = sorted_draws.index[sorted_draws["Power Ball"] == pb_number]
        
        if last_seen_idx.empty:
            gap = len(sorted_draws)
        else:
            gap = sorted_draws.index.max() - last_seen_idx.max()

        # Normalize: if gap >= max_gap, assign near 1.0
        weight = min(gap / max_gap, 1.0)

        return weight
    except Exception as e:
        logging.error(f"❌ Failed to calculate PowerBall bias weight for {pb_number}: {e}", exc_info=True)
        return 0.0
