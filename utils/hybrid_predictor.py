# utils/hybrid_predictor.py

import numpy as np
from typing import List, Tuple, Dict
import logging

from utils.custom_rules import (
    is_cold_number,
    number_occurrence_map,
    build_historical_full_sets,
    is_full_set_duplicate,
    is_sequential_set,
)

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]

def filter_and_rank_sets(
    generated_sets: List[List[int]],
    draw_history,
    ticket_excludes,
    top_n: int = 10
) -> Tuple[List[List[int]], Dict[str, int]]:
    """
    Apply statistical filtering to generated number sets.
    """
    filtered_sets = []
    removal_stats = {
        "duplicates": 0,
        "ticket_duplicates": 0,
        "cold_number_violation": 0,
        "full_set_duplicate": 0,
        "sequential_set": 0
    }

    freq_map = number_occurrence_map(draw_history)
    historical_full_sets = build_historical_full_sets(draw_history)

    for s in generated_sets:
        # Exclude if exact same as active ticket
        if any(set(s[:6]) == ticket for ticket in ticket_excludes):
            removal_stats["ticket_duplicates"] += 1
            continue

        # Exclude if any number is too cold (appeared very few times)
        if any(is_cold_number(num, freq_map) for num in s[:6]):
            removal_stats["cold_number_violation"] += 1
            continue

        # Exclude if full set already appeared in history
        if is_full_set_duplicate(s, historical_full_sets):
            removal_stats["full_set_duplicate"] += 1
            continue

        # Exclude fully sequential number sets
        if is_sequential_set(s):
            removal_stats["sequential_set"] += 1
            continue

        filtered_sets.append(s)

    # Sort by PowerBall ascending just to standardize
    filtered_sets = sorted(filtered_sets, key=lambda x: x[6])

    return filtered_sets[:top_n], removal_stats

def generate_unlikely_sets(draw_df, top_n=10):
    """
    Generate sets with least frequent numbers (unlikely to appear).
    """
    try:
        all_numbers = draw_df[NUMBER_COLUMNS].values.ravel()
        number_counts = dict(zip(*np.unique(all_numbers, return_counts=True)))
        sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1])  # sort by frequency

        unlikely_sets = []
        for i in range(0, len(sorted_numbers) - 6, 6):
            group = [num for num, _ in sorted_numbers[i:i+6]]
            if len(group) == 6:
                group_sorted = sorted(group)
                unlikely_sets.append(group_sorted + [np.random.randint(1, 11)])  # Random Powerball

            if len(unlikely_sets) >= top_n:
                break

        return unlikely_sets

    except Exception as e:
        logging.error(f"❌ Failed to generate unlikely sets: {e}", exc_info=True)
        return []

def generate_partial_match_sets(base_sets: List[List[int]], top_n=10):
    """
    Generate new sets that partially match old sets by 3 numbers (50% match).
    """
    try:
        partial_sets = []
        for base in base_sets:
            new_set = set(base[:6])  # Only main numbers
            # Randomly change half of them
            change_count = 3

            changed = list(new_set)[:]
            for _ in range(change_count):
                changed.pop(np.random.randint(0, len(changed)))

            while len(changed) < 6:
                new_num = np.random.randint(1, 41)
                if new_num not in changed:
                    changed.append(new_num)

            changed_sorted = sorted(changed)
            new_powerball = np.random.randint(1, 11)

            partial_sets.append(changed_sorted + [new_powerball])

            if len(partial_sets) >= top_n:
                break

        return partial_sets

    except Exception as e:
        logging.error(f"❌ Failed to generate partial match sets: {e}", exc_info=True)
        return []
