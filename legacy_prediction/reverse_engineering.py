# utils/reverse_engineering.py

import random
from analysis.match_rate_calculator import calculate_match_rate


def generate_random_set():
    """Generate a random valid lotto set (6 unique numbers between 1-40 + Powerball 1-10)."""
    main_numbers = sorted(random.sample(range(1, 41), 6))
    powerball = random.randint(1, 10)
    return main_numbers + [powerball]


def simulate_similar_sets(actual_draw, target_rate, tolerance=5, n_samples=5000):
    """
    Simulate random sets and find ones that match the target match rate within a tolerance.
    - target_rate: match rate in %
    - tolerance: acceptable difference ±
    - n_samples: how many random sets to simulate
    """
    matching_sets = []

    for _ in range(n_samples):
        candidate = generate_random_set()
        match = calculate_match_rate(candidate, actual_draw)

        if abs(match - target_rate) <= tolerance:
            matching_sets.append((candidate, match))

    # Sort best first
    matching_sets = sorted(matching_sets, key=lambda x: -x[1])

    return matching_sets


def predict_with_reverse(draw_df, exclusions=None):  # ✅ Added optional exclusions parameter
    """
    Wrapper to integrate with Level 1 system.
    Simulates random sets that partially match the latest draw.
    """
    if draw_df.empty:
        return []

    latest_draw = draw_df.iloc[-1]
    actual_draw = [int(latest_draw[str(i)]) for i in range(1, 7)] + [int(latest_draw["Power Ball"])]

    simulated = simulate_similar_sets(actual_draw, target_rate=50, tolerance=10, n_samples=5000)
    top_candidates = [s for s, score in simulated[:10]]

    return top_candidates
