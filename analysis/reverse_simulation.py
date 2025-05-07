import random
from utils.match_rate_calculator import calculate_match_rate


def generate_random_set():
    """Generate a random valid lotto set (6 unique numbers between 1-40 + Powerball 1-10)."""
    main_numbers = sorted(random.sample(range(1, 41), 6))
    powerball = random.randint(1, 10)
    return main_numbers + [powerball]


def simulate_similar_sets(actual_draw, target_rate, tolerance=5, n_samples=5000):
    """
    Simulate random sets and find ones that match the target match rate within a tolerance.
    - actual_draw: the reference draw to compare against
    - target_rate: match rate in %
    - tolerance: acceptable ± difference
    - n_samples: how many random sets to simulate
    """
    matching_sets = []

    for _ in range(n_samples):
        candidate = generate_random_set()
        match = calculate_match_rate(candidate, actual_draw)

        if abs(match - target_rate) <= tolerance:
            matching_sets.append((candidate, match))

    # Sort best matches first
    matching_sets.sort(key=lambda x: -x[1])

    return matching_sets
