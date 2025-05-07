import itertools
from predictors.ml_predictor import predict_with_ml
from predictors.symbolic_predictor import predict_with_symbolic
from predictors.walkforward_predictor import predict_with_walkforward
from predictors.reverse_engineering import predict_with_reverse
from predictors.custom_rules_predictor import apply_custom_rules
from predictors.ensemble_predictor import ensure_unique_and_valid


def level_1_all_methods(draw_history_df, exclusions=None):
    """Run all level 1 prediction methods individually."""
    methods = {
        'ML': predict_with_ml,
        'Symbolic': predict_with_symbolic,
        'Walkforward': predict_with_walkforward,
        'Reverse': predict_with_reverse,
        'Custom': lambda df, ex: apply_custom_rules(df)
    }
    results = {}
    for name, func in methods.items():
        prediction = func(draw_history_df, exclusions)
        prediction = ensure_unique_and_valid(prediction)
        results[name] = prediction
    return results


def level_2_combinations(level_1_results):
    """Generate predictions from combinations of two Level 1 methods."""
    combinations = list(itertools.combinations(level_1_results.keys(), 2))
    results = {}
    for combo in combinations:
        a, b = combo
        combined_set = combine_sets(level_1_results[a], level_1_results[b])
        combined_set = ensure_unique_and_valid(combined_set)
        results[f"{a}+{b}"] = combined_set
    return results


def level_3_with_custom(level_2_results, draw_history_df):
    """Apply custom rules to Level 2 results."""
    results = {}
    for name, base_set in level_2_results.items():
        custom_applied = apply_custom_rules(draw_history_df, base_set)
        custom_applied = ensure_unique_and_valid(custom_applied)
        results[f"{name}+Custom"] = custom_applied
    return results


def level_4_all_methods(level_1_results):
    """Combine all Level 1 methods together."""
    all_sets = list(level_1_results.values())
    combined_set = combine_multiple_sets(all_sets)
    combined_set = ensure_unique_and_valid(combined_set)
    return {"AllMethodsCombined": combined_set}


# --- Utility functions ---

def combine_sets(set_a, set_b):
    """Combine two sets by intersecting or voting logic (simple merge for now)."""
    combined = list(set(set_a[:-1] + set_b[:-1]))[:6]  # Ensure 6 numbers
    powerball = set_a[-1] if set_a[-1] == set_b[-1] else set_a[-1]  # Fallback logic
    return combined + [powerball]


def combine_multiple_sets(sets):
    """Combine multiple sets (vote-based or merge logic)."""
    number_votes = {}
    for s in sets:
        for n in s[:-1]:
            number_votes[n] = number_votes.get(n, 0) + 1
    sorted_by_votes = sorted(number_votes.items(), key=lambda x: -x[1])
    combined = [num for num, _ in sorted_by_votes][:6]

    powerballs = [s[-1] for s in sets]
    powerball = max(set(powerballs), key=powerballs.count)
    return combined + [powerball]


def run_all_levels(draw_history_df, exclusions=None):
    """Run predictions across all levels and return dictionary of sets."""
    results = {}
    lvl1 = level_1_all_methods(draw_history_df, exclusions)
    results.update({f"Level1_{k}": v for k, v in lvl1.items()})

    lvl2 = level_2_combinations(lvl1)
    results.update({f"Level2_{k}": v for k, v in lvl2.items()})

    lvl3 = level_3_with_custom(lvl2, draw_history_df)
    results.update({f"Level3_{k}": v for k, v in lvl3.items()})

    lvl4 = level_4_all_methods(lvl1)
    results.update({f"Level4_{k}": v for k, v in lvl4.items()})

    return results
