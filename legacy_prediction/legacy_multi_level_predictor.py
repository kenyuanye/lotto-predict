import itertools
import pandas as pd
import numpy as np

# Corrected legacy imports
from legacy_prediction.ml_predictor import predict_with_ml
from legacy_prediction.symbolic_predictor import predict_with_symbolic
from legacy_prediction.walkforward_predictor import predict_with_walkforward
from legacy_prediction.reverse_engineering import predict_with_reverse
from legacy_prediction.custom_rules_predictor import predict_custom_rules as apply_custom_rules
from legacy_prediction.ensemble_predictor import ensure_unique_and_valid
from utils.feature_engineering import build_features_for_prediction

# Must match training time
EXPECTED_FEATURES = [
    "Sum", "OddCount", "EvenCount", "Range", "ConsecCount",
    "HighCount", "LowCount", "RepeatsFromLast",
    "HotnessScore", "LastSeenGapAvg"
]


def level_1_all_methods(draw_history_df, exclusions=None):
    """Run all level 1 prediction methods individually and apply feature alignment."""
    methods = {
        'ML': predict_with_ml,
        'Symbolic': predict_with_symbolic,
        'Walkforward': predict_with_walkforward,
        'Reverse': predict_with_reverse,
        'Custom': lambda df, ex: apply_custom_rules(df)
    }

    results = {}

    # Ensure features include DrawIndex and expected columns
    df = draw_history_df.copy().reset_index(drop=True)
    df["DrawIndex"] = df.index
    df["Draw Number"] = df.index

    for name, func in methods.items():
        try:
            predictions = func(df, exclusions)
            predictions = ensure_unique_and_valid(predictions)
            results[name] = predictions
        except Exception as e:
            print(f"[Level1] {name} failed: {e}")
            import traceback
            traceback.print_exc()

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


# --- Utilities ---

def combine_sets(set_a, set_b):
    """Merge two number sets (simple voting/override logic)."""
    combined = list(set(set_a[:-1] + set_b[:-1]))[:6]
    powerball = set_a[-1] if set_a[-1] == set_b[-1] else set_a[-1]
    return combined + [powerball]


def combine_multiple_sets(sets):
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
    """
    Run all 4 levels of prediction logic.
    Returns:
        dict of { label → prediction set }
    """
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


def predict_with_multi_level(draw_df):
    """Default method for Level 1 compatibility."""
    return run_all_levels(draw_df).get("Level4_AllMethodsCombined", [])


def run_prediction_levels(draw_df, exclude_list=None, ticket_excludes=None, top_n=10, apply_hybrid=True):
    try:
        results = run_all_levels(draw_df)
        from analysis.accuracy_tracker import build_accuracy_comparison_dataframe
        level_accuracy_df = build_accuracy_comparison_dataframe(results)
        return results, level_accuracy_df
    except Exception as e:
        import logging
        logging.error(f"❌ Failed in run_prediction_levels: {e}", exc_info=True)
        return {}, pd.DataFrame()
