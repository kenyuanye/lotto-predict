import itertools
from predictors.symbolic_predictor import predict_with_symbolic
from predictors.walkforward_predictor import predict_with_walkforward
from predictors.reverse_engineering import predict_with_reverse
from predictors.custom_rules_predictor import apply_custom_rules
from predictors.ensemble_predictor import ensure_unique_and_valid
from predictors.ml_core import predict_with_ml
from core.feature_engineering import build_features_for_prediction

# === LEVEL 1 ===
def level_1_all_methods(draw_history_df, exclusions=None):
    """Run all level 1 prediction methods individually."""
    features_df = build_features_for_prediction(draw_history_df)
    latest_features = features_df.tail(1)

    methods = {
        'ML': lambda df, ex: predict_with_ml("models/ml", latest_features),
        'Symbolic': predict_with_symbolic,
        'Walkforward': predict_with_walkforward,
        'Reverse': predict_with_reverse,
        'Custom': lambda df, ex: apply_custom_rules(df)
    }

    results = {}
    for name, func in methods.items():
        try:
            prediction = func(draw_history_df, exclusions)
            prediction = ensure_unique_and_valid(prediction)
            results[name] = prediction
        except Exception as e:
            results[name] = [f"❌ Error in {name}: {e}"]

    return results

# === LEVEL 2 ===
def level_2_combinations(level_1_results):
    """Generate predictions from combinations of two Level 1 methods."""
    combinations = list(itertools.combinations(level_1_results.keys(), 2))
    results = {}
    for combo in combinations:
        a, b = combo
        try:
            combined_set = combine_sets(level_1_results[a], level_1_results[b])
            combined_set = ensure_unique_and_valid(combined_set)
            results[f"{a}+{b}"] = combined_set
        except Exception as e:
            results[f"{a}+{b}"] = [f"❌ Combine error: {e}"]
    return results

# === LEVEL 3 ===
def level_3_with_custom(level_2_results, draw_history_df):
    """Apply custom rules to Level 2 results."""
    results = {}
    for name, base_set in level_2_results.items():
        try:
            custom_applied = apply_custom_rules(draw_history_df, base_set)
            custom_applied = ensure_unique_and_valid(custom_applied)
            results[f"{name}+Custom"] = custom_applied
        except Exception as e:
            results[f"{name}+Custom"] = [f"❌ Custom rule error: {e}"]
    return results

# === LEVEL 4 ===
def level_4_all_methods(level_1_results):
    """Combine all Level 1 methods together."""
    try:
        all_sets = list(level_1_results.values())
        combined_set = combine_multiple_sets(all_sets)
        combined_set = ensure_unique_and_valid(combined_set)
        return {"AllMethodsCombined": combined_set}
    except Exception as e:
        return {"AllMethodsCombined": [f"❌ Level 4 error: {e}"]}

# === WRAPPERS FOR STREAMLIT ===
def run_level_1(draw_history_df, exclusions=None):
    results = level_1_all_methods(draw_history_df, exclusions)
    return {f"Level1_{k}": v for k, v in results.items()}

def run_level_2(draw_history_df, exclusions=None):
    level1 = level_1_all_methods(draw_history_df, exclusions)
    level2 = level_2_combinations(level1)
    return {f"Level2_{k}": v for k, v in level2.items()}

def run_level_3(draw_history_df, exclusions=None):
    level1 = level_1_all_methods(draw_history_df, exclusions)
    level2 = level_2_combinations(level1)
    level3 = level_3_with_custom(level2, draw_history_df)
    return {f"Level3_{k}": v for k, v in level3.items()}

def run_level_4(draw_history_df, exclusions=None):
    level1 = level_1_all_methods(draw_history_df, exclusions)
    level4 = level_4_all_methods(level1)
    return {f"Level4_{k}": v for k, v in level4.items()}

# === FULL PIPELINE FOR BATCH TESTING ===
def run_all_levels(draw_history_df, exclusions=None):
    """Run predictions across all levels and return dictionary of sets."""
    results = {}
    results.update(run_level_1(draw_history_df, exclusions))
    results.update(run_level_2(draw_history_df, exclusions))
    results.update(run_level_3(draw_history_df, exclusions))
    results.update(run_level_4(draw_history_df, exclusions))
    return results

# === COMBINING LOGIC ===
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

    powerballs = [s[-1] for s in sets if isinstance(s, list) and len(s) > 6]
    powerball = max(set(powerballs), key=powerballs.count) if powerballs else 1
    return combined + [powerball]
