from predictors.level_predictor import run_level_1
from collections import Counter
import itertools
import pandas as pd

# Utility to merge predictions from two methods via frequency vote
def merge_two_predictions(pred1, pred2):
    # Exclude Powerball
    merged_main = list(Counter(pred1["main"] + pred2["main"]).most_common(6))
    top_main = sorted(set([num for num, _ in merged_main]))[:6]

    # For Powerball, just pick the most common
    powerball = pred1["powerball"] if pred1["powerball"] == pred2["powerball"] else pred1["powerball"]
    return {"main": top_main, "powerball": powerball}


def run_level_2(draw_data):
    """
    Run all pairwise combinations of Level 1 predictors
    Returns a dictionary of results like:
    {
        'ml+symbolic': {main: [...], powerball: ...},
        'ml+walkforward': {...},
        ...
    }
    """
    level1_outputs = run_level_1(draw_data)
    method_names = list(level1_outputs.keys())

    results = {}
    for method1, method2 in itertools.combinations(method_names, 2):
        pred1 = level1_outputs[method1]
        pred2 = level1_outputs[method2]
        merged = merge_two_predictions(pred1, pred2)
        key = f"{method1}+{method2}"
        results[key] = merged

    return results

