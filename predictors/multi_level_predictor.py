# legacy_prediction/multi_level_predictor.py

from predictors.level_predictor import run_level_1
from collections import Counter
import itertools
import pandas as pd
import numpy as np


def merge_two_predictions(pred1, pred2):
    """
    Merges two prediction dictionaries into one via frequency vote for main numbers.
    Keeps Powerball if they match, else uses pred1's.
    """
    if not pred1 or not pred2:
        return {"main": [], "powerball": None}

    try:
        merged_main = list(Counter(pred1["main"] + pred2["main"]).most_common())
        top_main = sorted(set([num for num, _ in merged_main]))[:6]

        while len(top_main) < 6:
            filler = np.random.randint(1, 41)
            if filler not in top_main:
                top_main.append(filler)

        top_main = sorted(top_main[:6])
        powerball = pred1["powerball"] if pred1["powerball"] == pred2["powerball"] else pred1["powerball"]
        powerball = max(1, min(powerball, 10))

        return {"main": top_main, "powerball": powerball}
    except Exception as e:
        print(f"[merge_two_predictions] Failed to merge: {e}")
        return {"main": [], "powerball": None}


def combine_sets(set_a, set_b):
    """
    Combine two sets of predictions (each should be a list of 7 numbers).
    Returns a 7-number list or [] if invalid.
    """
    if not isinstance(set_a, list) or not isinstance(set_b, list):
        print("[combine_sets] One or both sets are not lists.")
        return []
    if len(set_a) != 7 or len(set_b) != 7:
        print(f"[combine_sets] Invalid input lengths: {len(set_a)} / {len(set_b)}")
        return []

    try:
        main_a, pb_a = set_a[:6], set_a[6]
        main_b, pb_b = set_b[:6], set_b[6]

        merged_main = list(Counter(main_a + main_b).most_common())
        final_main = sorted(set([num for num, _ in merged_main]))[:6]

        while len(final_main) < 6:
            filler = np.random.randint(1, 41)
            if filler not in final_main:
                final_main.append(filler)
        final_main = sorted(final_main[:6])

        final_pb = pb_a if pb_a == pb_b else pb_a
        final_pb = max(1, min(final_pb, 10))

        return final_main + [final_pb]
    except Exception as e:
        print(f"[combine_sets] Failed to merge sets: {e}")
        return []


def run_level_2(draw_data):
    """
    Run all pairwise combinations of Level 1 predictors.
    Returns a dictionary like:
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
        pred1_raw = level1_outputs.get(method1)
        pred2_raw = level1_outputs.get(method2)

        if not isinstance(pred1_raw, list) or not isinstance(pred2_raw, list):
            continue
        if not pred1_raw or not pred2_raw:
            continue
        if not isinstance(pred1_raw[0], list) or not isinstance(pred2_raw[0], list):
            continue
        if len(pred1_raw[0]) != 7 or len(pred2_raw[0]) != 7:
            print(f"[run_level_2] Skipping {method1}+{method2} due to invalid prediction lengths.")
            continue

        pred1 = {"main": pred1_raw[0][:6], "powerball": pred1_raw[0][6]}
        pred2 = {"main": pred2_raw[0][:6], "powerball": pred2_raw[0][6]}

        merged = merge_two_predictions(pred1, pred2)
        results[f"{method1}+{method2}"] = merged

    return results
