import json
import random
import pandas as pd
from collections import defaultdict
from typing import List, Dict

# === Constants ===
NUMBERS_RANGE = list(range(1, 41))
POWERBALL_RANGE = list(range(1, 11))
NUM_MAIN_NUMBERS = 6
CSV_LOG_FILE = "data/reward_learning_log.csv"

# === Helper Functions ===
def generate_random_set():
    main_numbers = sorted(random.sample(NUMBERS_RANGE, NUM_MAIN_NUMBERS))
    powerball = random.choice(POWERBALL_RANGE)
    return main_numbers + [powerball]

def calculate_match(predicted, actual):
    main_match = len(set(predicted[:6]) & set(actual[:6]))
    powerball_match = predicted[6] == actual[6]
    return main_match, powerball_match

def generate_formula_from_failures(failed_sets: List[List[int]]) -> Dict:
    pos_values = defaultdict(list)
    for fs in failed_sets:
        for pos in range(6):
            pos_values[f"pos_{pos+1}"].append(fs[pos])

    formula = {}
    for pos, values in pos_values.items():
        avg = sum(values) / len(values)
        formula[pos] = {
            "prefer_greater_than": round(avg),
            "avoid_values": list(set(v for v in values if values.count(v) > 1))
        }
    return formula

def invert_formula(formula: Dict) -> Dict:
    inverted = {}
    for pos, rules in formula.items():
        inverted[pos] = {
            "prefer_less_than": rules.get("prefer_greater_than", 20),
            "prefer_values": rules.get("avoid_values", [])
        }
    return inverted

def apply_formula(formula: Dict) -> List[int]:
    numbers = []
    for i in range(6):
        rules = formula.get(f"pos_{i+1}", {})
        pool = [n for n in NUMBERS_RANGE]
        if "prefer_greater_than" in rules:
            pool = [n for n in pool if n > rules["prefer_greater_than"]]
        if "prefer_less_than" in rules:
            pool = [n for n in pool if n < rules["prefer_less_than"]]
        if "avoid_values" in rules:
            pool = [n for n in pool if n not in rules["avoid_values"]]
        if "prefer_values" in rules:
            pool = pool + rules["prefer_values"]  # bias
        numbers.append(random.choice(pool) if pool else random.randint(1, 40))

    powerball = random.choice(POWERBALL_RANGE)
    return sorted(numbers) + [powerball]

# === Main Reward Learning Predictor ===
def reward_learning_predictor(draw_history: List[List[int]]) -> None:
    formula_A, formula_B = None, None
    logs = []

    for idx, actual_draw in enumerate(draw_history):
        failed_sets = []
        attempts = 0
        matched = False

        while not matched:
            prediction = generate_random_set()
            failed_sets.append(prediction)
            attempts += 1
            main_match, powerball_match = calculate_match(prediction, actual_draw)
            if main_match == 6 and powerball_match:
                matched = True

        formula_A = generate_formula_from_failures(failed_sets)
        formula_B = invert_formula(formula_A)

        # Test formula_A and formula_B on next draw if available
        try:
            next_draw = draw_history[idx + 1]
            pred_A = apply_formula(formula_A)
            pred_B = apply_formula(formula_B)
            acc_A = calculate_match(pred_A, next_draw)
            acc_B = calculate_match(pred_B, next_draw)
        except IndexError:
            acc_A = acc_B = (0, False)

        logs.append({
            "DrawIndex": idx,
            "Attempts": attempts,
            "FormulaA": json.dumps(formula_A),
            "FormulaB": json.dumps(formula_B),
            "FormulaA_MainMatch": acc_A[0],
            "FormulaA_PBMatch": acc_A[1],
            "FormulaB_MainMatch": acc_B[0],
            "FormulaB_PBMatch": acc_B[1]
        })

    pd.DataFrame(logs).to_csv(CSV_LOG_FILE, index=False)
    print(f"Reward learning predictor completed. Log saved to {CSV_LOG_FILE}")

