# === Combined Level Predictor ===

# --- ML Predictor ---
import numpy as np
import pandas as pd
import joblib
import os

def predict_with_ml(models_dir, features_df):
    predictions = []
    for i in range(1, 7):  # For each of the 6 main numbers
        model_path = os.path.join(models_dir, f"rf_model_pos{i}.pkl")
        model = joblib.load(model_path)
        pred = model.predict(features_df)
        predictions.append(pred[0])

    # Predict Powerball
    model_path_pb = os.path.join(models_dir, "rf_model_powerball.pkl")
    model_pb = joblib.load(model_path_pb)
    powerball_pred = model_pb.predict(features_df)

    return sorted(predictions), int(powerball_pred[0])


# --- Symbolic / Reward Learning Predictor ---
import random

def generate_random_set():
    main_numbers = sorted(random.sample(range(1, 41), 6))
    powerball = random.randint(1, 10)
    return main_numbers, powerball

def calculate_match(predicted, actual):
    return len(set(predicted) & set(actual))

def generate_formula_from_failures(failed_sets):
    # Example formula: count frequency of numbers and pick highest
    freq = {}
    for s in failed_sets:
        for num in s:
            freq[num] = freq.get(num, 0) + 1
    sorted_nums = sorted(freq.items(), key=lambda x: -x[1])
    return [num for num, _ in sorted_nums[:6]]

def invert_formula(formula):
    # For reward learning, invert top choices to explore others
    return [num for num in range(1, 41) if num not in formula][:6]

def apply_formula(formula, length=6):
    return sorted(formula[:length])

def reward_learning_predictor(past_failures):
    base_formula = generate_formula_from_failures(past_failures)
    alt_formula = invert_formula(base_formula)
    return apply_formula(alt_formula), random.randint(1, 10)

def predict_with_reward_learning(failure_history):
    predicted_numbers, powerball = reward_learning_predictor(failure_history)
    return predicted_numbers, powerball


# --- Level Orchestration and Accuracy ---
from datetime import datetime
import json

def load_trained_models(model_dir):
    models = {}
    for i in range(1, 7):
        path = os.path.join(model_dir, f"rf_model_pos{i}.pkl")
        if os.path.exists(path):
            models[f"pos{i}"] = joblib.load(path)
    pb_path = os.path.join(model_dir, "rf_model_powerball.pkl")
    if os.path.exists(pb_path):
        models["powerball"] = joblib.load(pb_path)
    return models

def run_level_1(methods, features_df, models_dir, failure_history=[]):
    results = {}
    if 'ml' in methods:
        results['ml'] = predict_with_ml(models_dir, features_df)
    if 'reward_learning' in methods:
        results['reward_learning'] = predict_with_reward_learning(failure_history)
    return results

def update_level_1_accuracies(log_file, predictions, actual_draw):
    timestamp = datetime.now().isoformat()
    entry = {
        "timestamp": timestamp,
        "actual": actual_draw,
        "predictions": predictions
    }
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(entry)
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)

def generate_accuracy_stub(methods):
    return {method: {"correct_main": 0, "correct_powerball": 0, "total": 0} for method in methods}

def run_prediction_levels(level_config, features_df, models_dir, failure_history, log_file):
    methods = level_config.get("methods", ["ml", "reward_learning"])
    predictions = run_level_1(methods, features_df, models_dir, failure_history)
    update_level_1_accuracies(log_file, predictions, actual_draw=[])  # Supply actual draw externally
    return predictions
