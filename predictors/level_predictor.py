# predictors/level_predictor.py

import traceback
import joblib
import os
import pandas as pd
import numpy as np

# Legacy prediction functions
from legacy_prediction.custom_rules_predictor import predict_with_custom_rules
from legacy_prediction.hybrid_predictor import predict_with_hybrid
from legacy_prediction.multi_level_predictor import predict_with_multi_level
from legacy_prediction.reverse_engineering import predict_with_reverse as predict_with_reverse_engineering
from legacy_prediction.walkforward_predictor import predict_with_walkforward
from legacy_prediction.ensemble_predictor import predict_with_ensemble
from legacy_prediction.ml_predictor import predict_with_ml
from legacy_prediction.residual_predictor import predict_with_residuals
from legacy_prediction.symbolic_predictor import predict_with_symbolic

from analysis.accuracy_tracker import update_accuracy_for_method
from utils.feature_engineering import build_features_for_prediction

# Load trained models for residuals
def load_trained_models():
    from utils.model_utils import load_models
    return load_models()

# Mapping Level 1 prediction methods
LEVEL1_METHODS = {
    'ml': predict_with_ml,
    'symbolic': predict_with_symbolic,
    'walkforward': predict_with_walkforward,
    'custom_rules': predict_with_custom_rules,
    'hybrid': predict_with_hybrid,
    'ensemble': predict_with_ensemble,
    'reverse_engineering': predict_with_reverse_engineering,
    'multi_level': predict_with_multi_level
}

def predict_individual_numbers(draw_df, top_n=10):
    """Predict each number using trained models."""
    try:
        X_all = build_features_for_prediction(draw_df)
        X_latest = X_all[-1:]

        model_dir = "models"
        main_number_models = []
        for i in range(1, 7):
            path = os.path.join(model_dir, f"model_pos{i}.pkl")
            model = joblib.load(path)
            main_number_models.append(model)

        powerball_model = joblib.load(os.path.join(model_dir, "model_powerball.pkl"))

        sets = []
        for _ in range(top_n):
            nums = [int(round(model.predict(X_latest)[0])) for model in main_number_models]
            main_numbers = sorted(set(max(min(n, 59), 1) for n in nums))
            while len(main_numbers) < 6:
                candidate = np.random.randint(1, 60)
                if candidate not in main_numbers:
                    main_numbers.append(candidate)
            main_numbers = sorted(main_numbers[:6])

            powerball = int(powerball_model.predict(X_latest)[0])
            powerball = max(min(powerball, 20), 1)

            sets.append(main_numbers + [powerball])

        return sets

    except Exception as e:
        print(f"[Per-Number Prediction] Error: {e}")
        traceback.print_exc()
        return []

def run_level_1(draw_history_df):
    """Run all Level 1 methods and return valid prediction results."""
    results = {}

    for method_name, method_fn in LEVEL1_METHODS.items():
        try:
            predictions = method_fn(draw_history_df)

            if isinstance(predictions, list):
                cleaned = [p for p in predictions if isinstance(p, list) and len(p) == 7]
                if cleaned:
                    results[method_name] = cleaned[:10]
                else:
                    print(f"[{method_name}] Invalid predictions skipped.")
            elif isinstance(predictions, dict) and "predictions" in predictions:
                sets = predictions["predictions"]
                cleaned = [p for p in sets if isinstance(p, list) and len(p) == 7]
                if cleaned:
                    results[method_name] = cleaned[:10]
                else:
                    print(f"[{method_name}] No valid predictions found.")
            else:
                print(f"[{method_name}] Unexpected prediction format.")
        except Exception as e:
            print(f"[Level 1] {method_name} failed: {e}")
            traceback.print_exc()

    # Residual method (special handling)
    try:
        models = load_trained_models()
        prediction = predict_with_residuals(draw_history_df, models)

        if isinstance(prediction, list):
            if all(isinstance(p, list) and len(p) == 7 for p in prediction):
                results["residual"] = prediction[:10]
            elif len(prediction) == 7 and all(isinstance(n, int) for n in prediction):
                results["residual"] = [prediction]
            else:
                print(f"[residual] Invalid prediction structure: {prediction}")
        else:
            print(f"[residual] Unexpected return type: {type(prediction)}")

    except Exception as e:
        print(f"[Level 1] residual failed: {e}")
        traceback.print_exc()

    # Per-number fallback
    try:
        results["per_number_model"] = predict_individual_numbers(draw_history_df, top_n=10)
    except Exception as e:
        print(f"[Level 1] per_number_model failed: {e}")
        traceback.print_exc()

    # Accuracy update
    try:
        update_level_1_accuracies(draw_history_df, draw_history_df.iloc[-1])
    except Exception as e:
        print(f"[Level 1 Accuracy] Update failed: {e}")
        traceback.print_exc()

    return results

def update_level_1_accuracies(draw_history_df, last_draw_result):
    for method_name in list(LEVEL1_METHODS.keys()) + ["residual"]:
        try:
            update_accuracy_for_method(method_name, draw_history_df, last_draw_result)
        except Exception as e:
            print(f"[Accuracy Update] Failed for {method_name}: {e}")

def generate_accuracy_stub(results_dict):
    return pd.DataFrame({
        "Method": list(results_dict.keys()),
        "Accuracy %": [None] * len(results_dict)
    })

def run_prediction_levels(draw_df, exclude_list=None, ticket_excludes=None, top_n=10, apply_hybrid=True):
    return run_level_1(draw_df)
