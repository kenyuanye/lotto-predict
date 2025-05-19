import traceback
import joblib
import os
import pandas as pd

# Import all legacy prediction functions
from legacy_prediction.custom_rules_predictor import predict_with_custom_rules
from legacy_prediction.hybrid_predictor import predict_with_hybrid
from legacy_prediction.multi_level_predictor import predict_with_multi_level  # legacy wrapper
from legacy_prediction.reverse_engineering import predict_with_reverse as predict_with_reverse_engineering
from legacy_prediction.walkforward_predictor import predict_with_walkforward
from legacy_prediction.ensemble_predictor import predict_with_ensemble
from legacy_prediction.ml_predictor import predict_with_ml
from legacy_prediction.residual_predictor import predict_with_residuals
from legacy_prediction.symbolic_predictor import predict_with_symbolic

from analysis.accuracy_tracker import update_accuracy_for_method

# Helper to load models (for residual)
def load_trained_models():
    from trainers.trainer import load_models
    return load_models()

# Map method names to callables (residual handled separately)
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

def run_level_1(draw_history_df):
    """
    Run all Level 1 methods individually.
    Returns:
        dict: { method_name: [top 10 prediction sets] }
    """
    results = {}

    for method_name, method_fn in LEVEL1_METHODS.items():
        try:
            predictions = method_fn(draw_history_df)
            if isinstance(predictions, list):
                results[method_name] = predictions[:10]
            elif isinstance(predictions, dict) and "predictions" in predictions:
                results[method_name] = predictions["predictions"][:10]
            else:
                results[method_name] = predictions
        except Exception as e:
            print(f"[Level 1] {method_name} failed: {e}")
            traceback.print_exc()

    # Residual method needs models passed in
    try:
        models = load_trained_models()
        predictions = predict_with_residuals(draw_history_df, models)
        results["residual"] = predictions[:10] if isinstance(predictions, list) else predictions
    except Exception as e:
        print(f"[Level 1] residual failed: {e}")
        traceback.print_exc()

    # Update accuracy tracking
    try:
        update_level_1_accuracies(draw_history_df, draw_history_df.iloc[-1])
    except Exception as e:
        print(f"[Level 1 Accuracy] Update failed: {e}")
        traceback.print_exc()

    return results  # ✅ Just the results dict

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