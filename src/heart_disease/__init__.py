"""
Heart Disease Prediction Package

This package provides modules for heart disease prediction using
a trained Random Forest classifier.
"""

from .predict import predict_heart_disease, load_model, predict_from_inputs, get_prediction_label, get_risk_level
from .utils import numerical_features, categorical_features, CATEGORICAL_LABELS, get_feature_info
from .preprocessing import create_preprocessor, load_data, preprocess_data, get_feature_names
from .train import train_model, evaluate_model, save_model, load_model as load_trained_model

__version__ = "0.1.0"
__all__ = [
    "predict_heart_disease",
    "load_model",
    "predict_from_inputs",
    "get_prediction_label",
    "get_risk_level",
    "numerical_features",
    "categorical_features",
    "CATEGORICAL_LABELS",
    "get_feature_info",
    "create_preprocessor",
    "load_data",
    "preprocess_data",
    "get_feature_names",
    "train_model",
    "evaluate_model",
    "save_model",
]