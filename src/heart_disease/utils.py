"""
Utility functions for heart disease prediction.
"""

from pathlib import Path
from typing import Tuple, List

# Feature definitions
numerical_features: List[str] = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

categorical_features: List[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

# Feature descriptions for documentation
FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = Male, 0 = Female)",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dL)",
    "fbs": "Fasting blood sugar > 120 mg/dL (1 = True, 0 = False)",
    "restecg": "Resting ECG results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = Yes, 0 = No)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment (0-2)",
    "ca": "Number of major vessels (0-3)",
    "thal": "Thalassemia (1-3)"
}

# Categorical feature mappings
CATEGORICAL_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        0: "Typical angina",
        1: "Atypical angina",
        2: "Non-anginal pain",
        3: "Asymptomatic"
    },
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {
        0: "Normal",
        1: "ST-T wave abnormality",
        2: "Left ventricular hypertrophy"
    },
    "exang": {0: "No", 1: "Yes"},
    "slope": {
        0: "Upsloping",
        1: "Flat",
        2: "Downsloping"
    },
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {
        1: "Normal",
        2: "Fixed defect",
        3: "Reversible defect"
    }
}


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_path() -> Path:
    """Get the data directory path."""
    return get_project_root() / "data" / "raw"


def get_models_path() -> Path:
    """Get the models directory path."""
    return get_project_root() / "models"


def get_feature_info() -> Tuple[List[str], List[str]]:
    """
    Get feature names and their types.
    
    Returns:
        Tuple of (numerical_features, categorical_features)
    """
    return numerical_features, categorical_features


def get_feature_description(feature: str) -> str:
    """
    Get the description for a feature.
    
    Args:
        feature: Feature name
        
    Returns:
        Feature description string
    """
    return FEATURE_DESCRIPTIONS.get(feature, "No description available")


def get_categorical_label(feature: str, value: int) -> str:
    """
    Get the human-readable label for a categorical feature value.
    
    Args:
        feature: Feature name
        value: Feature value
        
    Returns:
        Human-readable label string
    """
    labels = CATEGORICAL_LABELS.get(feature, {})
    return labels.get(value, str(value))