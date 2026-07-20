"""
Prediction module for heart disease prediction.

This module provides functions for making predictions using
the trained Random Forest classifier.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Any, List, Optional

from .utils import (
    numerical_features,
    categorical_features,
    get_models_path
)


def load_model(filepath: Optional[str] = None) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the model file. If None, uses default path.
        
    Returns:
        Loaded model
    """
    import joblib
    
    if filepath is None:
        filepath = str(get_models_path() / "heart_disease_rf.pkl")
    
    return joblib.load(filepath)


def create_input_dataframe(
    age: int,
    sex: int,
    cp: int,
    trestbps: int,
    chol: int,
    fbs: int,
    restecg: int,
    thalach: int,
    exang: int,
    oldpeak: float,
    slope: int,
    ca: int,
    thal: int
) -> pd.DataFrame:
    """
    Create a DataFrame with input features for prediction.
    
    Args:
        age: Age in years
        sex: Sex (1 = Male, 0 = Female)
        cp: Chest pain type (0-3)
        trestbps: Resting blood pressure (mm Hg)
        chol: Serum cholesterol (mg/dL)
        fbs: Fasting blood sugar > 120 mg/dL (1 = True, 0 = False)
        restecg: Resting ECG results (0-2)
        thalach: Maximum heart rate achieved
        exang: Exercise-induced angina (1 = Yes, 0 = No)
        oldpeak: ST depression induced by exercise
        slope: Slope of peak exercise ST segment (0-2)
        ca: Number of major vessels (0-3)
        thal: Thalassemia (1-3)
        
    Returns:
        DataFrame with single row of features
    """
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    return input_data


def predict_heart_disease(
    model: Any,
    input_data: Union[pd.DataFrame, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Make a prediction for heart disease.
    
    Args:
        model: Trained model
        input_data: Input features as DataFrame or dictionary
        
    Returns:
        Dictionary with prediction results:
        - prediction: 0 (no disease) or 1 (disease)
        - probability: Probability of heart disease
        - probability_no_disease: Probability of no heart disease
    """
    # Convert dict to DataFrame if needed
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    return {
        "prediction": int(prediction),
        "probability": float(probabilities[1]),
        "probability_no_disease": float(probabilities[0]),
        "has_disease": bool(prediction == 1)
    }


def predict_from_inputs(
    model: Any,
    age: int,
    sex: int,
    cp: int,
    trestbps: int,
    chol: int,
    fbs: int,
    restecg: int,
    thalach: int,
    exang: int,
    oldpeak: float,
    slope: int,
    ca: int,
    thal: int
) -> Dict[str, Any]:
    """
    Make a prediction from individual input values.
    
    Args:
        model: Trained model
        All other args: Feature values (see create_input_dataframe)
        
    Returns:
        Dictionary with prediction results
    """
    input_data = create_input_dataframe(
        age=age,
        sex=sex,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs,
        restecg=restecg,
        thalach=thalach,
        exang=exang,
        oldpeak=oldpeak,
        slope=slope,
        ca=ca,
        thal=thal
    )
    
    return predict_heart_disease(model, input_data)


def get_prediction_label(prediction: int) -> str:
    """
    Get human-readable prediction label.
    
    Args:
        prediction: 0 or 1
        
    Returns:
        Human-readable label
    """
    return "Heart Disease" if prediction == 1 else "No Heart Disease"


def get_risk_level(probability: float) -> str:
    """
    Get risk level based on probability.
    
    Args:
        probability: Probability of heart disease (0-1)
        
    Returns:
        Risk level string
    """
    if probability >= 0.7:
        return "High Risk"
    elif probability >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"


if __name__ == "__main__":
    # Example usage
    model = load_model()
    
    # Example prediction
    result = predict_from_inputs(
        model=model,
        age=50,
        sex=1,
        cp=2,
        trestbps=120,
        chol=200,
        fbs=0,
        restecg=1,
        thalach=150,
        exang=0,
        oldpeak=1.0,
        slope=2,
        ca=0,
        thal=3
    )
    
    print(f"Prediction: {get_prediction_label(result['prediction'])}")
    print(f"Risk Level: {get_risk_level(result['probability'])}")
    print(f"Probability of Heart Disease: {result['probability']:.2%}")