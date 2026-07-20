"""
Training module for heart disease prediction.

This module contains functions for training and evaluating
the Random Forest classifier.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import joblib

from .preprocessing import create_preprocessor
from .utils import get_models_path, numerical_features, categorical_features


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: Optional[int] = None
) -> Any:
    """
    Train a Random Forest classifier with preprocessing pipeline.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        
    Returns:
        Trained Pipeline object
    """
    # Create preprocessor
    preprocessor = create_preprocessor()
    
    # Create classifier
    classifier = RandomForestClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    
    # Create pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred
    }


def save_model(model: Any, filepath: Optional[str] = None) -> str:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save the model. If None, uses default path.
        
    Returns:
        Path where the model was saved
    """
    if filepath is None:
        filepath = str(get_models_path() / "heart_disease_rf.pkl")
    
    joblib.dump(model, filepath)
    return filepath


def load_model(filepath: Optional[str] = None) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the model file. If None, uses default path.
        
    Returns:
        Loaded model
    """
    if filepath is None:
        filepath = str(get_models_path() / "heart_disease_rf.pkl")
    
    return joblib.load(filepath)


def run_training_pipeline(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run the complete training pipeline.
    
    Args:
        data: Input DataFrame with target column
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained model, evaluation results)
    """
    # Prepare data
    X = data.drop("target", axis=1)
    y = data["target"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model = train_model(X_train, y_train, random_state=random_state)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    return model, results


if __name__ == "__main__":
    # Load data
    from .preprocessing import load_data
    
    data = load_data()
    print(f"Dataset shape: {data.shape}")
    
    # Run training pipeline
    model, results = run_training_pipeline(data)
    
    # Print results
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print(f"\nClassification Report:\n{classification_report(data['target'], results['y_pred'])}")
    
    # Save model
    save_model(model)
    print("\nModel saved to models/heart_disease_rf.pkl")