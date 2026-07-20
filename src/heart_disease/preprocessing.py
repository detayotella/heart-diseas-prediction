"""
Preprocessing module for heart disease prediction.

This module contains functions for data preprocessing and pipeline
construction for the Random Forest classifier.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .utils import numerical_features, categorical_features


def create_preprocessor() -> ColumnTransformer:
    """
    Create a ColumnTransformer for preprocessing features.
    
    The preprocessor handles:
    - Numerical features: median imputation + standard scaling
    - Categorical features: most frequent imputation + one-hot encoding
    
    Returns:
        Configured ColumnTransformer
    """
    # Numerical pipeline
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("standard_scaler", StandardScaler())
        ]
    )
    
    # Categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )
    
    return preprocessor


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the heart disease dataset.
    
    Args:
        filepath: Path to the CSV file. If None, uses default path.
        
    Returns:
        DataFrame with the dataset
    """
    if filepath is None:
        from .utils import get_data_path
        filepath = str(get_data_path() / "heart_data_set.csv")
    
    data = pd.read_csv(filepath)
    return data


def preprocess_data(
    data: pd.DataFrame,
    target_column: str = "target"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split data into features and target.
    
    Args:
        data: Input DataFrame
        target_column: Name of the target column
        
    Returns:
        Tuple of (features, target)
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Get feature names after preprocessing.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        
    Returns:
        List of feature names
    """
    # Get numerical feature names
    num_features = preprocessor.named_transformers_['num'].named_steps['standard_scaler'].get_feature_names_out(numerical_features)
    
    # Get categorical feature names
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    
    return list(num_features) + list(cat_features)