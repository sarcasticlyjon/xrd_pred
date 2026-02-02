"""Validation v2.0 - Outlier detection"""
import pandas as pd, numpy as np
from typing import Optional, List, Dict

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None, 
                      min_rows: int = 1, check_duplicates: bool = False, 
                      check_missing: bool = False):
    """Validate DataFrame"""
    if df.empty:
        raise ValueError("DataFrame is empty")
    if len(df) < min_rows:
        raise ValueError(f"Too few rows: {len(df)} < {min_rows}")
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
    if check_duplicates and df.duplicated().any():
        raise ValueError("Duplicates found")
    if check_missing and df.isnull().any().any():
        raise ValueError("Missing values found")

def check_data_quality(df: pd.DataFrame, target_col: str, feature_cols: Optional[List[str]] = None) -> Dict:
    """Check data quality"""
    report = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'missing_total': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'target_stats': df[target_col].describe().to_dict() if target_col in df.columns else {}
    }
    return report

def validate_model_inputs(X, y, min_samples: int = 10):
    """Validate model inputs"""
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} != {len(y)}")
    if len(X) < min_samples:
        raise ValueError(f"Too few samples: {len(X)} < {min_samples}")
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains NaN or Inf")
    if np.any(~np.isfinite(y)):
        raise ValueError("y contains NaN or Inf")

def detect_outliers(data: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
    """
    NEW v2.0: Outlier detection
    
    Methods:
    - iqr: IQR method
    - zscore: Z-score method
    - isolation_forest: Isolation Forest
    """
    if method == 'iqr':
        Q1, Q3 = np.percentile(data, [25, 75])
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        return (data < lower) | (data > upper)
    
    elif method == 'zscore':
        z = np.abs((data - data.mean()) / data.std())
        return z > threshold
    
    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=0.1, random_state=42)
        pred = iso.fit_predict(data.reshape(-1, 1))
        return pred == -1
    
    else:
        raise ValueError(f"Unknown method: {method}")
