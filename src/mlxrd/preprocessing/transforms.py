"""
Transforms v2.0

НОВОЕ:
- ✅ Сохранение параметров (для inverse)
- ✅ RobustScaler
- ✅ PowerTransformer
- ✅ TransformPipeline
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


class RobustScaler:
    """
    НОВОЕ v2.0: Robust scaling (устойчив к outliers)
    """
    def __init__(self, quantile_range=(25, 75)):
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None
    
    def fit(self, X):
        q_min, q_max = self.quantile_range
        self.center_ = np.percentile(X, 50, axis=0)
        self.scale_ = np.percentile(X, q_max, axis=0) - np.percentile(X, q_min, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self
    
    def transform(self, X):
        return (X - self.center_) / self.scale_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """NEW: обратная трансформация"""
        return X * self.scale_ + self.center_


class PowerTransformer:
    """НОВОЕ v2.0: Box-Cox / Yeo-Johnson"""
    def __init__(self, method='yeo-johnson'):
        self.method = method
        self.lambdas_ = None
    
    def fit(self, X):
        from sklearn.preprocessing import PowerTransformer as SkPT
        self.transformer_ = SkPT(method=self.method)
        self.transformer_.fit(X)
        return self
    
    def transform(self, X):
        return self.transformer_.transform(X)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


def normalize_intensity(
    intensities: np.ndarray,
    method: str = 'minmax',
    return_params: bool = False  # NEW
) -> np.ndarray:
    """
    УЛУЧШЕНО v2.0: return_params для inverse transform
    """
    params = {}
    
    if method == 'minmax':
        min_val = intensities.min()
        max_val = intensities.max()
        result = (intensities - min_val) / (max_val - min_val)
        params = {'min': min_val, 'max': max_val}
    
    elif method == 'zscore':
        mean_val = intensities.mean()
        std_val = intensities.std()
        result = (intensities - mean_val) / std_val
        params = {'mean': mean_val, 'std': std_val}
    
    elif method == 'robust':
        median = np.median(intensities)
        iqr = np.percentile(intensities, 75) - np.percentile(intensities, 25)
        result = (intensities - median) / iqr
        params = {'median': median, 'iqr': iqr}
    
    else:
        result = intensities
    
    if return_params:
        return result, params
    return result


def apply_gaussian_filter(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Сглаживание"""
    return gaussian_filter1d(data, sigma=sigma)


def baseline_correction(data: np.ndarray, method: str = 'linear') -> np.ndarray:
    """Коррекция baseline"""
    if method == 'linear':
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, deg=1)
        baseline = np.polyval(coeffs, x)
        return data - baseline
    return data


class TransformPipeline:
    """
    НОВОЕ v2.0: Чейнинг трансформаций
    """
    def __init__(self, steps: List[tuple]):
        self.steps = steps
    
    def fit(self, X):
        for name, transform in self.steps:
            if hasattr(transform, 'fit'):
                transform.fit(X)
        return self
    
    def transform(self, X):
        for name, transform in self.steps:
            X = transform.transform(X) if hasattr(transform, 'transform') else transform(X)
        return X
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
