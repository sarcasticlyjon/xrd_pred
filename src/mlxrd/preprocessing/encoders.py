"""
Encoders v2.0

НОВОЕ:
- ✅ min_samples_leaf защита
- ✅ save/load encoder
- ✅ Multiple encoding methods
"""

import numpy as np
import pandas as pd
import pickle
from typing import Optional
from sklearn.preprocessing import OneHotEncoder as SklearnOHE


class TargetEncoder:
    """
    УЛУЧШЕНО v2.0:
    - min_samples_leaf (защита от overfitting)
    - Лучшее сглаживание
    - Save/load support
    """
    
    def __init__(self, smooth: int = 20, min_samples_leaf: int = 10):
        """
        Args:
            smooth: сглаживание
            min_samples_leaf: мин. samples для кодирования (NEW!)
        """
        self.smooth = smooth
        self.min_samples_leaf = min_samples_leaf  # NEW
        self.encodings_ = {}
        self.global_mean_ = None
        self.is_fitted_ = False
    
    def fit(self, X, y):
        """Обучение"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        X = X.ravel()
        self.global_mean_ = y.mean()
        
        for val in np.unique(X):
            mask = X == val
            n_samples = mask.sum()
            
            # NEW: Защита от малых групп
            if n_samples < self.min_samples_leaf:
                self.encodings_[val] = self.global_mean_
            else:
                val_mean = y[mask].mean()
                # Сглаживание
                self.encodings_[val] = (
                    (n_samples * val_mean + self.smooth * self.global_mean_) /
                    (n_samples + self.smooth)
                )
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Трансформация"""
        if not self.is_fitted_:
            raise ValueError("Call fit() first")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = X.ravel()
        return np.array([self.encodings_.get(val, self.global_mean_) for val in X])
    
    def fit_transform(self, X, y):
        """Fit + transform"""
        return self.fit(X, y).transform(X)
    
    def save(self, filepath: str):
        """NEW: Сохранение encoder"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str):
        """NEW: Загрузка encoder"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class OneHotEncoder:
    """Обёртка sklearn OHE"""
    def __init__(self):
        self.encoder = SklearnOHE(sparse_output=False, handle_unknown='ignore')
    
    def fit(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
        self.encoder.fit(X)
        return self
    
    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
        return self.encoder.transform(X)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


def save_encoder(encoder, filepath: str):
    """NEW: Universal save"""
    with open(filepath, 'wb') as f:
        pickle.dump(encoder, f)


def load_encoder(filepath: str):
    """NEW: Universal load"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
