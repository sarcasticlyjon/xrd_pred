"""
Prediction v2.0

НОВОЕ:
- ✅ Batch prediction
- ✅ Progress tracking
"""

import numpy as np
from typing import List, Optional
from tqdm import tqdm


def make_predictions(model, X, verbose: bool = False):
    """Базовые предсказания"""
    return model.predict(X)


def batch_predict(
    model,
    X,
    batch_size: int = 1000,  # NEW!
    show_progress: bool = True  # NEW!
) -> np.ndarray:
    """
    НОВОЕ v2.0: Batch prediction для больших данных
    """
    n_samples = len(X)
    predictions = np.zeros(n_samples)
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    iterator = tqdm(range(n_batches), desc="Predicting") if show_progress else range(n_batches)
    
    for i in iterator:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_X = X[start_idx:end_idx]
        predictions[start_idx:end_idx] = model.predict(batch_X)
    
    return predictions


def predict_with_ensemble(
    models: List,
    X,
    method: str = 'mean'
) -> np.ndarray:
    """Ensemble prediction"""
    predictions = np.array([model.predict(X) for model in models])
    
    if method == 'mean':
        return predictions.mean(axis=0)
    elif method == 'median':
        return np.median(predictions, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")
