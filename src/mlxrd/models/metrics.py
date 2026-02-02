"""Metrics v2.0"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score


def compute_metrics(y_true, y_pred):
    """Вычисление метрик"""
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }


def evaluate_model(model, X_train, y_train, X_test, y_test, verbose=True):
    """Оценка модели"""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)
    
    metrics = {
        'r2_train': train_metrics['r2'],
        'mae_train': train_metrics['mae'],
        'rmse_train': train_metrics['rmse'],
        'r2_test': test_metrics['r2'],
        'mae_test': test_metrics['mae'],
        'rmse_test': test_metrics['rmse'],
    }
    
    if verbose:
        print(f"📊 Evaluation:")
        print(f"   Train R²: {metrics['r2_train']:.4f}")
        print(f"   Test R²:  {metrics['r2_test']:.4f}")
    
    return metrics


def cross_validate_model(model, X, y, cv=5):
    """Cross-validation"""
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return {
        'cv_r2_mean': scores.mean(),
        'cv_r2_std': scores.std(),
        'cv_scores': scores
    }
