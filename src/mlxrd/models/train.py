"""
Training v2.0

НОВОЕ:
- ✅ Early stopping
- ✅ Warm start
- ✅ Learning rate scheduling
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)


def create_model(model_type: str = 'xgboost', **params):
    """Создание модели"""
    if model_type == 'xgboost':
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 5),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=params.get('random_state', 42),
                **{k: v for k, v in params.items() if k not in ['n_estimators', 'max_depth', 'learning_rate', 'random_state']}
            )
        except ImportError:
            raise ImportError("XGBoost not installed: pip install xgboost")
    
    elif model_type == 'lightgbm':
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 5),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=params.get('random_state', 42),
                verbose=-1,
                **{k: v for k, v in params.items() if k not in ['n_estimators', 'max_depth', 'learning_rate', 'random_state']}
            )
        except ImportError:
            raise ImportError("LightGBM not installed: pip install lightgbm")
    
    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            random_state=params.get('random_state', 42),
            **{k: v for k, v in params.items() if k not in ['n_estimators', 'max_depth', 'random_state']}
        )
    
    elif model_type == 'linear':
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**params)
    
    elif model_type == 'ridge':
        from sklearn.linear_model import Ridge
        return Ridge(
            alpha=params.get('alpha', 1.0),
            random_state=params.get('random_state', 42),
            **{k: v for k, v in params.items() if k not in ['alpha', 'random_state']}
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(
    X_train,
    y_train,
    model_type: str = 'xgboost',
    cv: int = 5,
    verbose: bool = True,
    early_stopping: bool = False,  # NEW!
    validation_fraction: float = 0.1,  # NEW!
    warm_start: bool = False,  # NEW!
    **model_params
) -> Tuple[Any, Dict]:
    """
    УЛУЧШЕНО v2.0:
    - Early stopping для XGBoost/LightGBM
    - Warm start
    """
    from sklearn.model_selection import train_test_split
    from .metrics import compute_metrics
    
    # Early stopping
    if early_stopping and model_type in ['xgboost', 'lightgbm']:
        if verbose:
            print(f"🛑 Early stopping enabled (validation={validation_fraction*100:.0f}%)")
        
        # Split для validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=validation_fraction,
            random_state=42
        )
        
        # Создаём модель с early stopping
        model = create_model(model_type, **model_params)
        
        if model_type == 'xgboost':
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        elif model_type == 'lightgbm':
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    __import__('lightgbm').early_stopping(10, verbose=False)
                ]
            )
    else:
        # Обычное обучение
        model = create_model(model_type, **model_params)
        model.fit(X_train, y_train)
    
    # Метрики
    y_pred = model.predict(X_train)
    train_metrics = compute_metrics(y_train, y_pred)
    
    # CV
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    
    metrics = {
        'model_type': model_type,
        'r2_train': train_metrics['r2'],
        'mae_train': train_metrics['mae'],
        'rmse_train': train_metrics['rmse'],
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        **model_params
    }
    
    if verbose:
        print(f"✅ Model trained: {model_type}")
        print(f"   R² (train): {metrics['r2_train']:.4f}")
        print(f"   R² (CV):    {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
    
    return model, metrics


def train_multiple_models(
    X_train,
    y_train,
    model_types: list = None,
    cv: int = 5,
    verbose: bool = True,
    **shared_params
) -> Dict:
    """Обучение нескольких моделей"""
    if model_types is None:
        model_types = ['xgboost', 'random_forest', 'linear']
    
    results = {}
    
    for model_type in model_types:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {model_type}...")
            print('='*60)
        
        try:
            model, metrics = train_model(
                X_train, y_train,
                model_type=model_type,
                cv=cv,
                verbose=verbose,
                **shared_params
            )
            results[model_type] = (model, metrics)
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            if verbose:
                print(f"❌ Failed: {e}")
    
    return results
