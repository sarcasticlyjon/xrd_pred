"""
Hyperparameter Tuning v2.0

НОВОЕ:
- ✅ Custom param ranges для Optuna
- ✅ Save/load Optuna study
- ✅ Time budget
- ✅ Auto-selection метода
"""

import numpy as np
import pickle
from typing import Dict, Optional, Any
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import logging
import time

logger = logging.getLogger(__name__)


def optimize_with_optuna(
    X_train,
    y_train,
    model_type: str = 'xgboost',
    n_trials: int = 100,
    param_ranges: Optional[Dict] = None,  # NEW!
    cv: int = 5,
    time_budget: Optional[int] = None,  # NEW! в секундах
    save_study_path: Optional[str] = None,  # NEW!
    verbose: bool = True
) -> Dict:
    """
    УЛУЧШЕНО v2.0:
    - Custom param ranges
    - Time budget
    - Save study
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Optuna not installed: pip install optuna")
    
    from .train import create_model
    
    # Дефолтные ranges
    default_ranges = {
        'xgboost': {
            'n_estimators': (50, 300),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3, 'log'),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
        },
        'lightgbm': {
            'n_estimators': (50, 300),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3, 'log'),
            'subsample': (0.6, 1.0),
            'feature_fraction': (0.6, 1.0),
        },
        'random_forest': {
            'n_estimators': (50, 300),
            'max_depth': (5, 20),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 5),
        }
    }
    
    # NEW: Используем custom или дефолтные
    ranges = param_ranges or default_ranges.get(model_type, {})
    
    if verbose:
        print(f"🔍 Optuna optimization: {model_type}")
        if param_ranges:
            print(f"   Using custom param ranges")
        print(f"   Trials: {n_trials}")
        if time_budget:
            print(f"   Time budget: {time_budget}s")
    
    start_time = time.time()
    
    def objective(trial):
        # NEW: Динамическое создание параметров из ranges
        params = {}
        for param_name, param_range in ranges.items():
            if len(param_range) == 2:
                min_val, max_val = param_range
                if isinstance(min_val, int):
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                else:
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            elif len(param_range) == 3:
                min_val, max_val, scale = param_range
                params[param_name] = trial.suggest_float(
                    param_name, min_val, max_val,
                    log=(scale == 'log')
                )
        
        model = create_model(model_type, **params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        
        # NEW: Time budget check
        if time_budget and (time.time() - start_time) > time_budget:
            raise optuna.exceptions.OptunaError("Time budget exceeded")
        
        return scores.mean()
    
    # Создаём study
    study = optuna.create_study(direction='maximize')
    
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    except optuna.exceptions.OptunaError as e:
        if "Time budget" in str(e):
            if verbose:
                print(f"⏱️  Time budget reached: {len(study.trials)} trials completed")
    
    best_params = study.best_params
    
    # NEW: Save study
    if save_study_path:
        save_study(study, save_study_path)
        if verbose:
            print(f"💾 Study saved: {save_study_path}")
    
    if verbose:
        print(f"✅ Best score: {study.best_value:.4f}")
        print(f"   Best params: {best_params}")
    
    return best_params


def save_study(study, filepath: str):
    """NEW: Сохранение Optuna study"""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(study, f)


def load_study(filepath: str):
    """NEW: Загрузка Optuna study"""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def grid_search_cv(X_train, y_train, model_type: str = 'xgboost', param_grid: Optional[Dict] = None, cv: int = 5):
    """Grid Search"""
    from .train import create_model
    
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    
    model = create_model(model_type)
    grid = GridSearchCV(model, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    return grid.best_params_


def tune_hyperparameters(
    X_train,
    y_train,
    model_type: str = 'xgboost',
    method: str = 'auto',  # NEW! auto-selection
    param_grid: Optional[Dict] = None,
    param_ranges: Optional[Dict] = None,  # NEW!
    n_trials: int = 100,
    cv: int = 5,
    time_budget: Optional[int] = None,  # NEW!
    **kwargs
) -> Dict:
    """
    УЛУЧШЕНО v2.0:
    - method='auto' для автовыбора
    - param_ranges для optuna
    - time_budget
    """
    # NEW: Auto-select method
    if method == 'auto':
        n_samples = len(X_train)
        if n_samples < 1000:
            method = 'grid'
        elif n_samples < 10000:
            method = 'random'
        else:
            method = 'optuna'
        print(f"🤖 Auto-selected method: {method}")
    
    if method == 'optuna':
        return optimize_with_optuna(
            X_train, y_train,
            model_type=model_type,
            n_trials=n_trials,
            param_ranges=param_ranges,  # NEW!
            cv=cv,
            time_budget=time_budget,  # NEW!
            **kwargs
        )
    elif method == 'grid':
        return grid_search_cv(X_train, y_train, model_type, param_grid, cv)
    else:
        raise ValueError(f"Unknown method: {method}")
