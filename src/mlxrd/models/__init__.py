"""Models Module v2.0 - УЛУЧШЕНО"""
__version__ = "2.0.0"

from .train import create_model, train_model, train_multiple_models
from .tune import (
    optimize_with_optuna, grid_search_cv, tune_hyperparameters,
    save_study, load_study
)
from .predict import make_predictions, predict_with_ensemble, batch_predict
from .metrics import compute_metrics, evaluate_model, cross_validate_model
from .registry import ModelRegistry, save_model, load_model

__all__ = [
    'create_model', 'train_model', 'train_multiple_models',
    'optimize_with_optuna', 'grid_search_cv', 'tune_hyperparameters',
    'save_study', 'load_study',
    'make_predictions', 'predict_with_ensemble', 'batch_predict',
    'compute_metrics', 'evaluate_model', 'cross_validate_model',
    'ModelRegistry', 'save_model', 'load_model'
]
