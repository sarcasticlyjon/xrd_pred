"""Preprocessing Module v2.0 - УЛУЧШЕНО"""
__version__ = "2.0.0"

from .splits import group_train_test_split, get_test_set, stratified_group_split
from .encoders import TargetEncoder, OneHotEncoder, save_encoder, load_encoder
from .transforms import (
    normalize_intensity, RobustScaler, PowerTransformer,
    apply_gaussian_filter, baseline_correction, TransformPipeline
)
from .chemistry import parse_formula, add_element_columns, get_element_features

__all__ = [
    'group_train_test_split', 'get_test_set', 'stratified_group_split',
    'TargetEncoder', 'OneHotEncoder', 'save_encoder', 'load_encoder',
    'normalize_intensity', 'RobustScaler', 'PowerTransformer',
    'apply_gaussian_filter', 'baseline_correction', 'TransformPipeline',
    'parse_formula', 'add_element_columns', 'get_element_features'
]
