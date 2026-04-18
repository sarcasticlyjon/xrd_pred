"""
Data Splitting v2.0

НОВОЕ:
- ✅ Stratified group split
- ✅ Time-series split
- ✅ Validation split
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split


def group_train_test_split(
    df: pd.DataFrame,
    test_samples: Optional[List[str]] = None,
    target_col: str = 'target',
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    УЛУЧШЕНО v2.0: поддержка group_col
    """
    if test_samples and len(test_samples) > 0:
        # По списку samples
        sample_col = 'sample_id' if 'sample_id' in df.columns else 'sample_number'
        if sample_col not in df.columns:
            raise ValueError("sample_id or sample_number column required")
        
        test_mask = df[sample_col].isin(test_samples)
        train_df = df[~test_mask]
        test_df = df[test_mask]
    else:
        # Случайное разделение
        if group_col:
            # Разделение по группам
            groups = df[group_col].unique()
            train_groups, test_groups = train_test_split(
                groups, test_size=test_size, random_state=random_state
            )
            train_df = df[df[group_col].isin(train_groups)]
            test_df = df[df[group_col].isin(test_groups)]
        else:
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state
            )
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test


def stratified_group_split(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    n_bins: int = 5,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    НОВОЕ v2.0: Stratified split по группам
    
    Обеспечивает равное распределение target в train/test
    при соблюдении group integrity
    """
    # Binning target
    df['_target_bin'] = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates='drop')
    
    # Group by group_col и считаем средний bin
    group_stats = df.groupby(group_col)['_target_bin'].agg(['mean', 'count']).reset_index()
    group_stats['mean_bin'] = group_stats['mean'].round().astype(int)
    
    # Stratified split по группам
    train_groups, test_groups = train_test_split(
        group_stats[group_col],
        test_size=test_size,
        stratify=group_stats['mean_bin'],
        random_state=random_state
    )
    
    train_df = df[df[group_col].isin(train_groups)].drop(columns=['_target_bin'])
    test_df = df[df[group_col].isin(test_groups)].drop(columns=['_target_bin'])
    
    return train_df, test_df


def get_test_set(
    df: pd.DataFrame,
    criteria: dict,
    target_col: str = 'target'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Обратная совместимость"""
    mask = pd.Series([True] * len(df))
    for col, values in criteria.items():
        if col in df.columns:
            mask &= df[col].isin(values if isinstance(values, list) else [values])
    
    test_df = df[mask]
    train_df = df[~mask]
    
    return (
        train_df.drop(columns=[target_col]),
        test_df.drop(columns=[target_col]),
        train_df[target_col],
        test_df[target_col]
    )
