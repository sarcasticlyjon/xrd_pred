"""
Correlation v2.0

НОВОЕ:
- ✅ Оптимизированный mutual_info (параллельный)
- ✅ Partial correlation
- ✅ Улучшенная визуализация
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor


def correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """Корреляционная матрица"""
    return df.corr(method=method)


def _mutual_info_pair(args):
    """Helper для параллельного вычисления MI"""
    from sklearn.feature_selection import mutual_info_regression
    X, y = args
    return mutual_info_regression(X.reshape(-1, 1), y, random_state=42)[0]


def mutual_information_matrix(
    df: pd.DataFrame,
    n_jobs: int = 1  # NEW! Параллелизация
) -> pd.DataFrame:
    """
    УЛУЧШЕНО v2.0: Параллельное вычисление
    """
    from sklearn.feature_selection import mutual_info_regression
    
    columns = df.columns
    n_features = len(columns)
    mi_matrix = np.zeros((n_features, n_features))
    
    if n_jobs > 1:
        # NEW: Параллельное вычисление
        tasks = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                tasks.append((df[columns[i]].values, df[columns[j]].values))
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(_mutual_info_pair, tasks))
        
        # Заполнение матрицы
        idx = 0
        for i in range(n_features):
            mi_matrix[i, i] = 1.0
            for j in range(i + 1, n_features):
                mi_matrix[i, j] = mi_matrix[j, i] = results[idx]
                idx += 1
    else:
        # Последовательно
        for i in range(n_features):
            mi_matrix[i, i] = 1.0
            for j in range(i + 1, n_features):
                mi = mutual_info_regression(
                    df[columns[i]].values.reshape(-1, 1),
                    df[columns[j]].values,
                    random_state=42
                )[0]
                mi_matrix[i, j] = mi_matrix[j, i] = mi
    
    return pd.DataFrame(mi_matrix, index=columns, columns=columns)


def partial_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    НОВОЕ v2.0: Partial correlation
    """
    from scipy.stats import pearsonr
    from scipy.linalg import pinv
    
    corr = df.corr().values
    precision = pinv(corr)
    
    # Partial correlation from precision matrix
    partial_corr = np.zeros_like(precision)
    for i in range(len(precision)):
        for j in range(len(precision)):
            if i == j:
                partial_corr[i, j] = 1.0
            else:
                partial_corr[i, j] = -precision[i, j] / np.sqrt(precision[i, i] * precision[j, j])
    
    return pd.DataFrame(partial_corr, index=df.columns, columns=df.columns)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = 'pearson',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    annot: bool = False  # NEW! Опция аннотаций
):
    """
    УЛУЧШЕНО v2.0: лучшее форматирование
    """
    if method == 'partial':
        corr = partial_correlation(df)  # NEW!
    else:
        corr = correlation_matrix(df, method=method)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr,
        annot=annot,
        fmt='.2f' if annot else None,
        cmap='coolwarm',
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'}
    )
    
    ax.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def feature_target_correlation(
    df: pd.DataFrame,
    target_col: str,
    method: str = 'all',
    top_n: int = 20,
    plot: bool = True
) -> pd.DataFrame:
    """Feature-target correlation"""
    from scipy.stats import spearmanr
    
    results = {'feature': [], 'pearson': [], 'spearman': []}
    
    features = [col for col in df.columns if col != target_col]
    
    for feature in features:
        results['feature'].append(feature)
        
        # Pearson
        if method in ['all', 'pearson']:
            pearson = df[feature].corr(df[target_col])
            results['pearson'].append(pearson)
        
        # Spearman
        if method in ['all', 'spearman']:
            spearman, _ = spearmanr(df[feature], df[target_col])
            results['spearman'].append(spearman)
    
    result_df = pd.DataFrame(results)
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if 'pearson' in result_df.columns:
            top_features = result_df.nlargest(top_n, 'pearson', keep='all')
            ax.barh(top_features['feature'], top_features['pearson'], alpha=0.7, label='Pearson')
        
        ax.set_xlabel('Correlation')
        ax.set_title(f'Top {top_n} Features Correlation with {target_col}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
    
    return result_df
