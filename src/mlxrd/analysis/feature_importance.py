"""
Feature Importance v2.0

НОВОЕ:
- ✅ SHAP integration
- ✅ Parallel permutation importance
- ✅ Confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import warnings


def compute_feature_importance(
    model,
    feature_names: List[str],
    method: str = 'built-in'
) -> pd.DataFrame:
    """Feature importance"""
    if method == 'built-in':
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            raise ValueError("Model has no feature_importances_ or coef_")
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def permutation_importance_parallel(
    model,
    X,
    y,
    feature_names: List[str],
    n_repeats: int = 10,
    n_jobs: int = 1,  # NEW!
    scoring: str = 'r2'
) -> pd.DataFrame:
    """
    НОВОЕ v2.0: Параллельная permutation importance
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score, mean_absolute_error
    
    # Scorer
    if scoring == 'r2':
        from sklearn.metrics import make_scorer
        scorer = make_scorer(r2_score)
    else:
        scorer = scoring
    
    # Permutation importance с параллелизацией
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        n_jobs=n_jobs,  # NEW!
        scoring=scorer,
        random_state=42
    )
    
    # Результаты с confidence intervals
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,  # NEW!
        'ci_lower': result.importances_mean - 1.96 * result.importances_std,  # NEW!
        'ci_upper': result.importances_mean + 1.96 * result.importances_std   # NEW!
    }).sort_values('importance_mean', ascending=False)
    
    return importance_df


def shap_importance(
    model,
    X,
    feature_names: List[str],
    plot: bool = True
) -> Optional[pd.DataFrame]:
    """
    НОВОЕ v2.0: SHAP feature importance
    """
    try:
        import shap
        
        # Tree explainer для tree models
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            # KernelExplainer для других
            explainer = shap.KernelExplainer(
                model.predict,
                shap.sample(X, min(100, len(X)))
            )
        
        shap_values = explainer.shap_values(X)
        
        # Mean absolute SHAP
        importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': importance
        }).sort_values('shap_importance', ascending=False)
        
        if plot:
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        
        return importance_df
    
    except ImportError:
        warnings.warn("SHAP not installed: pip install shap")
        return None


def plot_top_features(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Optional[str] = None,
    with_ci: bool = False  # NEW! С доверительными интервалами
):
    """
    УЛУЧШЕНО v2.0: поддержка CI
    """
    import matplotlib.pyplot as plt
    
    top = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(top))
    
    # Если есть CI
    if with_ci and 'ci_lower' in top.columns:
        xerr = [
            top['importance_mean'].values - top['ci_lower'].values,
            top['ci_upper'].values - top['importance_mean'].values
        ]
        ax.barh(y_pos, top['importance_mean'].values, xerr=xerr, alpha=0.7, capsize=5)
    else:
        importance_col = 'importance_mean' if 'importance_mean' in top.columns else 'importance'
        ax.barh(y_pos, top[importance_col].values, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top['feature'].values)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Features')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return fig
