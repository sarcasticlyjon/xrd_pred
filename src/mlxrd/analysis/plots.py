"""
Plots v2.0

НОВОЕ:
- ✅ Автоматический размер фигуры
- ✅ Interactive plots (Plotly)
- ✅ Улучшенное форматирование
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
import warnings


def auto_figsize(n_items: int, base_size: Tuple[int, int] = (10, 6)) -> Tuple[int, int]:
    """
    НОВОЕ v2.0: Автоматический размер фигуры
    """
    if n_items <= 10:
        return base_size
    elif n_items <= 20:
        return (base_size[0], base_size[1] * 1.2)
    elif n_items <= 30:
        return (base_size[0], base_size[1] * 1.5)
    else:
        return (base_size[0], base_size[1] * 2)


def plot_predictions(
    y_true,
    y_pred,
    model_name: str = 'Model',
    save_path: Optional[str] = None,
    interactive: bool = False,  # NEW!
    figsize: Optional[Tuple[int, int]] = None
):
    """
    УЛУЧШЕНО v2.0:
    - Interactive режим (Plotly)
    - Автоматический figsize
    """
    from sklearn.metrics import r2_score, mean_absolute_error
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # NEW: Interactive plot
    if interactive:
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Scatter
            fig.add_trace(go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(size=6, opacity=0.6)
            ))
            
            # Perfect line
            min_val, max_val = y_true.min(), y_true.max()
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect',
                line=dict(dash='dash', color='red')
            ))
            
            fig.update_layout(
                title=f'{model_name} - R²={r2:.4f}, MAE={mae:.2f}',
                xaxis_title='True Values',
                yaxis_title='Predictions',
                hovermode='closest'
            )
            
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
            
            fig.show()
            return fig
        
        except ImportError:
            warnings.warn("Plotly not installed, falling back to matplotlib")
            interactive = False
    
    # Matplotlib plot
    if figsize is None:
        figsize = auto_figsize(len(y_true))  # NEW!
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(y_true, y_pred, alpha=0.6, s=30)
    
    # Perfect line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predictions', fontsize=12)
    ax.set_title(f'{model_name}\nR² = {r2:.4f}, MAE = {mae:.2f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_residuals(y_true, y_pred, save_path: Optional[str] = None, figsize=(12, 5)):
    """Residual plots"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6, s=30)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residual Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    feature_names,
    importance_values,
    top_n: int = 15,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None
):
    """
    УЛУЧШЕНО v2.0: автоматический figsize
    """
    # Sort
    indices = np.argsort(importance_values)[-top_n:]
    
    if figsize is None:
        figsize = auto_figsize(top_n, base_size=(10, 6))  # NEW!
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.barh(range(len(indices)), importance_values[indices], alpha=0.8)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_learning_curve(train_sizes, train_scores, val_scores, save_path: Optional[str] = None):
    """Learning curve"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    ax.plot(train_sizes, train_mean, 'o-', label='Train', lw=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    
    ax.plot(train_sizes, val_mean, 'o-', label='Validation', lw=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return fig


def plot_cv_scores(cv_scores, save_path: Optional[str] = None):
    """CV scores plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.boxplot([cv_scores], labels=['CV Scores'])
    ax.set_ylabel('R² Score')
    ax.set_title(f'Cross-Validation Scores\nMean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return fig


def create_interactive_plot(data_dict: dict, plot_type: str = 'scatter'):
    """
    НОВОЕ v2.0: Создание интерактивных графиков
    
    Args:
        data_dict: {'x': [...], 'y': [...], 'labels': [...]}
        plot_type: 'scatter', 'line', 'bar'
    """
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        if plot_type == 'scatter':
            fig.add_trace(go.Scatter(
                x=data_dict['x'],
                y=data_dict['y'],
                mode='markers',
                text=data_dict.get('labels'),
                marker=dict(size=8)
            ))
        
        fig.update_layout(hovermode='closest')
        return fig
    
    except ImportError:
        raise ImportError("Plotly required: pip install plotly")
