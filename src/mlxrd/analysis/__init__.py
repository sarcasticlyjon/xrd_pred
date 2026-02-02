"""Analysis Module v2.0 - УЛУЧШЕНО"""
__version__ = "2.0.0"

from .plots import (
    plot_predictions, plot_residuals, plot_feature_importance,
    plot_learning_curve, plot_cv_scores, create_interactive_plot
)
from .correlation import (
    correlation_matrix, mutual_information_matrix, partial_correlation,
    plot_correlation_heatmap, feature_target_correlation
)
from .feature_importance import (
    compute_feature_importance, permutation_importance_parallel,
    plot_top_features, shap_importance
)
from .reports import (
    generate_model_report, compare_models_report,
    generate_html_report
)

__all__ = [
    'plot_predictions', 'plot_residuals', 'plot_feature_importance',
    'plot_learning_curve', 'plot_cv_scores', 'create_interactive_plot',
    'correlation_matrix', 'mutual_information_matrix', 'partial_correlation',
    'plot_correlation_heatmap', 'feature_target_correlation',
    'compute_feature_importance', 'permutation_importance_parallel',
    'plot_top_features', 'shap_importance',
    'generate_model_report', 'compare_models_report', 'generate_html_report'
]
