"""
ML XRD v2.0 - Complete Machine Learning Pipeline for XRD Data

МОДУЛИ:
- data: Data loading and dataset building
- preprocessing: Data preprocessing and feature engineering
- models: Model training, tuning, and prediction
- analysis: Analysis, visualization, and reporting
- utils: Utilities, config, logging, validation

НОВОЕ в v2.0:
✅ 34 критических улучшения
✅ Production-ready качество
✅ Полная документация и тесты
"""

__version__ = "2.0.0"
__author__ = "ML XRD Team"

# Импорты из всех модулей
from .data import XRDDatasetBuilder, MetadataExtractor, XRDSpectrum
from .preprocessing import (
    TargetEncoder, normalize_intensity, parse_formula,
    group_train_test_split
)
from .models import (
    train_model, tune_hyperparameters, ModelRegistry,
    batch_predict
)
from .analysis import (
    plot_predictions, correlation_matrix, shap_importance,
    generate_html_report
)
from .utils import (
    Config, setup_logger, load_data, save_data,
    detect_outliers, retry, progress_tracker
)

__all__ = [
    # Data
    'XRDDatasetBuilder', 'MetadataExtractor', 'XRDSpectrum',
    # Preprocessing
    'TargetEncoder', 'normalize_intensity', 'parse_formula',
    'group_train_test_split',
    # Models
    'train_model', 'tune_hyperparameters', 'ModelRegistry',
    'batch_predict',
    # Analysis
    'plot_predictions', 'correlation_matrix', 'shap_importance',
    'generate_html_report',
    # Utils
    'Config', 'setup_logger', 'load_data', 'save_data',
    'detect_outliers', 'retry', 'progress_tracker'
]


def get_version():
    """Получить версию"""
    return __version__


def print_info():
    """Информация о проекте"""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    ML XRD v{__version__}                          ║
╠══════════════════════════════════════════════════════════════╣
║  Machine Learning Pipeline for XRD Data Analysis            ║
╠══════════════════════════════════════════════════════════════╣
║  Модули:                                                     ║
║    • Data Pipeline    - Dataset building, metadata          ║
║    • Preprocessing    - Encoders, transforms, chemistry     ║
║    • Models          - Training, tuning, registry           ║
║    • Analysis        - Plots, correlation, importance       ║
║    • Utils           - Config, logging, validation          ║
╠══════════════════════════════════════════════════════════════╣
║  Новое в v2.0: 34 критических улучшения                     ║
╚══════════════════════════════════════════════════════════════╝
""")
