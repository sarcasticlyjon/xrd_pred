"""
Reports v2.0

НОВОЕ:
- ✅ HTML reports
- ✅ Улучшенное форматирование
"""

import pandas as pd
from typing import Optional, Dict
from datetime import datetime


def generate_model_report(
    model_name: str,
    metrics: Dict,
    feature_importance: Optional[pd.DataFrame] = None,
    top_n_features: int = 15
) -> str:
    """Генерация текстового отчёта"""
    report = f"""
{'='*80}
MODEL REPORT: {model_name}
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

METRICS:
{'-'*80}
"""
    
    for key, value in metrics.items():
        if isinstance(value, float):
            report += f"  {key:25s}: {value:.4f}\n"
        else:
            report += f"  {key:25s}: {value}\n"
    
    if feature_importance is not None:
        report += f"\n{'='*80}\n"
        report += f"TOP {top_n_features} FEATURES:\n"
        report += f"{'-'*80}\n"
        
        top = feature_importance.head(top_n_features)
        for i, row in enumerate(top.itertuples(), 1):
            importance = row.importance if hasattr(row, 'importance') else row.importance_mean
            report += f"  {i:2d}. {row.feature:30s}: {importance:.4f}\n"
    
    report += f"\n{'='*80}\n"
    
    return report


def compare_models_report(
    models_metrics: Dict[str, Dict],
    sort_by: str = 'r2_test'
) -> str:
    """Сравнение моделей"""
    report = f"""
{'='*80}
MODELS COMPARISON
{'='*80}

Sorted by: {sort_by}

"""
    
    # Сортировка
    sorted_models = sorted(
        models_metrics.items(),
        key=lambda x: x[1].get(sort_by, -999),
        reverse=True
    )
    
    for i, (model_name, metrics) in enumerate(sorted_models, 1):
        report += f"{i}. {model_name}\n"
        for key in ['r2_train', 'r2_test', 'mae_test', 'rmse_test']:
            if key in metrics:
                report += f"   {key:15s}: {metrics[key]:.4f}\n"
        report += "\n"
    
    report += f"{'='*80}\n"
    
    return report


def generate_html_report(
    model_name: str,
    metrics: Dict,
    feature_importance: Optional[pd.DataFrame] = None,
    plots: Optional[Dict[str, str]] = None,
    output_path: str = 'report.html'
):
    """
    НОВОЕ v2.0: HTML отчёт
    
    Args:
        plots: {'predictions': 'path/to/img.png', 'importance': '...'}
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Report - {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #007bff; color: white; }}
        tr:hover {{ background: #f1f1f1; }}
        .metric {{ font-weight: bold; color: #007bff; }}
        img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; }}
        .timestamp {{ color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Model Report: {model_name}</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📈 Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
"""
    
    for key, value in metrics.items():
        if isinstance(value, float):
            html += f"<tr><td>{key}</td><td class='metric'>{value:.4f}</td></tr>\n"
        else:
            html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
    
    html += "</table>\n"
    
    # Feature importance
    if feature_importance is not None:
        html += "<h2>🔍 Feature Importance</h2>\n<table>\n"
        html += "<tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>\n"
        
        for i, row in enumerate(feature_importance.head(15).itertuples(), 1):
            imp = row.importance if hasattr(row, 'importance') else row.importance_mean
            html += f"<tr><td>{i}</td><td>{row.feature}</td><td>{imp:.4f}</td></tr>\n"
        
        html += "</table>\n"
    
    # Plots
    if plots:
        html += "<h2>📊 Visualizations</h2>\n"
        for plot_name, plot_path in plots.items():
            html += f"<h3>{plot_name.capitalize()}</h3>\n"
            html += f"<img src='{plot_path}' alt='{plot_name}'>\n"
    
    html += """
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✅ HTML report saved: {output_path}")
    
    return html
