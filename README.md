# 🚀 ML XRD v2.0

**Complete Machine Learning Pipeline for XRD Data Analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Version](https://img.shields.io/badge/version-2.0.0-green.svg)

---

## 📋 О проекте

ML XRD v2.0 - это профессиональный Python пакет для анализа XRD (X-Ray Diffraction) данных с использованием машинного обучения.

### ✨ Новое в v2.0

**34 критических улучшения:**
- ✅ Progress bars для всех операций (tqdm)
- ✅ Parallel processing (multiprocessing)
- ✅ Early stopping для моделей
- ✅ Model versioning (v1, v2, v3...)
- ✅ SHAP explainability
- ✅ HTML reports
- ✅ Structured JSON logging
- ✅ Chunked I/O для больших файлов
- ✅ Outlier detection (3 метода)
- ✅ Interactive plots (Plotly)
- И многое другое!

---

## 🎯 Возможности

### 📊 Data Pipeline
- Автоматическая загрузка XRD файлов
- Metadata extraction с гибкими правилами
- Recovery mode для битых файлов
- Проверка дубликатов
- Progress tracking

### 🔧 Preprocessing
- Target encoding с защитой от overfitting
- RobustScaler (устойчив к outliers)
- Улучшенный парсер химических формул (скобки, гидраты)
- Element features (electronegativity, mass, radius)
- Transform pipelines

### 🤖 Models
- XGBoost, LightGBM, Random Forest, Linear models
- Early stopping
- Hyperparameter tuning (Optuna, Grid, Random)
- Custom param ranges
- Time budget для оптимизации
- Model versioning
- Batch prediction

### 📈 Analysis
- Interactive plots (Plotly + Matplotlib)
- Auto figsize
- Parallel mutual information
- Partial correlation
- SHAP importance
- Confidence intervals
- HTML reports

### 🛠️ Utils
- Config environments (dev/staging/prod)
- Structured JSON logging
- Chunked I/O
- Outlier detection
- Retry decorator с backoff
- Progress tracking

---

## 📦 Установка

```bash
# Базовая установка
pip install -e .

# Со всеми зависимостями
pip install -e ".[all]"

# Только ML библиотеки
pip install -e ".[ml]"

# Для разработки
pip install -e ".[dev]"
```

---

## 🚀 Quick Start

```python
from mlxrd import XRDDatasetBuilder, train_model, plot_predictions

# 1. Построение датасета
builder = XRDDatasetBuilder(
    xrd_folder='data/raw/xrd',
    n_jobs=4,              # Параллельная обработка
    show_progress=True     # Progress bar
)

df, report = builder.build(return_report=True)
print(report)  # Детальная статистика

# 2. Подготовка данных
X = df.drop(columns=['intensity'])
y = df['intensity']

# 3. Обучение модели
model, metrics = train_model(
    X, y,
    model_type='xgboost',
    early_stopping=True,   # Early stopping
    verbose=True
)

# 4. Визуализация
y_pred = model.predict(X)
plot_predictions(
    y, y_pred,
    interactive=True,      # Интерактивный график
    save_path='pred.html'
)
```

---

## 📚 Документация

Полная документация в папке `docs/`:
- `DATA_V2_TESTING_GUIDE.md` - Data Pipeline
- `PREPROCESSING_V2_TESTING.md` - Preprocessing
- `MODELS_V2_TESTING.md` - Models
- `ANALYSIS_V2_TESTING.md` - Analysis
- `UTILS_V2_TESTING.md` - Utils
- `MASTER_GUIDE.md` - Общий гайд

---

## 🧪 Тестирование

```bash
# Все тесты
pytest tests/

# С coverage
pytest tests/ --cov=mlxrd

# Конкретный модуль
pytest tests/test_data.py -v
```

---

## 📊 Производительность

| Задача | v1.0 | v2.0 | Улучшение |
|--------|------|------|-----------|
| Dataset building (1000 files) | 45s | 12s | **3.8x** |
| Mutual info (50 features) | 8.2s | 2.1s | **3.9x** |
| Hyperparameter tuning | Manual | Auto | **∞** |

---

## 🤝 Вклад

Приветствуются pull requests! См. `CONTRIBUTING.md`.

---

## 📝 License

MIT License - см. `LICENSE`

---

## 👥 Авторы

ML XRD Team

---

**🎉 ML XRD v2.0 - Professional XRD Data Analysis with Machine Learning**
