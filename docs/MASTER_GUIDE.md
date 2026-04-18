# 📘 ML XRD v2.0 - MASTER GUIDE

**Полное руководство по использованию всех модулей**

---

## 📑 Содержание

1. [Введение](#введение)
2. [Установка](#установка)
3. [Быстрый старт](#быстрый-старт)
4. [Модули](#модули)
5. [Примеры использования](#примеры-использования)
6. [Тестирование](#тестирование)
7. [FAQ](#faq)

---

## 🎯 Введение

ML XRD v2.0 - это комплексный пакет для анализа XRD данных с машинным обучением.

### Что нового в v2.0?

**34 критических улучшения** распределены по 5 модулям:

| Модуль | Улучшений | Ключевые возможности |
|--------|-----------|---------------------|
| **Data Pipeline** | 6 | Progress bars, parallel processing, recovery mode |
| **Preprocessing** | 7 | min_samples_leaf, robust scaler, formula parser |
| **Models** | 7 | Early stopping, versioning, custom ranges |
| **Analysis** | 7 | Interactive plots, SHAP, HTML reports |
| **Utils** | 7 | JSON logging, chunked I/O, outlier detection |

---

## 📦 Установка

### Системные требования

- Python 3.10+
- 4+ GB RAM (рекомендуется 8 GB)
- 500 MB свободного места

### Шаги установки

```bash
# 1. Распаковать архив
tar -xzf ML_XRD_V2_PROJECT.tar.gz
cd ML_XRD_V2_COMPLETE

# 2. Создать виртуальное окружение (рекомендуется)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# 3. Установить пакет
pip install -e ".[all]"

# 4. Проверить установку
python -c "from mlxrd import print_info; print_info()"
```

### Выборочная установка

```bash
# Только базовые зависимости
pip install -e .

# ML библиотеки
pip install -e ".[ml]"

# Визуализация
pip install -e ".[viz]"

# Для разработки
pip install -e ".[dev]"
```

---

## 🚀 Быстрый старт

### Пример 1: Полный Pipeline

```python
from mlxrd import (
    XRDDatasetBuilder, TargetEncoder,
    train_model, plot_predictions, Config
)

# Настройка
config = Config.from_env('development')

# 1. Построение датасета
builder = XRDDatasetBuilder(
    xrd_folder='data/raw/xrd',
)
df = builder.build()

# 2. Препроцессинг
# XRDDatasetBuilder теперь работает только в point-wise режиме.
X = df[['material', 'substrate', '2thetta']]
y = df['intensity']

encoder = TargetEncoder(min_samples_leaf=10)
X_encoded = encoder.fit_transform(X['material'], y)

# 3. Обучение
model, metrics = train_model(
    X_encoded, y,
    model_type='xgboost',
    early_stopping=True
)

# 4. Визуализация
y_pred = model.predict(X_encoded)
plot_predictions(y, y_pred, interactive=True)
```

 
### Пример 1b: Точечный датасет

```python
from mlxrd import XRDPointDatasetBuilder

builder = XRDPointDatasetBuilder(
    xrd_folder='data/raw/xrd',
    metadata_xlsx='data/raw/metadata/B-series_long.xlsx',
)
df_points = builder.build()
print(df_points.head())
```

Аналогично через `XRDDatasetBuilder`:

```python
from mlxrd import XRDDatasetBuilder

df_points = XRDDatasetBuilder(
    xrd_folder='data/raw/xrd',
    metadata_file='data/raw/metadata/B-series_long.xlsx',
).build()
```

---

## 🧭 Как устроен pipeline (логика работы)

1. **Data Pipeline**  
   `XRDDatasetBuilder` (через `XRDPointDatasetBuilder`) читает XRD-файлы, извлекает метаданные и строит point-wise датафрейм.  
   Результат: таблица точек спектра (`2thetta`, `intensity`) с технологическими и химическими признаками.  
   `XRDDatasetBuilder` сохранён как совместимый wrapper; рекомендуется напрямую использовать `XRDPointDatasetBuilder`.


### 📄 Формат имён файлов (важно)

По умолчанию парсер ожидает один из вариантов:

- `Material_substrate_123.txt`
- `Material_123_substrate.txt`
- `Material_substrate.txt`
- `Material.txt`
- `123 Material.txt`
- `123 Material substrate.txt`
- `B-467_STO_LAO_4.txt`
- `B-467_STO_LAO_4.txt`


Если файл не подходит ни под один шаблон, он считается **неуспешно распарсенным** и
не попадёт в итоговый датасет. В таком случае рекомендуется:

1. Переименовать файлы под одну из схем выше.
2. Либо подготовить внешний метадатасет и сшивать его с данными после `build()`.




2. **Preprocessing**  
   Признаки очищаются и преобразуются:  
   - `TargetEncoder` кодирует категориальные поля.  
   - `parse_formula` извлекает элементы из химической формулы.  
   - `add_element_columns` добавляет численные признаки по элементам.  

3. **Models**  
   `train_model` обучает модель и возвращает метрики (R², MAE, RMSE + CV).  
   Опционально: `tune_hyperparameters` подбирает параметры, `ModelRegistry` версионирует артефакты.

4. **Analysis**  
   Построение графиков, интерпретация важности признаков и генерация HTML-отчётов.

### Пример 2: Hyperparameter Tuning

```python
from mlxrd import tune_hyperparameters

# Автоматический выбор метода
best_params = tune_hyperparameters(
    X, y,
    model_type='xgboost',
    method='auto',              # auto-select
    time_budget=300,            # 5 минут max
    save_study_path='study.pkl'
)

print(f"Best params: {best_params}")
```

### Пример 3: Model Versioning

```python
from mlxrd import ModelRegistry

registry = ModelRegistry('models/')

# Регистрация моделей
registry.register('xgb_model', model1, metrics1)  # v1
registry.register('xgb_model', model2, metrics2)  # v2
registry.register('xgb_model', model3, metrics3)  # v3

# Получить лучшую
best_key, best_model, info = registry.get_best_model(metric='r2_test')
print(f"Best: {best_key} with R²={info['metrics']['r2_test']:.4f}")
```

---

## 📚 Модули

### 1. Data Pipeline

**Основные функции:**
- `XRDDatasetBuilder` - совместимый wrapper для point-wise сборки
- `XRDPointDatasetBuilder` - точечный датасет (2θ/intensity + метаданные)
- `MetadataExtractor` - извлечение метаданных
- `XRDSpectrum` - работа со спектрами

**Новое в v2.0:**
- Progress bars
- Parallel processing (n_jobs)
- Recovery mode
- Point-wise-only dataset builder
- Нормализация substrate и аннотация sap1
- Интеграция Excel-метаданных и газовых долей

**Подробнее:** `DATA_V2_TESTING_GUIDE.md`

---

### 2. Preprocessing

**Основные функции:**
- `TargetEncoder` - target encoding
- `RobustScaler` - robust scaling
- `parse_formula` - парсинг формул
- `TransformPipeline` - chaining

**Новое в v2.0:**
- min_samples_leaf защита
- Save/load encoders
- Скобки и гидраты в формулах (для гидратов используйте символ `·`)
- Element features
- Inverse transform

**Подробнее:** `PREPROCESSING_V2_TESTING.md`

---

### 3. Models

**Основные функции:**
- `train_model` - обучение
- `tune_hyperparameters` - тюнинг
- `ModelRegistry` - версионирование
- `batch_predict` - batch prediction

**Новое в v2.0:**
- Early stopping
- Custom param ranges
- Time budget
- Auto-method selection
- Model versioning
- Save/load study

**Подробнее:** `MODELS_V2_TESTING.md`

---

### 4. Analysis

**Основные функции:**
- `plot_predictions` - графики
- `correlation_matrix` - корреляции
- `shap_importance` - SHAP
- `generate_html_report` - отчёты

**Новое в v2.0:**
- Interactive plots (Plotly)
- Auto figsize
- Parallel mutual info
- Partial correlation
- Confidence intervals
- HTML reports

**Подробнее:** `ANALYSIS_V2_TESTING.md`

---

### 5. Utils

**Основные функции:**
- `Config` - конфигурация
- `StructuredLogger` - логирование
- `chunked_read` - chunked I/O
- `detect_outliers` - outliers
- `retry` - retry decorator

**Новое в v2.0:**
- Config environments
- JSON logging
- Chunked I/O
- Outlier detection (3 метода)
- Retry с backoff
- Progress tracker

**Подробнее:** `UTILS_V2_TESTING.md`

---

## 💡 Примеры использования

### Работа с большими датасетами

```python
from mlxrd import chunked_read, progress_tracker

# Обработка по частям
for chunk in chunked_read('large_file.csv', chunksize=10000):
    # Обработка chunk
    processed = preprocess(chunk)
    
    # Сохранение результатов
    save_chunk(processed)
```

### Outlier Detection

```python
from mlxrd import detect_outliers

# IQR method
outliers = detect_outliers(data, method='iqr', threshold=1.5)

# Z-score
outliers = detect_outliers(data, method='zscore', threshold=3)

# Isolation Forest
outliers = detect_outliers(data, method='isolation_forest')

# Очистка данных
clean_data = data[~outliers]
```

### Structured Logging

```python
from mlxrd import StructuredLogger

logger = StructuredLogger('experiment', level='INFO')

logger.log('info', 'Training started',
          model='xgboost',
          dataset_size=5000,
          n_features=50)

# Output (JSON):
# {"timestamp": "2026-02-01T18:15:23", "level": "INFO", 
#  "message": "Training started", "model": "xgboost", ...}
```

### Retry Decorator

```python
from mlxrd import retry
import requests

@retry(max_attempts=5, delay=1.0, backoff=2.0)
def fetch_data(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Автоматически повторит при ошибке с exponential backoff
data = fetch_data('https://api.example.com/data')
```

---

## 🧪 Тестирование

### Запуск всех тестов

```bash
# Все модули
pytest tests/ -v

# С coverage
pytest tests/ --cov=mlxrd --cov-report=html

# Конкретный модуль
pytest tests/test_data.py -v
```

### Ручное тестирование

Каждый модуль имеет подробный Testing Guide с готовыми тестами:

1. **Data Pipeline:** 7 тестов в `DATA_V2_TESTING_GUIDE.md`
2. **Preprocessing:** 6 тестов в `PREPROCESSING_V2_TESTING.md`
3. **Models:** 7 тестов в `MODELS_V2_TESTING.md`
4. **Analysis:** 7 тестов в `ANALYSIS_V2_TESTING.md`
5. **Utils:** 7 тестов в `UTILS_V2_TESTING.md`

**Всего: 34 готовых теста**

### Замечания по тестированию

- Для гидратов в химических формулах используйте символ `·` (например, `CuSO4·5H2O`).  
  Точка `.` рассматривается как десятичный разделитель (например, `Ba0.5Sr0.5TiO3`).
- `XRDDatasetBuilder.build(return_report=True)` больше не поддерживается,
  используйте `build()` для point-wise сборки.

---

## ❓ FAQ

### Q: Как выбрать метод оптимизации?

A: Используйте `method='auto'`:
```python
best_params = tune_hyperparameters(X, y, method='auto')
# Small dataset (< 1000) → grid search
# Medium (1000-10000) → random search
# Large (> 10000) → optuna
```

### Q: Как работать с битыми XRD файлами?

A: Включите recovery mode:
```python
builder = XRDDatasetBuilder(
    xrd_folder='data',
    recovery_mode=True  # Попытка восстановить
)
```

### Q: Как ускорить обработку большого датасета?

A: Используйте chunking и предфильтрацию признаков:
```python
# Point-wise builder
builder = XRDDatasetBuilder(xrd_folder='data')

# Chunked I/O
for chunk in chunked_read('large.csv', chunksize=10000):
    process(chunk)
```

### Q: Как избежать overfitting в TargetEncoder?

A: Используйте min_samples_leaf:
```python
encoder = TargetEncoder(
    smooth=20,
    min_samples_leaf=10  # Группы < 10 → global_mean
)
```

### Q: Как сохранить все артефакты эксперимента?

A: Используйте ModelRegistry + HTML reports:
```python
# Сохранение модели
registry = ModelRegistry('models/')
registry.register('model_v1', model, metrics)

# HTML отчёт
generate_html_report(
    model_name='XGBoost',
    metrics=metrics,
    feature_importance=importance,
    output_path='report.html'
)
```

---

## 🔗 Ссылки

- **Документация:** `ML_XRD_V2_DOCUMENTATION/`
- **GitHub:** https://github.com/your-org/xrd_pred
- **Issues:** https://github.com/your-org/xrd_pred/issues

---

## 📧 Поддержка

Если у вас возникли вопросы:
1. Проверьте FAQ выше
2. Прочитайте соответствующий Testing Guide
3. Откройте Issue на GitHub
4. При публикации проекта добавьте актуальный контакт (email/чат/канал)

> ⚠️ Замените ссылку на ваш реальный репозиторий при публикации проекта.

---

**🎉 Спасибо за использование ML XRD v2.0!**

*Версия: 2.0.0*  
*Дата: 2026-02-01*
