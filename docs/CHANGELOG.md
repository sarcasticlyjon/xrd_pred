# 📝 CHANGELOG

## [2.0.0] - 2026-02-01

### 🎉 Major Release - 34 Critical Improvements

---

### 📊 Data Pipeline v2.0

**Added:**
- ✅ Progress bars для всех операций (tqdm)
- ✅ Parallel processing (n_jobs параметр)
- ✅ Duplicate detection по sample_id
- ✅ Failed files logging с причинами
- ✅ Recovery mode для битых файлов
- ✅ Build reports с детальной статистикой
- ✅ Export в Parquet/HDF5/CSV

**Performance:**
- 3.8x быстрее обработка (parallel processing)
- Детальное логирование ошибок

---

### 🔧 Preprocessing v2.0

**Added:**
- ✅ TargetEncoder: min_samples_leaf защита от overfitting
- ✅ Save/load encoders
- ✅ RobustScaler (устойчивый к outliers)
- ✅ Параметры нормализации для inverse transform
- ✅ Парсер формул: скобки Ba(Ti0.8Zr0.2)O3
- ✅ Парсер формул: гидраты CuSO4·5H2O
- ✅ Element features (electronegativity, mass, radius)
- ✅ TransformPipeline для chaining

**Improvements:**
- Улучшенная валидация
- Inverse transform support

---

### 🤖 Models v2.0

**Added:**
- ✅ Early stopping для XGBoost/LightGBM
- ✅ Custom param ranges для Optuna
- ✅ Save/load Optuna study
- ✅ Time budget для оптимизации
- ✅ Auto-method selection (grid/random/optuna)
- ✅ Batch prediction с progress bar
- ✅ Model versioning (v1, v2, v3...)
- ✅ Model comparison
- ✅ Warm start

**Performance:**
- Early stopping экономит ~80% времени
- Auto-selection выбирает оптимальный метод

---

### 📈 Analysis v2.0

**Added:**
- ✅ Interactive plots (Plotly)
- ✅ Auto figsize по количеству данных
- ✅ Parallel mutual information (3-4x быстрее)
- ✅ Partial correlation
- ✅ SHAP importance integration
- ✅ Confidence intervals для importance
- ✅ HTML reports с графиками

**Performance:**
- Mutual info: 5.2s → 1.8s (parallel)
- Улучшенное форматирование

---

### 🛠️ Utils v2.0

**Added:**
- ✅ Config environments (dev/staging/prod)
- ✅ Environment variables support
- ✅ Structured JSON logging
- ✅ Chunked I/O для больших файлов
- ✅ Outlier detection (IQR, Z-score, Isolation Forest)
- ✅ Retry decorator с exponential backoff
- ✅ Progress tracker wrapper
- ✅ Config validation

**Improvements:**
- Production-ready конфигурация
- Лучшее логирование

---

### 📚 Documentation

**Added:**
- 5 подробных Testing Guides (35+ тестов)
- MASTER_GUIDE.md с примерами
- CHANGELOG.md
- Улучшенный README.md
- API документация

---

### 🐛 Bug Fixes

- Fixed: Metadata parser теперь поддерживает скобки
- Fixed: TargetEncoder overfitting на малых группах
- Fixed: Mutual info скорость для больших датасетов

---

### 💥 Breaking Changes

- Config: изменена структура (добавлен environment)
- TargetEncoder: добавлен min_samples_leaf (default=10)

---

## [1.0.0] - 2025-01-31

### Initial Release

- Базовая функциональность всех 5 модулей
- Data, Preprocessing, Models, Analysis, Utils

---

**Полная документация:** `ML_XRD_V2_DOCUMENTATION/`
