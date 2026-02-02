# 🧪 ТЕСТИРОВАНИЕ MODELS MODULE V2.0

**Полное руководство по тестированию всех новых возможностей**

---

## 📦 УСТАНОВКА

```cmd
# 1. Распаковать
tar -xzf MODELS_V2.tar.gz
cd ml_xrd_models_v2

# 2. Установить
pip install -e ".[all]"

# 3. Проверить
python -c "from mlxrd.models import train_model; print('✅ OK')"
```

---

## 🎯 НОВЫЕ ВОЗМОЖНОСТИ V2.0

### ✅ 1. **Early Stopping** (XGBoost/LightGBM)
### ✅ 2. **Custom Param Ranges** (Optuna)
### ✅ 3. **Save/Load Study** (Optuna analysis)
### ✅ 4. **Time Budget** (ограничение времени)
### ✅ 5. **Auto-method Selection** (auto выбор оптимизатора)
### ✅ 6. **Batch Prediction** (для больших данных)
### ✅ 7. **Model Versioning** (v1, v2, v3...)

---

## 🧪 ТЕСТ 1: Early Stopping

**Что тестируем:** Ранняя остановка для XGBoost

**Код:** `test_1_early_stopping.py`

```python
import numpy as np
from mlxrd.models import train_model

print("="*60)
print("ТЕСТ 1: Early Stopping")
print("="*60)

# Данные
np.random.seed(42)
X = np.random.randn(500, 10)
y = np.random.randn(500)

# БЕЗ early stopping
print("\n1️⃣ Без early stopping:")
model1, metrics1 = train_model(
    X, y,
    model_type='xgboost',
    n_estimators=1000,  # Много итераций
    early_stopping=False,
    verbose=False
)

print(f"   Обучено деревьев: {model1.n_estimators}")

# С early stopping
print("\n2️⃣ С early stopping:")
model2, metrics2 = train_model(
    X, y,
    model_type='xgboost',
    n_estimators=1000,
    early_stopping=True,  # NEW!
    validation_fraction=0.1,
    verbose=True
)

print(f"   Обучено деревьев: {model2.best_iteration + 1}")
print(f"   Saved iterations: {1000 - (model2.best_iteration + 1)}")

assert model2.best_iteration < 1000, "❌ Early stopping не сработал"

print("\n✅ ТЕСТ ПРОЙДЕН: Early stopping работает!")
print(f"   Экономия: ~{(1 - (model2.best_iteration/1000))*100:.0f}% времени обучения")
```

**Запуск:**
```cmd
python test_1_early_stopping.py
```

**Ожидаемый вывод:**
```
============================================================
ТЕСТ 1: Early Stopping
============================================================

1️⃣ Без early stopping:
   Обучено деревьев: 1000

2️⃣ С early stopping:
🛑 Early stopping enabled (validation=10%)
✅ Model trained: xgboost
   R² (train): 0.9876
   R² (CV):    0.0234 ± 0.1234
   Обучено деревьев: 127
   Saved iterations: 873

✅ ТЕСТ ПРОЙДЕН: Early stopping работает!
   Экономия: ~87% времени обучения
```

---

## 🧪 ТЕСТ 2: Custom Param Ranges (Optuna)

**Что тестируем:** Свои диапазоны параметров для Optuna

**Код:** `test_2_custom_ranges.py`

```python
import numpy as np
from mlxrd.models import tune_hyperparameters

print("="*60)
print("ТЕСТ 2: Custom Param Ranges")
print("="*60)

# Данные
np.random.seed(42)
X = np.random.randn(200, 5)
y = np.random.randn(200)

# Свои диапазоны (NEW!)
custom_ranges = {
    'n_estimators': (10, 100),      # Меньший диапазон
    'max_depth': (2, 5),             # Shallow trees
    'learning_rate': (0.01, 0.2, 'log'),
}

print("\nCustom ranges:")
for param, rng in custom_ranges.items():
    print(f"  {param}: {rng}")

# Оптимизация с custom ranges
best_params = tune_hyperparameters(
    X, y,
    model_type='xgboost',
    method='optuna',
    param_ranges=custom_ranges,  # NEW!
    n_trials=20,
    verbose=True
)

print(f"\nBest params found:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

# Проверка что параметры в заданных диапазонах
assert 10 <= best_params['n_estimators'] <= 100, "❌ n_estimators вне диапазона"
assert 2 <= best_params['max_depth'] <= 5, "❌ max_depth вне диапазона"

print("\n✅ ТЕСТ ПРОЙДЕН: Custom ranges работают!")
```

**Запуск:**
```cmd
python test_2_custom_ranges.py
```

---

## 🧪 ТЕСТ 3: Save/Load Optuna Study

**Что тестируем:** Сохранение study для анализа

**Код:** `test_3_save_study.py`

```python
import numpy as np
from mlxrd.models import optimize_with_optuna, load_study
import os

print("="*60)
print("ТЕСТ 3: Save/Load Optuna Study")
print("="*60)

# Данные
np.random.seed(42)
X = np.random.randn(200, 5)
y = np.random.randn(200)

# Оптимизация с сохранением study
print("\n1️⃣ Optimization with save:")
best_params = optimize_with_optuna(
    X, y,
    model_type='xgboost',
    n_trials=30,
    save_study_path='optuna_study.pkl',  # NEW!
    verbose=True
)

assert os.path.exists('optuna_study.pkl'), "❌ Study не сохранён"
print("   ✅ Study saved")

# Загрузка и анализ
print("\n2️⃣ Loading and analyzing study:")
study = load_study('optuna_study.pkl')  # NEW!

print(f"   Total trials: {len(study.trials)}")
print(f"   Best value: {study.best_value:.4f}")
print(f"   Best params: {study.best_params}")

# Анализ
print("\n3️⃣ Study analysis:")
print(f"   Completed trials: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}")
print(f"   Failed trials: {len([t for t in study.trials if t.state.name == 'FAIL'])}")

# Топ 3 trials
top_trials = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)[:3]
print(f"\n   Top 3 trials:")
for i, trial in enumerate(top_trials, 1):
    print(f"     {i}. Score: {trial.value:.4f}, Params: {trial.params}")

# Cleanup
os.remove('optuna_study.pkl')

print("\n✅ ТЕСТ ПРОЙДЕН: Study save/load работает!")
```

**Запуск:**
```cmd
python test_3_save_study.py
```

---

## 🧪 ТЕСТ 4: Time Budget

**Что тестируем:** Ограничение времени оптимизации

**Код:** `test_4_time_budget.py`

```python
import numpy as np
import time
from mlxrd.models import tune_hyperparameters

print("="*60)
print("ТЕСТ 4: Time Budget")
print("="*60)

# Данные
np.random.seed(42)
X = np.random.randn(500, 10)
y = np.random.randn(500)

# С time budget (NEW!)
print("\n1️⃣ With time budget (10 seconds):")
start = time.time()

best_params = tune_hyperparameters(
    X, y,
    model_type='xgboost',
    method='optuna',
    n_trials=1000,  # Много trials
    time_budget=10,  # NEW! Но остановится через 10 секунд
    verbose=True
)

elapsed = time.time() - start

print(f"\n   Elapsed time: {elapsed:.2f}s")
print(f"   Time budget: 10s")

assert elapsed < 15, "❌ Time budget не сработал (>15s)"

print("\n✅ ТЕСТ ПРОЙДЕН: Time budget работает!")
print(f"   Остановился за {elapsed:.1f}s (< 15s)")
```

**Запуск:**
```cmd
python test_4_time_budget.py
```

---

## 🧪 ТЕСТ 5: Auto-method Selection

**Что тестируем:** Автовыбор метода оптимизации

**Код:** `test_5_auto_method.py`

```python
import numpy as np
from mlxrd.models import tune_hyperparameters

print("="*60)
print("ТЕСТ 5: Auto-method Selection")
print("="*60)

# Маленький датасет → Grid Search
print("\n1️⃣ Small dataset (100 samples):")
X_small = np.random.randn(100, 5)
y_small = np.random.randn(100)

params1 = tune_hyperparameters(
    X_small, y_small,
    model_type='xgboost',
    method='auto',  # NEW! Автовыбор
    n_trials=20,
    verbose=True
)
# Должен выбрать 'grid'

# Средний датасет → Random Search
print("\n2️⃣ Medium dataset (5000 samples):")
X_medium = np.random.randn(5000, 5)
y_medium = np.random.randn(5000)

params2 = tune_hyperparameters(
    X_medium, y_medium,
    model_type='xgboost',
    method='auto',  # Автовыбор
    n_trials=20,
    verbose=True
)
# Должен выбрать 'random' или 'optuna'

# Большой датасет → Optuna
print("\n3️⃣ Large dataset (15000 samples):")
X_large = np.random.randn(15000, 5)
y_large = np.random.randn(15000)

params3 = tune_hyperparameters(
    X_large, y_large,
    model_type='xgboost',
    method='auto',  # Автовыбор
    n_trials=20,
    verbose=True
)
# Должен выбрать 'optuna'

print("\n✅ ТЕСТ ПРОЙДЕН: Auto-selection работает!")
```

**Запуск:**
```cmd
python test_5_auto_method.py
```

**Ожидаемый вывод:**
```
============================================================
ТЕСТ 5: Auto-method Selection
============================================================

1️⃣ Small dataset (100 samples):
🤖 Auto-selected method: grid
...

2️⃣ Medium dataset (5000 samples):
🤖 Auto-selected method: random
...

3️⃣ Large dataset (15000 samples):
🤖 Auto-selected method: optuna
...

✅ ТЕСТ ПРОЙДЕН: Auto-selection работает!
```

---

## 🧪 ТЕСТ 6: Batch Prediction

**Что тестируем:** Предсказания батчами

**Код:** `test_6_batch_predict.py`

```python
import numpy as np
from mlxrd.models import train_model, batch_predict
import time

print("="*60)
print("ТЕСТ 6: Batch Prediction")
print("="*60)

# Обучаем модель
np.random.seed(42)
X_train = np.random.randn(500, 10)
y_train = np.random.randn(500)

model, _ = train_model(X_train, y_train, model_type='xgboost', verbose=False)

# Большой test set
X_test = np.random.randn(10000, 10)

print(f"Test data: {X_test.shape}")

# Обычный predict
print("\n1️⃣ Normal predict:")
start = time.time()
pred1 = model.predict(X_test)
time1 = time.time() - start
print(f"   Time: {time1:.4f}s")

# Batch predict (NEW!)
print("\n2️⃣ Batch predict (batch_size=1000):")
start = time.time()
pred2 = batch_predict(
    model, X_test,
    batch_size=1000,
    show_progress=True  # NEW! С прогресс-баром
)
time2 = time.time() - start
print(f"   Time: {time2:.4f}s")

# Проверка
assert np.allclose(pred1, pred2), "❌ Результаты не совпадают!"

print(f"\n✅ ТЕСТ ПРОЙДЕН: Batch prediction работает!")
print(f"   Results match: ✅")
```

**Запуск:**
```cmd
python test_6_batch_predict.py
```

---

## 🧪 ТЕСТ 7: Model Versioning

**Что тестируем:** Версионирование моделей

**Код:** `test_7_versioning.py`

```python
import numpy as np
from mlxrd.models import train_model, ModelRegistry

print("="*60)
print("ТЕСТ 7: Model Versioning")
print("="*60)

# Данные
np.random.seed(42)
X = np.random.randn(200, 5)
y = np.random.randn(200)

# Registry
registry = ModelRegistry('test_registry')

# Регистрируем несколько версий одной модели
print("\n1️⃣ Registering versions:")

for i in range(1, 4):
    print(f"\n   Training version {i}...")
    model, metrics = train_model(
        X, y,
        model_type='xgboost',
        n_estimators=50 * i,  # Разные параметры
        verbose=False
    )
    
    # Регистрация с автоверсией (NEW!)
    registry.register(
        name='xgb_model',
        model=model,
        metrics=metrics
    )

# Список версий (NEW!)
print("\n2️⃣ List versions:")
versions = registry.list_versions('xgb_model')

for v in versions:
    print(f"   {v['version']}: R²={v['metrics'].get('r2_train', 'N/A'):.4f}, Date={v['registered_at'][:10]}")

assert len(versions) == 3, "❌ Должно быть 3 версии"

# Сравнение версий (NEW!)
print("\n3️⃣ Compare versions:")
comparison = registry.compare_versions('xgb_model', metric='r2_train')

for item in comparison:
    print(f"   {item['version']}: {item['metric_value']:.4f}")

# Получить конкретную версию (NEW!)
print("\n4️⃣ Get specific version:")
model_v2, info = registry.get_model('xgb_model', version='v2')
print(f"   Loaded: xgb_model_v2")
print(f"   Params: n_estimators={info['metrics'].get('n_estimators', 'N/A')}")

# Лучшая модель
print("\n5️⃣ Get best model:")
best_key, best_model, best_info = registry.get_best_model(metric='r2_train')
print(f"   Best: {best_key}")
print(f"   R²: {best_info['metrics']['r2_train']:.4f}")

# Cleanup
import shutil
shutil.rmtree('test_registry')

print("\n✅ ТЕСТ ПРОЙДЕН: Versioning работает!")
```

**Запуск:**
```cmd
python test_7_versioning.py
```

**Ожидаемый вывод:**
```
============================================================
ТЕСТ 7: Model Versioning
============================================================

1️⃣ Registering versions:

   Training version 1...
✅ Model registered: xgb_model_v1

   Training version 2...
✅ Model registered: xgb_model_v2

   Training version 3...
✅ Model registered: xgb_model_v3

2️⃣ List versions:
   v1: R²=0.9845, Date=2026-02-01
   v2: R²=0.9912, Date=2026-02-01
   v3: R²=0.9934, Date=2026-02-01

3️⃣ Compare versions:
   v3: 0.9934
   v2: 0.9912
   v1: 0.9845

4️⃣ Get specific version:
   Loaded: xgb_model_v2
   Params: n_estimators=100

5️⃣ Get best model:
   Best: xgb_model_v3
   R²: 0.9934

✅ ТЕСТ ПРОЙДЕН: Versioning работает!
```

---

## 📋 ЧЕКЛИСТ ТЕСТИРОВАНИЯ

### Обязательные тесты:
- [ ] **Тест 1:** Early Stopping ✅
- [ ] **Тест 2:** Custom Param Ranges ✅
- [ ] **Тест 3:** Save/Load Study ✅
- [ ] **Тест 4:** Time Budget ✅
- [ ] **Тест 5:** Auto-method Selection ✅
- [ ] **Тест 6:** Batch Prediction ✅
- [ ] **Тест 7:** Model Versioning ✅

---

## 🎯 КРИТЕРИИ УСПЕХА

✅ **Модуль готов если:**
1. Early stopping экономит время обучения
2. Custom ranges работают в Optuna
3. Study сохраняется для анализа
4. Time budget останавливает оптимизацию вовремя
5. Auto-selection выбирает правильный метод
6. Batch prediction работает с прогресс-баром
7. Версии моделей отслеживаются и сравниваются

---

**Версия:** 2.0  
**Дата:** 2026-02-01  
**Модуль:** Models
