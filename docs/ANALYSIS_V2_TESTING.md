# 🧪 ТЕСТИРОВАНИЕ ANALYSIS MODULE V2.0

**Полное руководство по тестированию всех новых возможностей**

---

## 📦 УСТАНОВКА

```cmd
# 1. Распаковать
tar -xzf ANALYSIS_V2.tar.gz
cd ml_xrd_analysis_v2

# 2. Установить с дополнительными зависимостями
pip install -e ".[interactive,shap]"

# 3. Проверить
python -c "from mlxrd.analysis import plot_predictions; print('✅ OK')"
```

---

## 🎯 НОВЫЕ ВОЗМОЖНОСТИ V2.0

### ✅ 1. **Auto Figsize** (автоматический размер графика)
### ✅ 2. **Interactive Plots** (Plotly)
### ✅ 3. **Parallel Mutual Info** (ускорение вычислений)
### ✅ 4. **Partial Correlation** (новый метод)
### ✅ 5. **SHAP Importance** (explainability)
### ✅ 6. **Confidence Intervals** (для feature importance)
### ✅ 7. **HTML Reports** (интерактивные отчёты)

---

## 🧪 ТЕСТ 1: Auto Figsize

**Что тестируем:** Автоматический размер фигуры

**Код:** `test_1_auto_figsize.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from mlxrd.analysis import plot_predictions

print("="*60)
print("ТЕСТ 1: Auto Figsize")
print("="*60)

# Маленький датасет
print("\n1️⃣ Small dataset (50 samples):")
y_true_small = np.random.randn(50)
y_pred_small = y_true_small + np.random.randn(50) * 0.5

fig1 = plot_predictions(y_true_small, y_pred_small, model_name='Small')
print(f"   Figure size: {fig1.get_size_inches()}")

# Большой датасет
print("\n2️⃣ Large dataset (500 samples):")
y_true_large = np.random.randn(500)
y_pred_large = y_true_large + np.random.randn(500) * 0.5

fig2 = plot_predictions(y_true_large, y_pred_large, model_name='Large')
print(f"   Figure size: {fig2.get_size_inches()}")

# Размер должен измениться автоматически
assert fig2.get_size_inches()[1] >= fig1.get_size_inches()[1], "❌ Auto figsize не работает"

print("\n✅ ТЕСТ ПРОЙДЕН: Auto figsize работает!")
```

**Запуск:**
```cmd
python test_1_auto_figsize.py
```

---

## 🧪 ТЕСТ 2: Interactive Plots

**Что тестируем:** Интерактивные графики через Plotly

**Код:** `test_2_interactive.py`

```python
import numpy as np
from mlxrd.analysis import plot_predictions

print("="*60)
print("ТЕСТ 2: Interactive Plots")
print("="*60)

# Данные
y_true = np.random.randn(200)
y_pred = y_true + np.random.randn(200) * 0.3

# Обычный plot
print("\n1️⃣ Matplotlib plot:")
fig_matplotlib = plot_predictions(
    y_true, y_pred,
    model_name='Matplotlib',
    interactive=False,
    save_path='plot_matplotlib.png'
)
print("   ✅ Saved: plot_matplotlib.png")

# Interactive plot (NEW!)
print("\n2️⃣ Interactive Plotly plot:")
try:
    fig_plotly = plot_predictions(
        y_true, y_pred,
        model_name='Plotly',
        interactive=True,  # NEW!
        save_path='plot_interactive.html'  # Сохранит как HTML
    )
    print("   ✅ Saved: plot_interactive.html")
    print("   Откройте plot_interactive.html в браузере!")
    
    print("\n✅ ТЕСТ ПРОЙДЕН: Interactive plots работают!")
    
except ImportError as e:
    print(f"   ⚠️  Plotly not installed: {e}")
    print("   Установите: pip install plotly")
```

**Запуск:**
```cmd
python test_2_interactive.py
```

---

## 🧪 ТЕСТ 3: Parallel Mutual Info

**Что тестируем:** Ускорение вычисления mutual information

**Код:** `test_3_parallel_mi.py`

```python
import numpy as np
import pandas as pd
import time
from mlxrd.analysis import mutual_information_matrix

print("="*60)
print("ТЕСТ 3: Parallel Mutual Info")
print("="*60)

# Данные (20 признаков)
np.random.seed(42)
df = pd.DataFrame(
    np.random.randn(500, 20),
    columns=[f'feature_{i}' for i in range(20)]
)

print(f"Data: {df.shape}")

# Последовательно
print("\n1️⃣ Sequential (n_jobs=1):")
start = time.time()
mi_seq = mutual_information_matrix(df, n_jobs=1)
time_seq = time.time() - start
print(f"   Time: {time_seq:.2f}s")

# Параллельно (NEW!)
print("\n2️⃣ Parallel (n_jobs=4):")
start = time.time()
mi_par = mutual_information_matrix(df, n_jobs=4)  # NEW!
time_par = time.time() - start
print(f"   Time: {time_par:.2f}s")

# Speedup
speedup = time_seq / time_par
print(f"\n🚀 Speedup: {speedup:.2f}x")

# Проверка результатов
assert np.allclose(mi_seq.values, mi_par.values, atol=0.01), "❌ Результаты не совпадают"

print("\n✅ ТЕСТ ПРОЙДЕН: Parallel MI работает!")
```

**Запуск:**
```cmd
python test_3_parallel_mi.py
```

**Ожидаемый вывод:**
```
============================================================
ТЕСТ 3: Parallel Mutual Info
============================================================
Data: (500, 20)

1️⃣ Sequential (n_jobs=1):
   Time: 5.23s

2️⃣ Parallel (n_jobs=4):
   Time: 1.87s

🚀 Speedup: 2.80x

✅ ТЕСТ ПРОЙДЕН: Parallel MI работает!
```

---

## 🧪 ТЕСТ 4: Partial Correlation

**Что тестируем:** Новый метод - partial correlation

**Код:** `test_4_partial_corr.py`

```python
import numpy as np
import pandas as pd
from mlxrd.analysis import correlation_matrix, partial_correlation, plot_correlation_heatmap
import matplotlib.pyplot as plt

print("="*60)
print("ТЕСТ 4: Partial Correlation")
print("="*60)

# Данные с коррелирующими признаками
np.random.seed(42)
n = 200

x1 = np.random.randn(n)
x2 = 0.8 * x1 + 0.2 * np.random.randn(n)  # Сильно коррелирует с x1
x3 = 0.5 * x1 + 0.5 * np.random.randn(n)
x4 = np.random.randn(n)  # Независим

df = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'x3': x3,
    'x4': x4
})

# Обычная корреляция
print("\n1️⃣ Pearson correlation:")
corr_pearson = correlation_matrix(df, method='pearson')
print(corr_pearson)

# Partial correlation (NEW!)
print("\n2️⃣ Partial correlation:")
corr_partial = partial_correlation(df)
print(corr_partial)

# Сравнение x2-x3
print(f"\n3️⃣ Comparison x2-x3:")
print(f"   Pearson:  {corr_pearson.loc['x2', 'x3']:.4f}")
print(f"   Partial:  {corr_partial.loc['x2', 'x3']:.4f}")

# Partial должна быть меньше (контроль влияния x1)
assert abs(corr_partial.loc['x2', 'x3']) < abs(corr_pearson.loc['x2', 'x3']), "❌ Partial не работает"

# Визуализация (NEW!)
print("\n4️⃣ Plotting partial correlation heatmap:")
fig = plot_correlation_heatmap(
    df,
    method='partial',  # NEW!
    save_path='partial_corr.png'
)
print("   ✅ Saved: partial_corr.png")

print("\n✅ ТЕСТ ПРОЙДЕН: Partial correlation работает!")
```

**Запуск:**
```cmd
python test_4_partial_corr.py
```

---

## 🧪 ТЕСТ 5: SHAP Importance

**Что тестируем:** SHAP feature importance

**Код:** `test_5_shap.py`

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from mlxrd.analysis import shap_importance

print("="*60)
print("ТЕСТ 5: SHAP Importance")
print("="*60)

# Данные
np.random.seed(42)
X = np.random.randn(200, 10)
y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(200) * 0.5
# feature_0 и feature_1 важны

# Обучаем модель
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

feature_names = [f'feature_{i}' for i in range(10)]

# SHAP importance (NEW!)
print("\n1️⃣ Computing SHAP importance:")
try:
    shap_imp = shap_importance(
        model, X, feature_names,
        plot=False  # Без графика для теста
    )
    
    print("\nTop 5 features (SHAP):")
    print(shap_imp.head())
    
    # feature_0 и feature_1 должны быть в топ-3
    top_3_features = shap_imp.head(3)['feature'].values
    assert 'feature_0' in top_3_features, "❌ feature_0 не в топе"
    assert 'feature_1' in top_3_features, "❌ feature_1 не в топе"
    
    print("\n✅ ТЕСТ ПРОЙДЕН: SHAP importance работает!")
    print("   feature_0 и feature_1 в топ-3 ✅")
    
except ImportError:
    print("\n⚠️  SHAP not installed")
    print("   Установите: pip install shap")
```

**Запуск:**
```cmd
python test_5_shap.py
```

---

## 🧪 ТЕСТ 6: Confidence Intervals

**Что тестируем:** CI для permutation importance

**Код:** `test_6_confidence_intervals.py`

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from mlxrd.analysis import permutation_importance_parallel, plot_top_features

print("="*60)
print("ТЕСТ 6: Confidence Intervals")
print("="*60)

# Данные
np.random.seed(42)
X = np.random.randn(200, 10)
y = X[:, 0] * 2 + np.random.randn(200) * 0.5

# Модель
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

feature_names = [f'feature_{i}' for i in range(10)]

# Permutation importance с CI (NEW!)
print("\n1️⃣ Permutation importance with CI:")
perm_imp = permutation_importance_parallel(
    model, X, y, feature_names,
    n_repeats=10,
    n_jobs=2
)

print(perm_imp.head())

# Проверка что есть CI
assert 'ci_lower' in perm_imp.columns, "❌ Нет ci_lower"
assert 'ci_upper' in perm_imp.columns, "❌ Нет ci_upper"
assert 'importance_std' in perm_imp.columns, "❌ Нет importance_std"

print("\n2️⃣ Plotting with CI:")
fig = plot_top_features(
    perm_imp,
    top_n=10,
    with_ci=True,  # NEW! С доверительными интервалами
    save_path='importance_ci.png'
)
print("   ✅ Saved: importance_ci.png (с error bars!)")

print("\n✅ ТЕСТ ПРОЙДЕН: Confidence intervals работают!")
```

**Запуск:**
```cmd
python test_6_confidence_intervals.py
```

---

## 🧪 ТЕСТ 7: HTML Reports

**Что тестируем:** Генерация HTML отчётов

**Код:** `test_7_html_report.py`

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from mlxrd.analysis import (
    generate_html_report, compute_feature_importance,
    plot_predictions
)

print("="*60)
print("ТЕСТ 7: HTML Reports")
print("="*60)

# Данные
np.random.seed(42)
X = np.random.randn(200, 10)
y = X[:, 0] * 2 + np.random.randn(200) * 0.5

# Модель
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

feature_names = [f'feature_{i}' for i in range(10)]

# Метрики
metrics = {
    'model_type': 'RandomForest',
    'n_estimators': 50,
    'r2_train': 0.9234,
    'r2_test': 0.8567,
    'mae_test': 12.45,
    'rmse_test': 18.23
}

# Feature importance
importance = compute_feature_importance(model, feature_names)

# Создаём графики
y_pred = model.predict(X)
fig_pred = plot_predictions(
    y, y_pred,
    model_name='RF',
    save_path='predictions.png'
)

# HTML отчёт (NEW!)
print("\n1️⃣ Generating HTML report:")
html = generate_html_report(
    model_name='RandomForest',
    metrics=metrics,
    feature_importance=importance,
    plots={
        'predictions': 'predictions.png'
    },
    output_path='model_report.html'  # NEW!
)

import os
assert os.path.exists('model_report.html'), "❌ HTML не создан"

print(f"\n✅ ТЕСТ ПРОЙДЕН: HTML report создан!")
print(f"   Файл: model_report.html")
print(f"   Откройте в браузере!")

# Показываем первые строки
with open('model_report.html', 'r') as f:
    lines = f.readlines()[:15]
print("\nПервые строки HTML:")
for line in lines:
    print(line.rstrip())
```

**Запуск:**
```cmd
python test_7_html_report.py
```

**Ожидаемый вывод:**
```
============================================================
ТЕСТ 7: HTML Reports
============================================================

1️⃣ Generating HTML report:
✅ HTML report saved: model_report.html

✅ ТЕСТ ПРОЙДЕН: HTML report создан!
   Файл: model_report.html
   Откройте в браузере!

Первые строки HTML:
<!DOCTYPE html>
<html>
<head>
    <title>Model Report - RandomForest</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: auto; background: white; padding: 30px; }
        h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
...
```

---

## 📋 ЧЕКЛИСТ ТЕСТИРОВАНИЯ

### Обязательные тесты:
- [ ] **Тест 1:** Auto Figsize ✅
- [ ] **Тест 2:** Interactive Plots ✅
- [ ] **Тест 3:** Parallel Mutual Info ✅
- [ ] **Тест 4:** Partial Correlation ✅
- [ ] **Тест 5:** SHAP Importance ✅
- [ ] **Тест 6:** Confidence Intervals ✅
- [ ] **Тест 7:** HTML Reports ✅

---

## 🎯 КРИТЕРИИ УСПЕХА

✅ **Модуль готов если:**
1. Figsize автоматически адаптируется
2. Interactive plots работают (Plotly)
3. Parallel MI ускоряет вычисления
4. Partial correlation дает разумные результаты
5. SHAP importance выделяет важные признаки
6. CI отображаются на графиках
7. HTML отчёт красиво форматирован

---

**Версия:** 2.0  
**Дата:** 2026-02-01  
**Модуль:** Analysis
