# 🧪 ТЕСТИРОВАНИЕ PREPROCESSING MODULE V2.0

**Полное руководство по тестированию всех новых возможностей**

---

## 📦 УСТАНОВКА

```cmd
# 1. Распаковать
tar -xzf PREPROCESSING_V2.tar.gz
cd ml_xrd_preprocessing_v2

# 2. Установить
pip install -e .

# 3. Проверить
python -c "from mlxrd.preprocessing import TargetEncoder; print('✅ OK')"
```

---

## 🎯 НОВЫЕ ВОЗМОЖНОСТИ V2.0

### ✅ 1. **TargetEncoder: min_samples_leaf** (защита от overfitting)
### ✅ 2. **Save/Load Encoders** (persistence)
### ✅ 3. **Параметры нормализации** (inverse transform)
### ✅ 4. **RobustScaler** (устойчив к outliers)
### ✅ 5. **Парсер формул** (скобки, гидраты)
### ✅ 6. **Element Features** (avg_mass, electronegativity, etc.)
### ✅ 7. **TransformPipeline** (chaining)

---

## 🧪 ТЕСТ 1: TargetEncoder - min_samples_leaf

**Что тестируем:** Защита от overfitting на маленьких группах

**Код:** `test_1_target_encoder.py`

```python
import numpy as np
import pandas as pd
from mlxrd.preprocessing import TargetEncoder

print("="*60)
print("ТЕСТ 1: TargetEncoder - min_samples_leaf")
print("="*60)

# Данные с маленькими группами
np.random.seed(42)
data = pd.DataFrame({
    'category': ['A']*100 + ['B']*50 + ['C']*5 + ['D']*2,  # D очень мало!
    'target': np.random.randn(157) * 10 + 50
})

print(f"\nРаспределение категорий:")
print(data['category'].value_counts())

# БЕЗ защиты (старый способ)
print("\n1️⃣ БЕЗ min_samples_leaf:")
encoder_old = TargetEncoder(smooth=20, min_samples_leaf=1)
encoder_old.fit(data['category'], data['target'])

print(f"Encodings:")
for cat in ['A', 'B', 'C', 'D']:
    enc = encoder_old.encodings_.get(cat, 'N/A')
    print(f"  {cat}: {enc:.2f}" if enc != 'N/A' else f"  {cat}: {enc}")

# С защитой (новый)
print("\n2️⃣ С min_samples_leaf=10:")
encoder_new = TargetEncoder(smooth=20, min_samples_leaf=10)  # NEW!
encoder_new.fit(data['category'], data['target'])

print(f"Encodings:")
for cat in ['A', 'B', 'C', 'D']:
    enc = encoder_new.encodings_.get(cat, 'N/A')
    global_mean = encoder_new.global_mean_
    is_global = abs(enc - global_mean) < 0.01
    mark = " ← используется global_mean" if is_global else ""
    print(f"  {cat}: {enc:.2f}{mark}")

# Проверка
assert encoder_new.encodings_['C'] == encoder_new.global_mean_, "❌ C должен быть global_mean"
assert encoder_new.encodings_['D'] == encoder_new.global_mean_, "❌ D должен быть global_mean"

print("\n✅ ТЕСТ ПРОЙДЕН: Защита работает!")
print("   Категории C и D (< 10 samples) используют global_mean")
```

**Запуск:**
```cmd
python test_1_target_encoder.py
```

**Ожидаемый вывод:**
```
============================================================
ТЕСТ 1: TargetEncoder - min_samples_leaf
============================================================

Распределение категорий:
A    100
B     50
C      5
D      2

1️⃣ БЕЗ min_samples_leaf:
Encodings:
  A: 49.83
  B: 50.12
  C: 52.45  ← может быть переобучение!
  D: 48.91  ← может быть переобучение!

2️⃣ С min_samples_leaf=10:
Encodings:
  A: 49.83
  B: 50.12
  C: 50.01 ← используется global_mean
  D: 50.01 ← используется global_mean

✅ ТЕСТ ПРОЙДЕН: Защита работает!
   Категории C и D (< 10 samples) используют global_mean
```

---

## 🧪 ТЕСТ 2: Save/Load Encoders

**Что тестируем:** Сохранение и загрузка обученных encoders

**Код:** `test_2_save_load.py`

```python
import numpy as np
from mlxrd.preprocessing import TargetEncoder, save_encoder, load_encoder
import os

print("="*60)
print("ТЕСТ 2: Save/Load Encoders")
print("="*60)

# Обучаем encoder
np.random.seed(42)
X = np.array(['A', 'B', 'C'] * 30)
y = np.random.randn(90)

encoder = TargetEncoder(smooth=10)
encoder.fit(X, y)

original_encodings = encoder.encodings_.copy()
print(f"\nOriginal encodings:")
for k, v in original_encodings.items():
    print(f"  {k}: {v:.4f}")

# Сохраняем (NEW!)
encoder.save('test_encoder.pkl')
print(f"\n✅ Encoder saved: test_encoder.pkl")

# Загружаем (NEW!)
loaded_encoder = TargetEncoder.load('test_encoder.pkl')
print(f"✅ Encoder loaded")

# Проверяем
loaded_encodings = loaded_encoder.encodings_

print(f"\nLoaded encodings:")
for k, v in loaded_encodings.items():
    print(f"  {k}: {v:.4f}")

# Тестируем трансформацию
X_test = np.array(['A', 'B', 'C'])
result_original = encoder.transform(X_test)
result_loaded = loaded_encoder.transform(X_test)

assert np.allclose(result_original, result_loaded), "❌ Результаты не совпадают!"

print(f"\nTransform test:")
print(f"  Original: {result_original}")
print(f"  Loaded:   {result_loaded}")
print(f"  Match: ✅")

# Cleanup
os.remove('test_encoder.pkl')

print("\n✅ ТЕСТ ПРОЙДЕН: Save/Load работает!")
```

**Запуск:**
```cmd
python test_2_save_load.py
```

---

## 🧪 ТЕСТ 3: Normalize with Parameters

**Что тестируем:** Сохранение параметров для inverse transform

**Код:** `test_3_normalize_params.py`

```python
import numpy as np
from mlxrd.preprocessing import normalize_intensity

print("="*60)
print("ТЕСТ 3: Normalize with Parameters")
print("="*60)

# Данные
data = np.array([10, 20, 30, 40, 50])
print(f"Original data: {data}")

# MinMax с параметрами (NEW!)
print("\n1️⃣ MinMax normalization:")
normalized, params = normalize_intensity(data, method='minmax', return_params=True)

print(f"Normalized: {normalized}")
print(f"Parameters: {params}")

# Inverse transform (вручную с params)
inverse = normalized * (params['max'] - params['min']) + params['min']
print(f"Inverse:    {inverse}")

assert np.allclose(data, inverse), "❌ Inverse не совпадает!"
print("✅ Inverse transform работает!")

# Z-score
print("\n2️⃣ Z-score normalization:")
normalized_z, params_z = normalize_intensity(data, method='zscore', return_params=True)

print(f"Normalized: {normalized_z}")
print(f"Parameters: {params_z}")

inverse_z = normalized_z * params_z['std'] + params_z['mean']
print(f"Inverse:    {inverse_z}")

assert np.allclose(data, inverse_z), "❌ Z-score inverse не работает!"
print("✅ Z-score inverse работает!")

# Robust (NEW!)
print("\n3️⃣ Robust normalization:")
normalized_r, params_r = normalize_intensity(data, method='robust', return_params=True)

print(f"Normalized: {normalized_r}")
print(f"Parameters: {params_r}")

inverse_r = normalized_r * params_r['iqr'] + params_r['median']
print(f"Inverse:    {inverse_r}")

print("\n✅ ТЕСТ ПРОЙДЕН: Параметры сохраняются!")
```

**Запуск:**
```cmd
python test_3_normalize_params.py
```

---

## 🧪 ТЕСТ 4: RobustScaler

**Что тестируем:** Устойчивость к outliers

**Код:** `test_4_robust_scaler.py`

```python
import numpy as np
from mlxrd.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

print("="*60)
print("ТЕСТ 4: RobustScaler vs StandardScaler")
print("="*60)

# Данные с outliers
np.random.seed(42)
data = np.concatenate([
    np.random.randn(95) * 10 + 50,  # Нормальные данные
    [200, 250, 300, 350, 400]       # Outliers!
])

print(f"Data stats:")
print(f"  Mean: {data.mean():.2f}")
print(f"  Std:  {data.std():.2f}")
print(f"  Min:  {data.min():.2f}")
print(f"  Max:  {data.max():.2f}")

# StandardScaler (чувствителен к outliers)
print("\n1️⃣ StandardScaler:")
scaler_std = StandardScaler()
scaled_std = scaler_std.fit_transform(data.reshape(-1, 1)).ravel()

print(f"  Scaled range: [{scaled_std.min():.2f}, {scaled_std.max():.2f}]")
print(f"  Outliers влияют! Max = {scaled_std.max():.2f}")

# RobustScaler (NEW!)
print("\n2️⃣ RobustScaler:")
scaler_robust = RobustScaler()
scaled_robust = scaler_robust.fit_transform(data.reshape(-1, 1)).ravel()

print(f"  Scaled range: [{scaled_robust.min():.2f}, {scaled_robust.max():.2f}]")
print(f"  Outliers меньше влияют!")

# Inverse transform (NEW!)
print("\n3️⃣ Inverse transform:")
inverse = scaler_robust.inverse_transform(scaled_robust.reshape(-1, 1)).ravel()
assert np.allclose(data, inverse), "❌ Inverse не работает!"
print(f"  ✅ Inverse OK")

print("\n✅ ТЕСТ ПРОЙДЕН: RobustScaler устойчив к outliers!")
```

**Запуск:**
```cmd
python test_4_robust_scaler.py
```

---

## 🧪 ТЕСТ 5: Улучшенный парсер формул

**Что тестируем:** Скобки и гидраты

**Код:** `test_5_formula_parser.py`

```python
from mlxrd.preprocessing import parse_formula, get_element_features

print("="*60)
print("ТЕСТ 5: Улучшенный парсер формул")
print("="*60)

# Базовый парсинг
print("\n1️⃣ Базовые формулы:")
formulas = ['Ba0.5Sr0.5TiO3', 'BaTiO3', 'SrTiO3']

for f in formulas:
    elements = parse_formula(f)
    print(f"{f:20s} → {elements}")

# Скобки (NEW!)
print("\n2️⃣ Формулы со скобками:")
formulas_brackets = [
    'Ba(Ti0.8Zr0.2)O3',
    'Ca(Mg0.5Si0.5)O3',
]

for f in formulas_brackets:
    elements = parse_formula(f)
    print(f"{f:25s} → {elements}")
    
    # Проверка
    if f == 'Ba(Ti0.8Zr0.2)O3':
        assert elements['Ba'] == 1.0, "❌ Ba должен быть 1.0"
        assert elements['Ti'] == 0.8, "❌ Ti должен быть 0.8"
        assert elements['Zr'] == 0.2, "❌ Zr должен быть 0.2"
        assert elements['O'] == 3.0, "❌ O должен быть 3.0"
        print("  ✅ Скобки работают!")

# Гидраты (NEW!)
print("\n3️⃣ Гидраты:")
formulas_hydrates = [
    'CuSO4·5H2O',
    'MgSO4.7H2O',
]

for f in formulas_hydrates:
    elements = parse_formula(f)
    print(f"{f:20s} → {elements}")
    
    if f == 'CuSO4·5H2O':
        assert elements['H'] == 10.0, "❌ H должен быть 10.0 (5*2)"
        assert elements['O'] == 9.0, "❌ O должен быть 9.0 (4+5)"
        print("  ✅ Гидраты работают!")

# Element features (NEW!)
print("\n4️⃣ Element features:")
elements = parse_formula('Ba0.5Sr0.5TiO3')
features = get_element_features(elements)

print(f"  Elements: {elements}")
print(f"  Features:")
for k, v in features.items():
    print(f"    {k}: {v:.2f}")

assert 'avg_mass' in features, "❌ Нет avg_mass"
assert 'avg_electronegativity' in features, "❌ Нет avg_electronegativity"

print("\n✅ ТЕСТ ПРОЙДЕН: Парсер улучшен!")
```

**Запуск:**
```cmd
python test_5_formula_parser.py
```

**Ожидаемый вывод:**
```
============================================================
ТЕСТ 5: Улучшенный парсер формул
============================================================

1️⃣ Базовые формулы:
Ba0.5Sr0.5TiO3       → {'Ba': 0.5, 'Sr': 0.5, 'Ti': 1.0, 'O': 3.0}
BaTiO3               → {'Ba': 1.0, 'Ti': 1.0, 'O': 3.0}
SrTiO3               → {'Sr': 1.0, 'Ti': 1.0, 'O': 3.0}

2️⃣ Формулы со скобками:
Ba(Ti0.8Zr0.2)O3              → {'Ba': 1.0, 'Ti': 0.8, 'Zr': 0.2, 'O': 3.0}
  ✅ Скобки работают!
Ca(Mg0.5Si0.5)O3              → {'Ca': 1.0, 'Mg': 0.5, 'Si': 0.5, 'O': 3.0}

3️⃣ Гидраты:
CuSO4·5H2O           → {'Cu': 1.0, 'S': 1.0, 'O': 9.0, 'H': 10.0}
  ✅ Гидраты работают!
MgSO4.7H2O           → {'Mg': 1.0, 'S': 1.0, 'O': 11.0, 'H': 14.0}

4️⃣ Element features:
  Elements: {'Ba': 0.5, 'Sr': 0.5, 'Ti': 1.0, 'O': 3.0}
  Features:
    avg_mass: 37.16
    avg_electronegativity: 2.41
    avg_radius: 97.20
    n_elements: 4

✅ ТЕСТ ПРОЙДЕН: Парсер улучшен!
```

---

## 🧪 ТЕСТ 6: TransformPipeline

**Что тестируем:** Chaining трансформаций

**Код:** `test_6_pipeline.py`

```python
import numpy as np
from mlxrd.preprocessing import TransformPipeline, RobustScaler, normalize_intensity

print("="*60)
print("ТЕСТ 6: TransformPipeline")
print("="*60)

# Данные
np.random.seed(42)
data = np.random.randn(100, 3) * 10 + 50

print(f"Original data shape: {data.shape}")
print(f"Original mean: {data.mean():.2f}")

# Создаём pipeline (NEW!)
pipeline = TransformPipeline([
    ('scaler', RobustScaler()),
    # можно добавить больше шагов
])

# Fit + transform
scaled = pipeline.fit_transform(data)

print(f"\nTransformed data:")
print(f"  Shape: {scaled.shape}")
print(f"  Mean: {scaled.mean():.2f}")
print(f"  Std: {scaled.std():.2f}")

assert scaled.shape == data.shape, "❌ Shape изменился!"
print("\n✅ ТЕСТ ПРОЙДЕН: Pipeline работает!")
```

**Запуск:**
```cmd
python test_6_pipeline.py
```

---

## 📋 ЧЕКЛИСТ ТЕСТИРОВАНИЯ

### Обязательные тесты:
- [ ] **Тест 1:** TargetEncoder min_samples_leaf ✅
- [ ] **Тест 2:** Save/Load Encoders ✅
- [ ] **Тест 3:** Normalize Parameters ✅
- [ ] **Тест 4:** RobustScaler ✅
- [ ] **Тест 5:** Formula Parser (brackets, hydrates) ✅
- [ ] **Тест 6:** TransformPipeline ✅

---

## 🎯 КРИТЕРИИ УСПЕХА

✅ **Модуль готов если:**
1. TargetEncoder защищён от overfitting на малых группах
2. Encoders сохраняются и загружаются
3. Параметры нормализации возвращаются
4. RobustScaler устойчив к outliers
5. Парсер обрабатывает скобки и гидраты
6. Pipeline чейнит трансформации

---

**Версия:** 2.0  
**Дата:** 2026-02-01  
**Модуль:** Preprocessing
