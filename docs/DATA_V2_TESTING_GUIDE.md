# 🧪 ТЕСТИРОВАНИЕ DATA MODULE V2.0

**Полное руководство по тестированию всех новых возможностей**

---

## 📦 УСТАНОВКА

```cmd
# 1. Распаковать
tar -xzf DATA_MODULE_V2.tar.gz
cd ml_xrd_data_v2

# 2. Установить
pip install -e .

# 3. Проверить
python -c "from mlxrd.data import XRDDatasetBuilder; print('✅ OK')"
```

---

## 🎯 НОВЫЕ ВОЗМОЖНОСТИ V2.0

### ✅ 1. **Point-wise сборка** (`2thetta`/`intensity`)
### ✅ 2. **Wrapper compatibility** (`XRDDatasetBuilder` -> point-wise)
### ✅ 3. **Metadata enrichment** (Excel + gas ratios)
### ✅ 4. **Deprecated API guard** (`return_report=True` -> ошибка)
### ✅ 5. **Multiple Export Formats** (CSV, Parquet)

---

## 📄 Формат имён файлов (важно)

Парсер метаданных работает по шаблонам:

- `Material_substrate_123.txt`
- `Material_123_substrate.txt`
- `Material_substrate.txt`
- `Material.txt`
- `123 Material.txt`
- `123 Material substrate.txt`
- `B-467_STO_LAO_4.txt`

Если имена отличаются (например, другой порядок или дополнительные токены),
файл будет считаться **неуспешно распарсенным** и не попадёт в итоговый датасет.

---

## 📊 Какие столбцы формируются

`XRDDatasetBuilder` теперь формирует только point-wise датасет и
возвращает колонки `2thetta` и `intensity` (плюс метаданные/признаки).

### Точечный датасет (основной формат)

Для построения датасета с колонками `2thetta/intensity` и расширенными
технологическими параметрами используйте `XRDPointDatasetBuilder`.
Он собирает данные по каждой точке спектра и добавляет поля из Excel-метаданных.

Пример:

```python
from mlxrd import XRDPointDatasetBuilder

builder = XRDPointDatasetBuilder(
    xrd_folder='data/raw/xrd',
    metadata_xlsx='data/raw/metadata/B-series_long.xlsx',
)
df_points = builder.build()
print(df_points.columns)
```

Через `XRDDatasetBuilder`:

```python
from mlxrd import XRDDatasetBuilder

df_points = XRDDatasetBuilder(
    xrd_folder='data/raw/xrd',
    metadata_file='data/raw/metadata/B-series_long.xlsx',
).build()
print(df_points.columns)
```
---

## 📄 Формат имён файлов (важно)

Парсер метаданных работает по шаблонам:

- `Material_substrate_123.txt`
- `Material_123_substrate.txt`
- `Material_substrate.txt`
- `Material.txt`
- `123 Material.txt`
- `123 Material substrate.txt`

Если имена отличаются (например, другой порядок или дополнительные токены),
файл будет считаться **неуспешно распарсенным** и не попадёт в итоговый датасет.

---

## 🧪 ТЕСТ 1: Базовая point-wise сборка

**Что тестируем:** `XRDDatasetBuilder` формирует только point-wise датасет.

**Код:** `test_1_pointwise_build.py`

```python
from mlxrd.data import XRDDatasetBuilder

builder = XRDDatasetBuilder(
    xrd_folder='path/to/your/xrd_files',
    metadata_file='path/to/metadata.xlsx',
)
df = builder.build()

assert {'2thetta', 'intensity'}.issubset(df.columns)
print(f"✅ OK: {len(df)} rows, point-wise columns present")
```

---

## 🧪 ТЕСТ 2: Проверка обёртки XRDDatasetBuilder

**Что тестируем:** `XRDDatasetBuilder` и `XRDPointDatasetBuilder` дают одинаковую схему.

**Код:** `test_2_wrapper_consistency.py`

```python
from mlxrd.data import XRDDatasetBuilder, XRDPointDatasetBuilder

folder = 'path/to/your/xrd_files'
meta = 'path/to/metadata.xlsx'

df_wrap = XRDDatasetBuilder(folder, metadata_file=meta).build()
df_point = XRDPointDatasetBuilder(folder, metadata_xlsx=meta).build()

assert set(df_wrap.columns) == set(df_point.columns)
print("✅ Wrapper consistency passed")
```

---

## 🧪 ТЕСТ 3: Поведение deprecated-параметра

**Что тестируем:** `return_report=True` больше не поддерживается.

**Код:** `test_3_return_report_deprecated.py`

```python
from mlxrd.data import XRDDatasetBuilder

builder = XRDDatasetBuilder('path/to/xrd')
try:
    builder.build(return_report=True)
except ValueError as e:
    assert 'return_report' in str(e)
    print("✅ deprecated behavior confirmed")
```

**Примечание:**  
`Parsing success` — доля файлов, успешно распарсенных в метаданные.  
Файлы с ошибками чтения или некорректным спектром учитываются как неуспешные.

---

## 🧪 ТЕСТ 4 (АРХИВ): Failed Files Logging

**Статус:** удалено из актуального API.  
`save_failed_log` и отдельный failed-log pipeline были частью агрегированного режима и больше не поддерживаются.

---

## 🧪 ТЕСТ 5 (АРХИВ): Recovery Mode

**Статус:** удалено из актуального API.  
Recovery-поведение было частью удалённой агрегированной ветки `XRDDatasetBuilder`.

---

## 🧪 ТЕСТ 7: Export Formats

**Что тестируем:** Экспорт в разные форматы

**Код:** `test_7_export.py`

```python
from mlxrd.data import XRDDatasetBuilder
import pandas as pd
import os

print("="*60)
print("ТЕСТ 7: Export Formats")
print("="*60)

builder = XRDDatasetBuilder('path/to/xrd', verbose=False)
df = builder.build()

print(f"Dataset: {df.shape}")

# CSV
print("\n1️⃣ Export to CSV:")
builder.export(df, 'output.csv', format='csv')
assert os.path.exists('output.csv'), "❌ CSV not created"
df_csv = pd.read_csv('output.csv')
assert len(df_csv) == len(df), "❌ CSV size mismatch"
print("   ✅ CSV OK")

# Parquet
print("\n2️⃣ Export to Parquet:")
builder.export(df, 'output.parquet', format='parquet')
assert os.path.exists('output.parquet'), "❌ Parquet not created"
df_parquet = pd.read_parquet('output.parquet')
assert len(df_parquet) == len(df), "❌ Parquet size mismatch"
print("   ✅ Parquet OK")

# Auto-detect
print("\n3️⃣ Export with auto-detect:")
builder.export(df, 'output_auto.parquet', format='auto')
print("   ✅ Auto-detect OK")

print("\n✅ ТЕСТ ПРОЙДЕН: Все форматы работают!")

# Cleanup
os.remove('output.csv')
os.remove('output.parquet')
os.remove('output_auto.parquet')
```

**Запуск:**
```cmd
python test_7_export.py
```

**Ожидаемый вывод:**
```
============================================================
ТЕСТ 7: Export Formats
============================================================
Dataset: (132, 8)

1️⃣ Export to CSV:
✅ Exported: output.csv (0.12 MB, csv)
   ✅ CSV OK

2️⃣ Export to Parquet:
✅ Exported: output.parquet (0.05 MB, parquet)
   ✅ Parquet OK

3️⃣ Export with auto-detect:
✅ Exported: output_auto.parquet (0.05 MB, parquet)
   ✅ Auto-detect OK

✅ ТЕСТ ПРОЙДЕН: Все форматы работают!
```

---

## 📋 ЧЕКЛИСТ ТЕСТИРОВАНИЯ

### Обязательные тесты:
- [ ] **Тест 1:** Базовая point-wise сборка ✅
- [ ] **Тест 2:** Wrapper consistency ✅
- [ ] **Тест 3:** Deprecated API guard ✅
- [ ] **Тест 7:** Export Formats ✅

### Опциональные проверки:
- [ ] Работа с большими датасетами (1000+ файлов)
- [ ] Memory usage (не должно расти при large datasets)

---

## 🎯 КРИТЕРИИ УСПЕХА

✅ **Модуль готов если:**
1. Базовая point-wise сборка проходит успешно
2. Wrapper (`XRDDatasetBuilder`) совпадает по схеме с `XRDPointDatasetBuilder`
3. `return_report=True` корректно отклоняется
4. Экспорт в CSV/Parquet работает

---

**Версия:** 2.0  
**Дата:** 2026-02-01  
**Модуль:** Data Pipeline
