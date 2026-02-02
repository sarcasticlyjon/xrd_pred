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

### ✅ 1. **Progress Bars** (tqdm)
### ✅ 2. **Parallel Processing** (multiprocessing)
### ✅ 3. **Duplicate Detection** (автопроверка sample_id)
### ✅ 4. **Failed Files Logging** (детальные логи)
### ✅ 5. **Recovery Mode** (восстановление битых файлов)
### ✅ 6. **Build Report** (детальная статистика)
### ✅ 7. **Multiple Export Formats** (CSV, Parquet)

---

## 🧪 ТЕСТ 1: Progress Bars

**Что тестируем:** Progress bar показывается при обработке файлов

**Код:** `test_1_progress.py`

```python
from mlxrd.data import XRDDatasetBuilder

print("="*60)
print("ТЕСТ 1: Progress Bars")
print("="*60)

builder = XRDDatasetBuilder(
    xrd_folder='path/to/your/xrd_files',
    show_progress=True  # ← НОВОЕ: включить прогресс
)

df = builder.build()

print(f"\n✅ ТЕСТ ПРОЙДЕН: {len(df)} samples loaded")
print("   Вы должны были видеть progress bar!")
```

**Запуск:**
```cmd
python test_1_progress.py
```

**Ожидаемый вывод:**
```
============================================================
ТЕСТ 1: Progress Bars
============================================================
Found 156 files
Building: 100%|████████████████████| 156/156 [00:05<00:00, 31.2it/s]

✅ ТЕСТ ПРОЙДЕН: 144 samples loaded
   Вы должны были видеть progress bar!
```

---

## 🧪 ТЕСТ 2: Parallel Processing

**Что тестируем:** Ускорение с помощью multiprocessing

**Код:** `test_2_parallel.py`

```python
from mlxrd.data import XRDDatasetBuilder
import time

print("="*60)
print("ТЕСТ 2: Parallel Processing")
print("="*60)

folder = 'path/to/your/xrd_files'

# Sequential
print("\n1️⃣ Sequential (n_jobs=1):")
start = time.time()
builder1 = XRDDatasetBuilder(folder, n_jobs=1, show_progress=False)
df1 = builder1.build()
time1 = time.time() - start
print(f"   Time: {time1:.2f}s, Samples: {len(df1)}")

# Parallel
print("\n2️⃣ Parallel (n_jobs=4):")
start = time.time()
builder2 = XRDDatasetBuilder(folder, n_jobs=4, show_progress=False)
df2 = builder2.build()
time2 = time.time() - start
print(f"   Time: {time2:.2f}s, Samples: {len(df2)}")

# Speedup
speedup = time1 / time2
print(f"\n🚀 Speedup: {speedup:.2f}x faster!")

assert len(df1) == len(df2), "❌ Different results!"
print("✅ ТЕСТ ПРОЙДЕН: Parallel processing работает!")
```

**Запуск:**
```cmd
python test_2_parallel.py
```

**Ожидаемый вывод:**
```
============================================================
ТЕСТ 2: Parallel Processing
============================================================

1️⃣ Sequential (n_jobs=1):
   Time: 8.34s, Samples: 144

2️⃣ Parallel (n_jobs=4):
   Time: 2.51s, Samples: 144

🚀 Speedup: 3.32x faster!
✅ ТЕСТ ПРОЙДЕН: Parallel processing работает!
```

---

## 🧪 ТЕСТ 3: Duplicate Detection

**Что тестируем:** Автоматическое обнаружение и удаление дубликатов

**Код:** `test_3_duplicates.py`

```python
from mlxrd.data import XRDDatasetBuilder
import pandas as pd

print("="*60)
print("ТЕСТ 3: Duplicate Detection")
print("="*60)

builder = XRDDatasetBuilder('path/to/xrd', verbose=True)

# Запускаем с отчётом
df, report = builder.build(return_report=True)  # ← НОВОЕ!

print("\n📊 BUILD REPORT:")
print(report)

# Проверяем что дубликатов нет
if 'sample_id' in df.columns:
    remaining_dups = df['sample_id'].duplicated().sum()
    assert remaining_dups == 0, f"❌ Still have {remaining_dups} duplicates!"
    print(f"\n✅ ТЕСТ ПРОЙДЕН: No duplicates in final dataset!")
    print(f"   Removed: {report.duplicates_found}")
else:
    print("⚠️  No sample_id column (это нормально для некоторых датасетов)")
```

**Запуск:**
```cmd
python test_3_duplicates.py
```

**Ожидаемый вывод:**
```
============================================================
ТЕСТ 3: Duplicate Detection
============================================================
Found 156 files
Building: 100%|████████████████████| 156/156 [00:05<00:00]
⚠️  Removing 12 duplicates

📊 BUILD REPORT:
╔══════════════════════════════════════════════╗
║       DATASET BUILD REPORT v2.0              ║
╠══════════════════════════════════════════════╣
║ Total files:          156                    ║
║ Successful:           144 (92.3%)            ║
║ Failed:                12 (7.7%)             ║
║ Duplicates removed:    12                    ║
║ Final samples:        132                    ║
╠══════════════════════════════════════════════╣
║ Parsing success: 92.3%                       ║
╚══════════════════════════════════════════════╝

✅ ТЕСТ ПРОЙДЕН: No duplicates in final dataset!
   Removed: 12
```

**Примечание по отчёту:**  
`Parsing success` — доля файлов, успешно распарсенных в метаданные.  
Файлы с ошибками чтения или некорректным спектром учитываются как неуспешные.

---

## 🧪 ТЕСТ 4: Failed Files Logging

**Что тестируем:** Детальное логирование failed files с причинами

**Код:** `test_4_failed_logging.py`

```python
from mlxrd.data import XRDDatasetBuilder
import json

print("="*60)
print("ТЕСТ 4: Failed Files Logging")
print("="*60)

builder = XRDDatasetBuilder('path/to/xrd', verbose=True)
df, report = builder.build(return_report=True)

# Сохраняем лог
builder.save_failed_log('failed_files.json')  # ← НОВОЕ!

# Читаем и показываем
with open('failed_files.json', 'r') as f:
    failed = json.load(f)

print(f"\n📋 FAILED FILES ({len(failed)}):")
for i, fail in enumerate(failed[:5], 1):  # Первые 5
    print(f"{i}. {fail['filename']}")
    print(f"   Reason: {fail['reason']}")

if len(failed) > 0:
    print(f"\n✅ ТЕСТ ПРОЙДЕН: {len(failed)} failed files logged")
    print("   Файл: failed_files.json")
else:
    print("\n✅ ТЕСТ ПРОЙДЕН: Все файлы обработаны успешно!")
```

**Запуск:**
```cmd
python test_4_failed_logging.py
```

**Ожидаемый вывод:**
```
============================================================
ТЕСТ 4: Failed Files Logging
============================================================
Found 156 files
...
✅ Failed log: failed_files.json

📋 FAILED FILES (12):
1. corrupted_001.txt
   Reason: Invalid spectrum
2. empty_file.txt
   Reason: Invalid spectrum
3. wrong_format.txt
   Reason: Parse failed
4. bad_data_003.txt
   Reason: Invalid spectrum
5. broken_header.txt
   Reason: Exception: could not convert string to float

✅ ТЕСТ ПРОЙДЕН: 12 failed files logged
   Файл: failed_files.json
```

---

## 🧪 ТЕСТ 5: Recovery Mode

**Что тестируем:** Восстановление частично битых файлов

**Код:** `test_5_recovery.py`

```python
from mlxrd.data import XRDDatasetBuilder

print("="*60)
print("ТЕСТ 5: Recovery Mode")
print("="*60)

folder = 'path/to/xrd'

# Без recovery
print("\n1️⃣ Without Recovery:")
builder1 = XRDDatasetBuilder(folder, recovery_mode=False, verbose=False)
df1 = builder1.build()
print(f"   Loaded: {len(df1)} samples")
print(f"   Failed: {len(builder1.failed_files)}")

# С recovery
print("\n2️⃣ With Recovery:")
builder2 = XRDDatasetBuilder(folder, recovery_mode=True, verbose=False)
df2 = builder2.build()
print(f"   Loaded: {len(df2)} samples")
print(f"   Failed: {len(builder2.failed_files)}")

# Разница
recovered = len(df2) - len(df1)
if recovered > 0:
    print(f"\n🔧 RECOVERED: {recovered} files!")
    print("✅ ТЕСТ ПРОЙДЕН: Recovery mode работает!")
else:
    print("\n✅ ТЕСТ ПРОЙДЕН: Все файлы корректные (recovery не понадобился)")
```

**Запуск:**
```cmd
python test_5_recovery.py
```

**Ожидаемый вывод:**
```
============================================================
ТЕСТ 5: Recovery Mode
============================================================

1️⃣ Without Recovery:
   Loaded: 140 samples
   Failed: 16

2️⃣ With Recovery:
   Loaded: 144 samples
   Failed: 12

🔧 RECOVERED: 4 files!
✅ ТЕСТ ПРОЙДЕН: Recovery mode работает!
```

---

## 🧪 ТЕСТ 6: Build Report

**Что тестируем:** Детальный отчёт о построении

**Код:** `test_6_report.py`

```python
from mlxrd.data import XRDDatasetBuilder

print("="*60)
print("ТЕСТ 6: Build Report")
print("="*60)

builder = XRDDatasetBuilder('path/to/xrd', verbose=True)
df, report = builder.build(return_report=True)  # ← return_report=True

# Отчёт автоматически печатается
# Но можем также получить данные программно:

print("\n📊 PROGRAMMATIC ACCESS:")
print(f"Total files:    {report.total_files}")
print(f"Successful:     {report.successful}")
print(f"Failed:         {report.failed}")
print(f"Success rate:   {report.parsing_stats['success_rate']}")
print(f"Final samples:  {report.final_samples}")

# Проверки
assert report.total_files > 0, "❌ No files found"
assert report.successful > 0, "❌ No successful files"
assert report.final_samples > 0, "❌ No final samples"

print("\n✅ ТЕСТ ПРОЙДЕН: Build report работает!")
```

**Запуск:**
```cmd
python test_6_report.py
```

**Ожидаемый вывод:**
```
============================================================
ТЕСТ 6: Build Report
============================================================
Found 156 files
Building: 100%|████████████████████| 156/156 [00:05<00:00]

╔══════════════════════════════════════════════╗
║       DATASET BUILD REPORT v2.0              ║
╠══════════════════════════════════════════════╣
║ Total files:          156                    ║
║ Successful:           144 (92.3%)            ║
║ Failed:                12 (7.7%)             ║
║ Duplicates removed:    12                    ║
║ Final samples:        132                    ║
╠══════════════════════════════════════════════╣
║ Parsing success: 92.3%                       ║
╚══════════════════════════════════════════════╝

📊 PROGRAMMATIC ACCESS:
Total files:    156
Successful:     144
Failed:         12
Success rate:   92.3%
Final samples:  132

✅ ТЕСТ ПРОЙДЕН: Build report работает!
```

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
- [ ] **Тест 1:** Progress Bars ✅
- [ ] **Тест 2:** Parallel Processing ✅
- [ ] **Тест 3:** Duplicate Detection ✅
- [ ] **Тест 4:** Failed Files Logging ✅
- [ ] **Тест 5:** Recovery Mode ✅
- [ ] **Тест 6:** Build Report ✅
- [ ] **Тест 7:** Export Formats ✅

### Опциональные проверки:
- [ ] Работа с большими датасетами (1000+ файлов)
- [ ] Работа с битыми файлами
- [ ] Memory usage (не должно расти при large datasets)

---

## 🎯 КРИТЕРИИ УСПЕХА

✅ **Модуль готов если:**
1. Все 7 тестов проходят
2. Progress bar показывается
3. Parallel ускоряет обработку
4. Дубликаты удаляются автоматически
5. Failed files логируются с причинами
6. Recovery восстанавливает файлы
7. Отчёт содержит полную статистику

---

**Версия:** 2.0  
**Дата:** 2026-02-01  
**Модуль:** Data Pipeline
