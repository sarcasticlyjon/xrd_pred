"""
Model Registry v2.0

НОВОЕ:
- ✅ Model versioning
- ✅ Model comparison
- ✅ Lineage tracking
"""

import pickle
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime


class ModelRegistry:
    """
    УЛУЧШЕНО v2.0:
    - Версионирование моделей
    - Сравнение версий
    - История изменений
    """
    
    def __init__(self, registry_dir: str = 'models'):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.registry_dir / 'registry_index.json'
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Загрузка индекса"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Сохранение индекса"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def register(
        self,
        name: str,
        model,
        metrics: Optional[Dict] = None,
        version: Optional[str] = None,  # NEW!
        metadata: Optional[Dict] = None
    ) -> str:
        """
        NEW v2.0: Версионирование
        
        Returns:
            model_path: путь к сохранённой модели
        """
        # NEW: Auto-increment version
        if version is None:
            existing_versions = [
                v for k, v in self.index.items()
                if k.startswith(f"{name}_v")
            ]
            version = f"v{len(existing_versions) + 1}"
        
        model_key = f"{name}_{version}"
        model_path = self.registry_dir / f"{model_key}.pkl"
        
        # Сохранение модели
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Обновление индекса
        self.index[model_key] = {
            'name': name,
            'version': version,
            'path': str(model_path),
            'registered_at': datetime.now().isoformat(),
            'metrics': metrics or {},
            'metadata': metadata or {}
        }
        
        self._save_index()
        
        print(f"✅ Model registered: {model_key}")
        return str(model_path)
    
    def get_model(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Tuple[Any, Dict]:
        """
        NEW v2.0: Получить модель по версии
        
        Если version=None, возвращает последнюю версию
        """
        if version:
            model_key = f"{name}_{version}"
        else:
            # Последняя версия
            versions = [k for k in self.index.keys() if k.startswith(f"{name}_")]
            if not versions:
                raise ValueError(f"Model '{name}' not found")
            model_key = sorted(versions)[-1]  # последняя по алфавиту
        
        if model_key not in self.index:
            raise ValueError(f"Model '{model_key}' not found")
        
        info = self.index[model_key]
        
        with open(info['path'], 'rb') as f:
            model = pickle.load(f)
        
        return model, info
    
    def list_versions(self, name: str) -> List[Dict]:
        """NEW: Список всех версий модели"""
        versions = [
            {
                'version': info['version'],
                'registered_at': info['registered_at'],
                'metrics': info.get('metrics', {})
            }
            for key, info in self.index.items()
            if info['name'] == name
        ]
        
        return sorted(versions, key=lambda x: x['version'])
    
    def compare_versions(
        self,
        name: str,
        metric: str = 'r2_test'
    ) -> List[Dict]:
        """NEW: Сравнение версий модели"""
        versions = self.list_versions(name)
        
        comparison = []
        for v in versions:
            comparison.append({
                'version': v['version'],
                'metric_value': v['metrics'].get(metric, None),
                'date': v['registered_at']
            })
        
        return sorted(comparison, key=lambda x: x['metric_value'] or -999, reverse=True)
    
    def get_best_model(
        self,
        metric: str = 'r2_test'
    ) -> Tuple[str, Any, Dict]:
        """Получить лучшую модель по метрике"""
        best_key = None
        best_score = -float('inf')
        
        for key, info in self.index.items():
            score = info.get('metrics', {}).get(metric, -float('inf'))
            if score > best_score:
                best_score = score
                best_key = key
        
        if best_key is None:
            raise ValueError("No models in registry")
        
        info = self.index[best_key]
        with open(info['path'], 'rb') as f:
            model = pickle.load(f)
        
        return best_key, model, info


def save_model(model, filepath: str, metadata: Optional[Dict] = None):
    """Простое сохранение модели"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    if metadata:
        metadata_path = Path(filepath).with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_model(filepath: str, load_metadata: bool = False):
    """Простая загрузка модели"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    if load_metadata:
        metadata_path = Path(filepath).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return model, metadata
    
    return model
