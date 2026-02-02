"""
Config v2.0

НОВОЕ:
- ✅ Environment-specific configs (dev/staging/prod)
- ✅ Config validation
- ✅ Environment variables support
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
import yaml
import json
import os
from pathlib import Path


@dataclass
class Config:
    """
    УЛУЧШЕНО v2.0:
    - Environment support
    - Validation
    - .env integration
    """
    # General
    project_name: str = 'ML_XRD'
    version: str = '2.0.0'
    environment: str = 'development'  # NEW! dev/staging/prod
    random_seed: int = 42
    
    # Paths
    data_dir: str = 'data'
    models_dir: str = 'models'
    results_dir: str = 'results'
    logs_dir: str = 'logs'
    
    # Data
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Models
    default_model: str = 'xgboost'
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    
    # Tuning
    optuna_trials: int = 100
    optuna_timeout: Optional[int] = None
    
    # Visualization
    figure_dpi: int = 150
    figure_format: str = 'png'
    plot_style: str = 'seaborn'
    
    # Logging
    log_level: str = 'INFO'
    log_to_file: bool = True
    log_format: str = 'json'  # NEW! 'json' or 'text'
    
    def __post_init__(self):
        """Validation после создания"""
        self.validate()
    
    def validate(self):
        """NEW: Валидация конфига"""
        assert self.environment in ['development', 'staging', 'production'], \
            f"Invalid environment: {self.environment}"
        assert 0 < self.test_size < 1, "test_size должен быть в (0, 1)"
        assert self.cv_folds > 1, "cv_folds должен быть > 1"
        assert self.log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']
    
    @classmethod
    def from_env(cls, env: str = 'development'):
        """
        NEW: Создать конфиг для environment
        
        Загружает базовый конфиг + environment-specific overrides
        """
        config = cls(environment=env)
        
        # Environment-specific overrides
        if env == 'production':
            config.log_level = 'WARNING'
            config.cv_folds = 10
            config.optuna_trials = 200
        elif env == 'staging':
            config.log_level = 'INFO'
            config.cv_folds = 7
        
        # Load from environment variables
        config._load_from_env_vars()
        
        return config
    
    def _load_from_env_vars(self):
        """NEW: Загрузка из environment variables"""
        # Пример: MLXRD_RANDOM_SEED=123
        prefix = 'MLXRD_'
        
        for key in asdict(self).keys():
            env_key = f"{prefix}{key.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                
                # Конвертация типов
                current_value = getattr(self, key)
                if isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                elif isinstance(current_value, bool):
                    value = value.lower() in ['true', '1', 'yes']
                
                setattr(self, key, value)
    
    def save(self, filepath: str):
        """Сохранение конфига"""
        path = Path(filepath)
        
        if path.suffix == '.yaml':
            with open(filepath, 'w') as f:
                yaml.dump(asdict(self), f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(asdict(self), f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    
    @classmethod
    def load(cls, filepath: str):
        """Загрузка конфига"""
        path = Path(filepath)
        
        if path.suffix == '.yaml':
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return asdict(self)


def load_config(filepath: str) -> Config:
    """Загрузить конфиг из файла"""
    return Config.load(filepath)


def save_config(config: Config, filepath: str):
    """Сохранить конфиг в файл"""
    config.save(filepath)


def get_default_config() -> Config:
    """Получить дефолтный конфиг"""
    return Config()
