"""Logging v2.0 - Structured JSON logging"""
import logging, json, sys
from datetime import datetime
from typing import Optional, Dict, Any

class StructuredLogger:
    """NEW v2.0: JSON structured logging"""
    def __init__(self, name: str, level: str = 'INFO'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        if not self._has_json_formatter():
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JsonFormatter())
            self.logger.addHandler(handler)

    def _has_json_formatter(self) -> bool:
        return any(
            isinstance(handler.formatter, JsonFormatter)
            for handler in self.logger.handlers
        )
    
    def log(self, level: str, message: str, **kwargs):
        """Log with structured data"""
        extra = {'extra_data': kwargs}
        getattr(self.logger, level.lower())(message, extra=extra)

class JsonFormatter(logging.Formatter):
    """JSON formatter"""
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        return json.dumps(log_data)

def setup_logger(name: str, level: str = 'INFO', log_file: Optional[str] = None, format_string: Optional[str] = None):
    """Setup logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    
    fmt = format_string or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    console.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console)
    
    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    
    return logger

def get_logger(name: str):
    """Get existing logger"""
    return logging.getLogger(name)

class ExperimentLogger:
    """Experiment logging"""
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
    
    def log_experiment(self, name: str, params: Dict, metrics: Dict, metadata: Optional[Dict] = None):
        """Log experiment"""
        import os, json
        os.makedirs(self.log_dir, exist_ok=True)
        
        log_entry = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'params': params,
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        filepath = f"{self.log_dir}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f:
            json.dump(log_entry, f, indent=2)

def log_experiment(name: str, params: Dict, metrics: Dict, log_dir: str = 'logs'):
    """Quick experiment logging"""
    logger = ExperimentLogger(log_dir)
    logger.log_experiment(name, params, metrics)
