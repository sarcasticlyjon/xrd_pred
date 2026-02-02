"""Helpers v2.0 - Retry, Progress"""
import time, functools, numpy as np
from typing import Optional, Callable
from tqdm import tqdm

def set_random_seed(seed: int):
    """Set random seed"""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except: pass
    try:
        import torch
        torch.manual_seed(seed)
    except: pass

def get_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """Get timestamp"""
    from datetime import datetime
    return datetime.now().strftime(format)

def memory_usage() -> dict:
    """Memory usage"""
    try:
        import psutil, os
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        return {
            'process_mb': mem.rss / 1024 / 1024,
            'process_gb': mem.rss / 1024 / 1024 / 1024
        }
    except: return {}

def print_memory_usage():
    """Print memory usage"""
    mem = memory_usage()
    if mem:
        print(f"Memory: {mem['process_mb']:.1f} MB")

def timing_decorator(func):
    """Timing decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"⏱️  {func.__name__}: {time.time()-start:.2f}s")
        return result
    return wrapper

class Timer:
    """Timer context manager"""
    def __init__(self, name: str = "Operation"):
        self.name = name
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, *args):
        print(f"⏱️  {self.name}: {time.time()-self.start:.2f}s")

def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def dict_to_pretty_string(d: dict, indent: int = 0) -> str:
    """Pretty print dict"""
    lines = []
    for k, v in d.items():
        lines.append(' '*indent + f"{k}: {v}")
    return '\n'.join(lines)

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    NEW v2.0: Retry decorator with exponential backoff
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt+1} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator

def progress_tracker(iterable, desc: str = "Processing"):
    """NEW v2.0: Progress tracker wrapper"""
    return tqdm(iterable, desc=desc)
