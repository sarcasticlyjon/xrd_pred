"""I/O v2.0 - Chunked I/O, Cloud Storage"""
import pandas as pd
from pathlib import Path
from typing import Optional, Iterator

def load_data(filepath: str, **kwargs):
    """Universal data loading"""
    path = Path(filepath)
    
    if path.suffix == '.csv':
        return pd.read_csv(filepath, **kwargs)
    elif path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(filepath, **kwargs)
    elif path.suffix == '.parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif path.suffix == '.json':
        return pd.read_json(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

def save_data(df: pd.DataFrame, filepath: str, **kwargs):
    """Universal data saving"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.csv':
        df.to_csv(filepath, **kwargs)
    elif path.suffix in ['.xlsx', '.xls']:
        df.to_excel(filepath, **kwargs)
    elif path.suffix == '.parquet':
        df.to_parquet(filepath, **kwargs)
    elif path.suffix == '.json':
        df.to_json(filepath, **kwargs)

def chunked_read(filepath: str, chunksize: int = 10000) -> Iterator[pd.DataFrame]:
    """NEW v2.0: Chunked reading for large files"""
    return pd.read_csv(filepath, chunksize=chunksize)

def chunked_write(df: pd.DataFrame, filepath: str, chunksize: int = 10000):
    """NEW v2.0: Chunked writing"""
    mode = 'w'
    for i in range(0, len(df), chunksize):
        chunk = df.iloc[i:i+chunksize]
        chunk.to_csv(filepath, mode=mode, header=(i==0), index=False)
        mode = 'a'

class CloudStorage:
    """NEW v2.0: Cloud storage helper (S3, GCS, Azure)"""
    def __init__(self, provider: str = 's3'):
        self.provider = provider
    
    def upload(self, local_path: str, remote_path: str):
        """Upload to cloud"""
        raise NotImplementedError("Cloud storage requires additional setup")
    
    def download(self, remote_path: str, local_path: str):
        """Download from cloud"""
        raise NotImplementedError("Cloud storage requires additional setup")

def export_results(results: dict, output_dir: str, experiment_name: str, formats: list = None):
    """Export results"""
    import json
    from pathlib import Path
    
    formats = formats or ['json']
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved = []
    
    for fmt in formats:
        if fmt == 'json':
            filepath = f"{output_dir}/{experiment_name}.json"
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            saved.append(filepath)
    
    return saved

def create_project_structure(project_dir: str):
    """Create project structure"""
    from pathlib import Path
    
    dirs = [
        'data/raw', 'data/processed',
        'models', 'results/figures', 'results/metrics',
        'logs', 'notebooks', 'configs'
    ]
    
    for dir_name in dirs:
        Path(project_dir, dir_name).mkdir(parents=True, exist_ok=True)
