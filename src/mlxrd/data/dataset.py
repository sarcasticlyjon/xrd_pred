"""Dataset Builder v2.0 - ALL NEW FEATURES"""
import pandas as pd, numpy as np, logging, json
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from .metadata import MetadataExtractor
from .xrd import XRDSpectrum

logger = logging.getLogger(__name__)

@dataclass
class BuildReport:
    """НОВОЕ: Детальный отчёт"""
    total_files: int
    successful: int
    failed: int
    duplicates_found: int
    final_samples: int
    failed_files: List[Dict]
    parsing_stats: Dict
    
    def __str__(self):
        return f"""
╔══════════════════════════════════════════════╗
║       DATASET BUILD REPORT v2.0              ║
╠══════════════════════════════════════════════╣
║ Total files:        {self.total_files:5d}                    ║
║ Successful:         {self.successful:5d} ({self.successful/self.total_files*100:.1f}%)         ║
║ Failed:             {self.failed:5d} ({self.failed/self.total_files*100:.1f}%)         ║
║ Duplicates removed: {self.duplicates_found:5d}                    ║
║ Final samples:      {self.final_samples:5d}                    ║
╠══════════════════════════════════════════════╣
║ Parsing success: {self.parsing_stats.get('success_rate', 'N/A')}                ║
╚══════════════════════════════════════════════╝"""

class XRDDatasetBuilder:
    """
    НОВОЕ v2.0:
    - ✅ Progress bars (tqdm)
    - ✅ Parallel processing
    - ✅ Duplicate detection
    - ✅ Failed files logging
    - ✅ Multiple formats
    """
    def __init__(self, xrd_folder, metadata_file=None, verbose=True, recovery_mode=True, n_jobs=1, show_progress=True):
        self.xrd_folder = Path(xrd_folder)
        self.metadata_file = metadata_file
        self.verbose = verbose
        self.recovery_mode = recovery_mode
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        self.extractor = MetadataExtractor()
        self.failed_files = []
        
        if not self.xrd_folder.exists():
            raise ValueError(f"Folder not found: {self.xrd_folder}")
    
    def _process_file(self, filepath):
        """НОВОЕ: Обработка с логированием ошибок"""
        try:
            spectrum = XRDSpectrum.from_file(str(filepath), self.recovery_mode)
            if not spectrum or not spectrum.is_valid:
                self.failed_files.append({'filename': filepath.name, 'reason': 'Invalid spectrum'})
                return None
            
            metadata = self.extractor.parse(filepath.name)
            if not metadata:
                self.failed_files.append({'filename': filepath.name, 'reason': 'Parse failed'})
                return None
            
            return {**metadata, 'filename': filepath.name, 'n_points': len(spectrum.angles),
                    'intensity_mean': float(spectrum.intensities.mean()), 'intensity_std': float(spectrum.intensities.std())}
        except Exception as e:
            self.failed_files.append({'filename': filepath.name, 'reason': str(e)})
            return None
    
    def build(self, return_report=False):
        """НОВОЕ: return_report для статистики"""
        files = list(self.xrd_folder.glob('*.txt'))
        if not files:
            raise ValueError(f"No .txt files in {self.xrd_folder}")
        
        if self.verbose:
            print(f"Found {len(files)} files")
        
        results = []
        
        # НОВОЕ: Parallel processing
        if self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {executor.submit(self._process_file, f): f for f in files}
                iterator = tqdm(as_completed(futures), total=len(files), desc="Building") if self.show_progress else as_completed(futures)
                for future in iterator:
                    result = future.result()
                    if result: results.append(result)
        else:
            iterator = tqdm(files, desc="Building") if self.show_progress else files
            for f in iterator:
                result = self._process_file(f)
                if result: results.append(result)
        
        df = pd.DataFrame(results)
        
        # НОВОЕ: Duplicate detection
        dups = 0
        if 'sample_number' in df.columns and 'material' in df.columns:
            df['sample_id'] = df['material'] + '_' + df.get('substrate', 'unk') + '_' + df['sample_number'].astype(str)
            dups = df['sample_id'].duplicated().sum()
            if dups > 0:
                if self.verbose: print(f"⚠️  Removing {dups} duplicates")
                df = df[~df['sample_id'].duplicated()]
        
        if return_report:
            report = BuildReport(len(files), len(results), len(self.failed_files), dups, len(df), self.failed_files, self.extractor.get_stats())
            if self.verbose: print(report)
            return df, report
        
        return df
    
    def save_failed_log(self, path='failed_files.json'):
        """НОВОЕ: Сохранить лог failed files"""
        with open(path, 'w') as f:
            json.dump(self.failed_files, f, indent=2)
        print(f"✅ Failed log: {path}")
    
    def export(self, df, path, format='auto'):
        """НОВОЕ: Multiple formats"""
        path = Path(path)
        if format == 'auto':
            format = 'parquet' if path.suffix == '.parquet' else 'csv'
        
        if format == 'csv': df.to_csv(path, index=False)
        elif format == 'parquet': df.to_parquet(path, index=False)
        
        size = path.stat().st_size / 1024 / 1024
        print(f"✅ Exported: {path} ({size:.2f} MB, {format})")
