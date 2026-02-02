"""XRD Spectrum v2.0 with Recovery Mode"""
import numpy as np, pandas as pd, logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class XRDSpectrum:
    """НОВОЕ v2.0: Recovery mode, validation"""
    def __init__(self, angles, intensities, filename=None):
        self.angles = np.array(angles)
        self.intensities = np.array(intensities)
        self.filename = filename
        self.is_valid = self._validate()
        self.validation_errors = []
    
    def _validate(self):
        if len(self.angles) != len(self.intensities): return False
        if len(self.angles) == 0: return False
        if not np.all(np.isfinite(self.angles)): return False
        if not np.all(np.isfinite(self.intensities)): return False
        return True
    
    @classmethod
    def from_file(cls, filepath, recovery_mode=True):
        try:
            data = pd.read_csv(
                filepath,
                sep=r'\s+',
                header=None,
                names=['angle', 'intensity'],
                comment='#',
                engine='python',
                encoding='utf-8',
                encoding_errors='replace',
                on_bad_lines='skip',
            )
            data['angle'] = pd.to_numeric(data['angle'], errors='coerce')
            data['intensity'] = pd.to_numeric(data['intensity'], errors='coerce')
            data = data.dropna()
            if data.empty:
                return None
            spec = cls(data['angle'].values, data['intensity'].values, Path(filepath).name)
            if not spec.is_valid and recovery_mode:
                return cls._recover(data, Path(filepath).name)
            return spec
        except Exception as e:
            logger.error(f"Error: {filepath} - {e}")
            return None
    
    @classmethod
    def _recover(cls, data, filename):
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 50: return None
        data = data.sort_values('angle').drop_duplicates('angle')
        data['intensity'] = data['intensity'].clip(lower=0)
        logger.info(f"✅ Recovered: {filename}")
        return cls(data['angle'].values, data['intensity'].values, filename)
