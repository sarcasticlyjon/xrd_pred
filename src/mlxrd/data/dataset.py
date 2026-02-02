"""Dataset Builder v2.0 - ALL NEW FEATURES"""
import pandas as pd, numpy as np, logging, json
from pathlib import Path
from typing import Optional, List, Dict, Iterable
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from .metadata import MetadataExtractor, parse_filename_safe, parse_gas_ratio
from .xrd import XRDSpectrum
from ..preprocessing.chemistry import add_element_columns

logger = logging.getLogger(__name__)

@dataclass
class BuildReport:
    """НОВОЕ: Детальный отчёт"""
    total_files: int
    successful: int
    duplicates_found: int
    final_samples: int
    failed_files: List[Dict]
    parsing_stats: Dict

    @property
    def failed(self) -> int:
        return len(self.failed_files)
    
    def __str__(self):
        return f"""
╔══════════════════════════════════════════════╗
║       DATASET BUILD REPORT v2.0              ║
╠══════════════════════════════════════════════╣
║ Total files:        {self.total_files:5d}                    ║
║ Successful:         {self.successful:5d} ({self._rate(self.successful):.1f}%)         ║
║ Failed:             {self.failed:5d} ({self._rate(self.failed):.1f}%)         ║
║ Duplicates removed: {self.duplicates_found:5d}                    ║
║ Final samples:      {self.final_samples:5d}                    ║
╠══════════════════════════════════════════════╣
║ Parsing success: {self.parsing_stats.get('success_rate', 'N/A')}                ║
╚══════════════════════════════════════════════╝"""

    def _rate(self, value: int) -> float:
        if self.total_files == 0:
            return 0.0
        return value / self.total_files * 100

class XRDDatasetBuilder:
    """
    НОВОЕ v2.0:
    - ✅ Progress bars (tqdm)
    - ✅ Parallel processing
    - ✅ Duplicate detection
    - ✅ Failed files logging
    - ✅ Multiple formats
    """
    def __init__(
        self,
        xrd_folder,
        metadata_file=None,
        verbose=True,
        recovery_mode=True,
        n_jobs=1,
        show_progress=True,
        pointwise=False,
        sap_intensity_threshold=7000.0,
    ):
        self.xrd_folder = Path(xrd_folder)
        self.metadata_file = metadata_file
        self.verbose = verbose
        self.recovery_mode = recovery_mode
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        self.pointwise = pointwise
        self.sap_intensity_threshold = sap_intensity_threshold
        self.extractor = MetadataExtractor()
        self.failed_files = []
        
        if not self.xrd_folder.exists():
            raise ValueError(f"Folder not found: {self.xrd_folder}")
    
    def _process_file(self, filepath):
        """НОВОЕ: Обработка с логированием ошибок"""
        try:
            spectrum = XRDSpectrum.from_file(str(filepath), self.recovery_mode)
            if not spectrum or not spectrum.is_valid:
                return {
                    'result': None,
                    'error': {'filename': filepath.name, 'reason': 'Invalid spectrum'},
                    'parsed': False,
                }
            
            metadata = self.extractor.parse(filepath.name)
            if not metadata:
                return {
                    'result': None,
                    'error': {'filename': filepath.name, 'reason': 'Parse failed'},
                    'parsed': False,
                }
            
            return {
                'result': {
                    **metadata,
                    'filename': filepath.name,
                    'n_points': len(spectrum.angles),
                    'intensity_mean': float(spectrum.intensities.mean()),
                    'intensity_std': float(spectrum.intensities.std()),
                },
                'error': None,
                'parsed': True,
            }
        except Exception as e:
            return {
                'result': None,
                'error': {'filename': filepath.name, 'reason': str(e)},
                'parsed': False,
            }
    
    def build(self, return_report=False):
        """НОВОЕ: return_report для статистики"""
        if self.pointwise or (self.metadata_file and str(self.metadata_file).lower().endswith('.xlsx')):
            point_builder = XRDPointDatasetBuilder(
                xrd_folder=str(self.xrd_folder),
                metadata_xlsx=str(self.metadata_file) if self.metadata_file else None,
                sap_intensity_threshold=self.sap_intensity_threshold,
            )
            return point_builder.build()
        files = list(self.xrd_folder.glob('*.txt'))
        if not files:
            raise ValueError(f"No .txt files in {self.xrd_folder}")
        
        if self.verbose:
            print(f"Found {len(files)} files")
        
        results = []
        failed_files = []
        parsing_total = len(files)
        parsing_success = 0
        
        # НОВОЕ: Parallel processing
        if self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {executor.submit(self._process_file, f): f for f in files}
                iterator = tqdm(as_completed(futures), total=len(files), desc="Building") if self.show_progress else as_completed(futures)
                for future in iterator:
                    outcome = future.result()
                    if outcome['result']:
                        results.append(outcome['result'])
                    if outcome['error']:
                        failed_files.append(outcome['error'])
                    if outcome['parsed']:
                        parsing_success += 1
        else:
            iterator = tqdm(files, desc="Building") if self.show_progress else files
            for f in iterator:
                outcome = self._process_file(f)
                if outcome['result']:
                    results.append(outcome['result'])
                if outcome['error']:
                    failed_files.append(outcome['error'])
                if outcome['parsed']:
                    parsing_success += 1

        parsing_failed = parsing_total - parsing_success
        parsing_rate = (parsing_success / parsing_total * 100) if parsing_total > 0 else 0
        parsing_stats = {
            'total': parsing_total,
            'success': parsing_success,
            'failed': parsing_failed,
            'success_rate': f"{parsing_rate:.1f}%",
        }

        self.failed_files = failed_files
        
        df = pd.DataFrame(results)
        
        # НОВОЕ: Duplicate detection
        dups = 0
        if 'sample_number' in df.columns and 'material' in df.columns:
            substrate = df.get('substrate', 'unk')
            if hasattr(substrate, "fillna"):
                substrate = substrate.fillna('unk')
            df['sample_id'] = df['material'] + '_' + substrate + '_' + df['sample_number'].astype(str)
            dups = df['sample_id'].duplicated().sum()
            if dups > 0:
                if self.verbose: print(f"⚠️  Removing {dups} duplicates")
                df = df[~df['sample_id'].duplicated()]
        
        if return_report:
            report = BuildReport(
                len(files),
                len(results),
                dups,
                len(df),
                failed_files,
                parsing_stats,
            )
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


class XRDPointDatasetBuilder:
    """Build a point-wise dataset with 2theta/intensity + metadata."""
    def __init__(
        self,
        xrd_folder: str,
        metadata_xlsx: Optional[str] = None,
        sap_intensity_threshold: float = 7000.0,
    ):
        self.xrd_folder = Path(xrd_folder)
        self.metadata_xlsx = Path(metadata_xlsx) if metadata_xlsx else None
        self.sap_intensity_threshold = sap_intensity_threshold

    def _load_metadata(self) -> pd.DataFrame:
        if not self.metadata_xlsx:
            return pd.DataFrame()
        sheets = pd.ExcelFile(self.metadata_xlsx).sheet_names
        data_ex = pd.concat(
            [pd.read_excel(self.metadata_xlsx, sheet_name=s) for s in sheets],
            ignore_index=True,
        )
        data_ex = data_ex[
            ['number', 'target', 'substrate', 't_min', 'Т_mV', 'P_Pa', 'gas', 'Pwr']
        ].copy()
        data_ex['number'] = data_ex['number'].astype(str)

        gas_parsed = data_ex['gas'].apply(parse_gas_ratio)
        gas_df = pd.DataFrame(gas_parsed.tolist())
        data_ex = pd.concat([data_ex, gas_df], axis=1)
        data_ex = data_ex.rename(columns={'number': 'sample_number'})
        return data_ex

    def _collect_file_metadata(self, files: Iterable[Path]) -> pd.DataFrame:
        rows = []
        for filepath in files:
            parsed = parse_filename_safe(filepath.name)
            if parsed.get('is_valid'):
                rows.append({
                    'filename': filepath.name,
                    'sample_number': parsed.get('sample_number'),
                    'material': parsed.get('material'),
                    'substrate': parsed.get('substrate'),
                    'annealed': int(bool(parsed.get('annealed'))),
                })
        df_meta = pd.DataFrame(rows)
        if not df_meta.empty:
            df_meta['substrate'] = self._normalize_substrate(df_meta['substrate'])
        return df_meta

    def _apply_duplicate_suffix(self, df_meta: pd.DataFrame) -> pd.DataFrame:
        if df_meta.empty:
            return df_meta
        df_meta = df_meta.copy()
        df_meta['dup_index'] = df_meta.groupby(
            ['sample_number', 'substrate', 'annealed'],
            dropna=False,
        ).cumcount() + 1
        def _format_sample_number(row: pd.Series) -> str:
            sample = str(row['sample_number'])
            if '-' in sample and not sample.endswith('-1') and row['dup_index'] == 1:
                return sample
            return f"{sample}-{int(row['dup_index'])}"
        df_meta['sample_number'] = df_meta.apply(_format_sample_number, axis=1)
        df_meta = df_meta.drop(columns=['dup_index'])
        return df_meta

    def _normalize_substrate(self, series: pd.Series) -> pd.Series:
        replacements = {
            'sapp': 'sap',
            'sic': 'SiC',
            'sapp-Pt': 'Pt-sap',
        }
        series = series.replace(replacements)
        prefixes_swap = {'BST', 'Cu', 'Au', 'Pt', 'VO2'}
        prefixes_no_swap = {'Si', 'SiC'}
        normalized = series.copy()
        for idx, value in series.items():
            if isinstance(value, str) and '-' in value:
                parts = value.split('-')
                if len(parts) == 2:
                    first, second = parts
                    if first in prefixes_swap:
                        normalized.loc[idx] = f"{second}-{first}"
                    elif first in prefixes_no_swap or second in prefixes_no_swap:
                        normalized.loc[idx] = f"{first}-{second}"
        normalized = normalized.replace({
            'alum-Au': 'Au-alum',
            'sap-Pt': 'Pt-sap',
            'alum-Pt': 'Pt-alum',
            'Si-SiO2': 'SiO2-Si',
            'SiC-Pt': 'Pt-SiC',
            'sap-VO2': 'VO2-sap',
            'sap-BST': 'BST-sap',
            'sap-Cu': 'Cu-sap',
            'alum-VO2': 'VO2-alum',
            'SiO2-VO2': 'VO2-SiO2',
            'Si-VO2': 'VO2-Si',
            'SiC-VO2': 'VO2-SiC',
            'SiC-BST': 'BST-SiC',
            'alum-BST': 'BST-alum',
        })
        return normalized

    def _load_spectrum(self, filepath: Path) -> Optional[pd.DataFrame]:
        data = pd.read_csv(
            filepath,
            sep=r'\s+',
            header=None,
            names=['2thetta', 'intensity'],
            comment='#',
            engine='python',
            encoding='utf-8',
            encoding_errors='replace',
            on_bad_lines='skip',
        )
        data['2thetta'] = pd.to_numeric(data['2thetta'], errors='coerce')
        data['intensity'] = pd.to_numeric(data['intensity'], errors='coerce')
        data = data.dropna()
        if data.empty:
            return None
        return data

    def build(
        self,
        equipment_map: Optional[Dict] = None,
        annotate_sap1: bool = True,
    ) -> pd.DataFrame:
        files = list(self.xrd_folder.glob('*.txt'))
        df_meta = self._collect_file_metadata(files)
        data_ex = self._load_metadata()
        if not data_ex.empty:
            df_meta = df_meta[df_meta['sample_number'].isin(data_ex['sample_number'])]
            df_meta = df_meta.merge(
                data_ex,
                on='sample_number',
                how='left',
                suffixes=('', '_meta'),
            )
        df_meta = self._apply_duplicate_suffix(df_meta)

        combined = []
        for row in df_meta.itertuples(index=False):
            filepath = self.xrd_folder / row.filename
            spectrum = self._load_spectrum(filepath)
            if spectrum is None:
                continue
            spectrum = spectrum.copy()
            spectrum['sample_number'] = row.sample_number
            for col in df_meta.columns:
                if col not in spectrum.columns and col != 'filename':
                    spectrum[col] = getattr(row, col)
            combined.append(spectrum)

        if not combined:
            return pd.DataFrame()

        data = pd.concat(combined, ignore_index=True)
        if 'substrate_x' in data.columns:
            data = data.rename(columns={'substrate_x': 'substrate'})
        if 'substrate_y' in data.columns:
            data = data.drop(columns=['substrate_y'])
        if 'substrate_meta' in data.columns:
            data = data.drop(columns=['substrate_meta'])
        if annotate_sap1 and 'substrate' in data.columns:
            sap_mask = (data['substrate'] == 'sap') & (data['intensity'] > self.sap_intensity_threshold)
            sap_samples = data.loc[sap_mask, 'sample_number'].unique()
            data.loc[data['sample_number'].isin(sap_samples), 'substrate'] = 'sap1'

        if equipment_map:
            if all(isinstance(key, int) for key in equipment_map.keys()):
                reversed_map = {value: key for key, value in equipment_map.items()}
                equipment_map = reversed_map
            data['equipment_number'] = data['sample_number'].map(equipment_map)

        if 'target' in data.columns:
            data = add_element_columns(data, formula_col='target')
            element_map = {
                'element_Ba': 'Ba',
                'element_Sr': 'Sr',
                'element_Ti': 'Ti',
                'element_Zr': 'Zr',
                'element_Sn': 'Sn',
            }
            data = data.rename(columns=element_map)
            drop_cols = [
                col for col in data.columns
                if col.startswith('element_') and col not in element_map.values()
            ]
            drop_cols += [
                col for col in ['avg_mass', 'avg_electronegativity', 'avg_radius', 'n_elements']
                if col in data.columns
            ]
            if drop_cols:
                data = data.drop(columns=drop_cols)

        if 'Ar_percent' in data.columns:
            data = data.rename(columns={'Ar_percent': 'Ar_per'})
        if 'O2_percent' in data.columns:
            data = data.rename(columns={'O2_percent': 'O_per'})
        if 'gas' in data.columns:
            data = data.drop(columns=['gas'])

        return data
