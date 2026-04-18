"""Dataset builders for point-wise XRD datasets."""
import pandas as pd
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Iterable
from .metadata import parse_filename_safe, parse_gas_ratio
from ..preprocessing.chemistry import add_element_columns

logger = logging.getLogger(__name__)

class XRDDatasetBuilder:
    """Backward-compatible wrapper that now always builds point-wise datasets."""
    def __init__(
        self,
        xrd_folder,
        metadata_file=None,
        pointwise=False,
        sap_intensity_threshold=7000.0,
        **kwargs,
    ):
        self.xrd_folder = Path(xrd_folder)
        self.metadata_file = metadata_file
        self.pointwise = pointwise
        self.sap_intensity_threshold = sap_intensity_threshold
        self._unused_kwargs = kwargs
        if kwargs:
            ignored = ", ".join(sorted(kwargs.keys()))
            warnings.warn(
                f"XRDDatasetBuilder игнорирует устаревшие параметры: {ignored}. "
                "Используется только point-wise режим.",
                DeprecationWarning,
                stacklevel=2,
            )
        if pointwise is not False:
            warnings.warn(
                "Параметр `pointwise` устарел: XRDDatasetBuilder всегда работает в point-wise режиме.",
                DeprecationWarning,
                stacklevel=2,
            )
        
        if not self.xrd_folder.exists():
            raise ValueError(f"Folder not found: {self.xrd_folder}")
    
    def build(self, return_report=False, equipment_map: Optional[Dict] = None, annotate_sap1: bool = True):
        """Build point-wise dataset only.

        Args:
            return_report: Deprecated, no longer supported.
            equipment_map: Optional map for equipment annotation.
            annotate_sap1: Whether to relabel high-intensity sap samples to sap1.
        """
        if return_report:
            raise ValueError(
                "return_report больше не поддерживается: агрегированный режим удалён, "
                "используйте только point-wise сборку."
            )
        point_builder = XRDPointDatasetBuilder(
            xrd_folder=str(self.xrd_folder),
            metadata_xlsx=str(self.metadata_file) if self.metadata_file else None,
            sap_intensity_threshold=self.sap_intensity_threshold,
        )
        return point_builder.build(equipment_map=equipment_map, annotate_sap1=annotate_sap1)
    
    def export(self, df, path, format='auto'):
        """Export dataset to CSV or Parquet."""
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
