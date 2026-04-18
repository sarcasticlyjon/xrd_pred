"""
Тесты для модуля metadata
"""
import pytest
from mlxrd.data.metadata import parse_filename, parse_gas_ratio, validate_filename, parse_filename_safe


class TestParseFilename:
    """Тесты для parse_filename"""
    
    def test_parse_simple_format(self):
        """Тест базового формата без электрода и отжига"""
        result = parse_filename("1993_BST_sap_4.txt")
        
        assert result['sample_number'] == '1993'
        assert result['material'] == 'BST'
        assert result['substrate'] == 'sap'
        assert result['annealed'] is False
        assert result['extra'] == '4'
        assert result['is_valid'] is True
    
    def test_parse_with_electrode_and_anneal(self):
        """Тест файла с дополнительными токенами и отжигом"""
        result = parse_filename("2198_BST_Pt_alum_anneal_8.txt")
        
        assert result['sample_number'] == '2198'
        assert result['material'] == 'BST'
        assert result['substrate'] == 'Pt'
        assert result['annealed'] is True
        assert result['extra'] == 'alum_8'
        assert result['is_valid'] is True
    
    def test_parse_b_series(self):
        """Тест B-серии"""
        result = parse_filename("B-374_BSnT_alum_laser_4_2_4.txt")
        
        assert result['sample_number'] == 'B-374'
        assert result['material'] == 'BSnT'
        assert result['substrate'] == 'alum'
        assert result['extra'] == 'laser_4_2_4'
        assert result['is_valid'] is True
    
    def test_parse_with_anneal_temp_time(self):
        """Тест отжига и сохранения остаточных токенов в extra"""
        result = parse_filename("2476_SBN_sap_anneal_1150_60_4.txt")
        
        assert result['sample_number'] == '2476'
        assert result['material'] == 'SBN'
        assert result['substrate'] == 'sap'
        assert result['annealed'] is True
        assert result['extra'] == '1150_60_4'
        assert result['is_valid'] is True
    
    def test_parse_with_processing_notes(self):
        """Тест обработки дополнительных меток в extra"""
        result = parse_filename("2347_BZT_SiC_anneal_other_side_8.txt")
        
        assert result['sample_number'] == '2347'
        assert result['material'] == 'BZT'
        assert result['substrate'] == 'SiC'
        assert result['annealed'] is True
        assert result['extra'] == 'other_side_8'
        assert result['is_valid'] is True
    
    def test_parse_complex_material(self):
        """Тест сложного названия материала"""
        result = parse_filename("2328_BSrZrTi_0_5__sic_anneal_4.txt")
        
        assert result['sample_number'] == '2328'
        assert result['material'] == 'BSrZrTi'
        assert result['substrate'] is None
        assert result['annealed'] is True
        assert result['extra'] == '0_5_sic_4'
        assert result['is_valid'] is True
    
    def test_invalid_date_format(self):
        """Тест отклонения файлов с датой"""
        with pytest.raises(ValueError, match="дат"):
            parse_filename("09.06_1_VO2_sapp_4.txt")
    
    def test_invalid_composite_format(self):
        """Тест отклонения композитного формата"""
        with pytest.raises(ValueError, match="композит"):
            parse_filename("30BTO_20SiO2_50PT_340_4.txt")
    
    def test_safe_parse_invalid(self):
        """Тест безопасного парсинга невалидного файла"""
        result = parse_filename_safe("09.06_1_VO2_sapp_4.txt")
        
        assert result['is_valid'] is False
        assert result['skip_reason'] == 'parse_error'


class TestParseGasRatio:
    """Тесты для parse_gas_ratio"""
    
    def test_parse_ratio(self):
        """Тест парсинга соотношения газов"""
        result = parse_gas_ratio("Ar:O2 9:1")
        
        assert result['Ar_percent'] == 90.0
        assert result['O2_percent'] == 10.0
    
    def test_parse_single_gas(self):
        """Тест парсинга одного газа"""
        result = parse_gas_ratio("Ar")
        
        assert result['Ar_percent'] == 100.0
    
    def test_parse_empty(self):
        """Тест парсинга пустой строки"""
        result = parse_gas_ratio("")
        assert result == {}
        
        result = parse_gas_ratio(None)
        assert result == {}
    
    def test_parse_three_gases(self):
        """Тест парсинга трёх газов"""
        result = parse_gas_ratio("Ar:O2:N2 6:3:1")
        
        assert result['Ar_percent'] == 60.0
        assert result['O2_percent'] == 30.0
        assert result['N2_percent'] == 10.0


class TestValidateFilename:
    """Тесты для validate_filename"""
    
    def test_valid_filename(self):
        """Проверка валидного имени"""
        assert validate_filename("2198_BST_Pt_alum_anneal_8.txt") is True
    
    def test_invalid_filename(self):
        """Проверка невалидного имени"""
        assert validate_filename("09.06_1_VO2_sapp_4.txt") is False
        assert validate_filename("30BTO_20SiO2_50PT_340_4.txt") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
