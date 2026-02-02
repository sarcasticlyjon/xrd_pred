"""
Chemistry v2.0

НОВОЕ:
- ✅ Поддержка скобок Ba(Ti0.8Zr0.2)O3
- ✅ Гидраты CuSO4·5H2O
- ✅ Элементные признаки
"""

import re
import pandas as pd
from typing import Dict, Optional


# Элементы и их свойства
ELEMENT_PROPERTIES = {
    'H': {'mass': 1.008, 'electronegativity': 2.20, 'radius': 53},
    'C': {'mass': 12.011, 'electronegativity': 2.55, 'radius': 67},
    'N': {'mass': 14.007, 'electronegativity': 3.04, 'radius': 56},
    'O': {'mass': 15.999, 'electronegativity': 3.44, 'radius': 48},
    'Ba': {'mass': 137.327, 'electronegativity': 0.89, 'radius': 253},
    'Sr': {'mass': 87.62, 'electronegativity': 0.95, 'radius': 219},
    'Ti': {'mass': 47.867, 'electronegativity': 1.54, 'radius': 176},
    'Zr': {'mass': 91.224, 'electronegativity': 1.33, 'radius': 206},
    'Al': {'mass': 26.982, 'electronegativity': 1.61, 'radius': 184},
    'Si': {'mass': 28.086, 'electronegativity': 1.90, 'radius': 146},
    'Ca': {'mass': 40.078, 'electronegativity': 1.00, 'radius': 231},
    'Mg': {'mass': 24.305, 'electronegativity': 1.31, 'radius': 173},
    'Cu': {'mass': 63.546, 'electronegativity': 1.90, 'radius': 145},
    'S': {'mass': 32.065, 'electronegativity': 2.58, 'radius': 88},
}


def parse_formula(formula: str, handle_brackets=True) -> Dict[str, float]:
    """
    УЛУЧШЕНО v2.0: поддержка скобок и гидратов
    
    Examples:
        >>> parse_formula('Ba0.5Sr0.5TiO3')
        {'Ba': 0.5, 'Sr': 0.5, 'Ti': 1.0, 'O': 3.0}
        
        >>> parse_formula('Ba(Ti0.8Zr0.2)O3')  # NEW!
        {'Ba': 1.0, 'Ti': 0.8, 'Zr': 0.2, 'O': 3.0}
        
        >>> parse_formula('CuSO4·5H2O')  # NEW!
        {'Cu': 1.0, 'S': 1.0, 'O': 9.0, 'H': 10.0}
    """
    # Обработка гидратов
    if '·' in formula or '.' in formula:
        # CuSO4·5H2O → CuSO4 + 5H2O
        parts = re.split(r'[·.]', formula)
        elements = {}
        for part in parts:
            part_elements = parse_formula(part, handle_brackets=False)
            for el, count in part_elements.items():
                elements[el] = elements.get(el, 0) + count
        return elements
    
    # Обработка скобок
    if handle_brackets and '(' in formula:
        # Ba(Ti0.8Zr0.2)O3 → разбираем
        elements = {}
        
        # Найти все скобки
        bracket_pattern = r'\(([^)]+)\)(\d*\.?\d*)'
        matches = re.finditer(bracket_pattern, formula)
        
        for match in matches:
            inner_formula = match.group(1)
            multiplier = float(match.group(2)) if match.group(2) else 1.0
            
            # Парсим внутри скобок
            inner_elements = parse_formula(inner_formula, handle_brackets=False)
            for el, count in inner_elements.items():
                elements[el] = elements.get(el, 0) + count * multiplier
            
            # Удаляем обработанную часть
            formula = formula.replace(match.group(0), '')
        
        # Парсим оставшееся
        remaining = parse_formula(formula, handle_brackets=False)
        for el, count in remaining.items():
            elements[el] = elements.get(el, 0) + count
        
        return elements
    
    # Базовый парсинг без скобок
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)
    
    elements = {}
    for element, count in matches:
        if element:
            count = float(count) if count else 1.0
            elements[element] = elements.get(element, 0) + count
    
    return elements


def get_element_features(elements: Dict[str, float]) -> Dict[str, float]:
    """
    НОВОЕ v2.0: Вычисление признаков из элементов
    
    Returns:
        features: avg_mass, avg_electronegativity, avg_radius, etc.
    """
    if not elements:
        return {}
    
    total_count = sum(elements.values())
    
    avg_mass = sum(
        elements.get(el, 0) * ELEMENT_PROPERTIES.get(el, {}).get('mass', 0)
        for el in elements
    ) / total_count
    
    avg_en = sum(
        elements.get(el, 0) * ELEMENT_PROPERTIES.get(el, {}).get('electronegativity', 0)
        for el in elements
    ) / total_count
    
    avg_radius = sum(
        elements.get(el, 0) * ELEMENT_PROPERTIES.get(el, {}).get('radius', 0)
        for el in elements
    ) / total_count
    
    return {
        'avg_mass': avg_mass,
        'avg_electronegativity': avg_en,
        'avg_radius': avg_radius,
        'n_elements': len(elements)
    }


def add_element_columns(df: pd.DataFrame, formula_col: str = 'material') -> pd.DataFrame:
    """
    УЛУЧШЕНО v2.0: добавляет элементные признаки
    """
    df = df.copy()
    
    # Парсим формулы
    parsed = df[formula_col].apply(parse_formula)
    
    # Все уникальные элементы
    all_elements = set()
    for elements in parsed:
        all_elements.update(elements.keys())
    
    # Добавляем колонки для элементов
    for element in sorted(all_elements):
        df[f'element_{element}'] = parsed.apply(lambda x: x.get(element, 0))
    
    # NEW: Добавляем вычисленные признаки
    features = parsed.apply(get_element_features)
    for feat in ['avg_mass', 'avg_electronegativity', 'avg_radius', 'n_elements']:
        df[feat] = features.apply(lambda x: x.get(feat, 0))
    
    return df
