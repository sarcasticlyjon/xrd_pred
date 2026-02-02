"""
Тесты для модуля chemistry
"""
from mlxrd.preprocessing.chemistry import parse_formula


def test_parse_formula_decimal_not_hydrate_split():
    """Не разрезать десятичные дроби при разборе формул."""
    result = parse_formula("Ba0.5Sr0.5TiO3")

    assert result == {"Ba": 0.5, "Sr": 0.5, "Ti": 1.0, "O": 3.0}
