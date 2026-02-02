"""Metadata Extraction v2.0"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ELECTRODES = {"Pt", "Au", "Ag", "Ti", "Al"}
SPECIAL_SUBSTRATES = {"sic": "SiC"}

@dataclass
class ParsingRule:
    pattern: str
    fields: List[str]
    priority: int = 0

class MetadataExtractor:
    """НОВОЕ v2.0: Custom rules, validation, statistics"""
    def __init__(self):
        self.rules = []
        self.stats = {'total': 0, 'success': 0, 'failed': 0}
        self._load_defaults()
    
    def _load_defaults(self):
        self.rules = [
            ParsingRule(r'^(?P<material>[A-Z][A-Za-z0-9.]+)_(?P<substrate>[a-z]+)_(?P<sample_number>\d+)', ['material', 'substrate', 'sample_number'], 10),
            ParsingRule(r'^(?P<material>[A-Z][A-Za-z0-9.]+)_(?P<sample_number>\d+)_(?P<substrate>[a-z]+)', ['material', 'sample_number', 'substrate'], 9),
            ParsingRule(r'^(?P<material>[A-Z][A-Za-z0-9.]+)_(?P<substrate>[a-z]+)', ['material', 'substrate'], 8),
            ParsingRule(r'^(?P<material>[A-Z][A-Za-z0-9.]+)', ['material'], 1),
        ]
    
    def parse(self, filename: str) -> Optional[Dict]:
        self.stats['total'] += 1
        name = filename.rsplit('.', 1)[0]
        for rule in self.rules:
            match = re.match(rule.pattern, name)
            if match:
                self.stats['success'] += 1
                return match.groupdict()
        self.stats['failed'] += 1
        return None
    
    def get_stats(self):
        rate = (self.stats['success']/self.stats['total']*100) if self.stats['total'] > 0 else 0
        return {**self.stats, 'success_rate': f"{rate:.1f}%"}


def _normalize_substrate(token: str) -> str:
    if token.lower() in SPECIAL_SUBSTRATES:
        return SPECIAL_SUBSTRATES[token.lower()]
    return token


def _reject_invalid_patterns(name: str) -> None:
    if re.match(r"^\d{1,2}\.\d{1,2}", name):
        raise ValueError("Неверный формат даты в имени файла")
    if re.match(r"^\d+[A-Za-z].*_\d+[A-Za-z]", name):
        raise ValueError("Неверный композитный формат имени файла")


def _extract_scan_number(tokens: List[str]) -> Optional[int]:
    for token in reversed(tokens):
        if token.isdigit():
            return int(token)
    return None


def parse_gas_ratio(value: Optional[str]) -> Dict[str, float]:
    if not value:
        return {}
    text = value.strip()
    if not text:
        return {}

    parts = text.split()
    gases = parts[0].split(":")
    if len(parts) == 1:
        if len(gases) == 1:
            return {f"{gases[0]}_percent": 100.0}
        return {}

    ratios = [float(item) for item in parts[1].split(":")]
    if len(ratios) != len(gases) or sum(ratios) <= 0:
        return {}

    total = sum(ratios)
    return {f"{gas}_percent": ratio / total * 100 for gas, ratio in zip(gases, ratios)}


def parse_filename(filename: str) -> Dict:
    name = filename.rsplit(".", 1)[0]
    _reject_invalid_patterns(name)

    tokens = [token for token in name.split("_") if token]
    if len(tokens) < 3:
        raise ValueError("Недостаточно данных для парсинга имени файла")

    sample_id = tokens[0]
    material = tokens[1]
    index = 2

    while index < len(tokens) and tokens[index].isdigit():
        index += 1

    electrode = None
    if index < len(tokens) and tokens[index] in ELECTRODES:
        electrode = tokens[index]
        index += 1

    if index >= len(tokens):
        raise ValueError("Отсутствует подложка в имени файла")

    substrate = _normalize_substrate(tokens[index])
    index += 1

    remaining = tokens[index:]
    has_annealing = "anneal" in remaining
    annealing_temp = None
    annealing_time = None

    processing_tokens: List[str] = []
    if has_annealing:
        anneal_index = remaining.index("anneal")
        processing_tokens.extend([t for t in remaining[:anneal_index] if not t.isdigit()])
        after_anneal = remaining[anneal_index + 1 :]
        if len(after_anneal) >= 2 and after_anneal[0].isdigit() and after_anneal[1].isdigit():
            annealing_temp = int(after_anneal[0])
            annealing_time = int(after_anneal[1])
            after_anneal = after_anneal[2:]
        processing_tokens.extend([t for t in after_anneal if not t.isdigit()])
        scan_number = _extract_scan_number(after_anneal)
    else:
        processing_tokens.extend([t for t in remaining if not t.isdigit() and t != "anneal"])
        scan_number = _extract_scan_number(remaining)

    processing_notes = " ".join(processing_tokens) if processing_tokens else None

    return {
        "sample_id": sample_id,
        "material": material,
        "electrode": electrode,
        "substrate": substrate,
        "has_annealing": has_annealing,
        "annealing_temp": annealing_temp,
        "annealing_time": annealing_time,
        "processing_notes": processing_notes,
        "scan_number": scan_number,
        "is_valid": True,
    }


def validate_filename(filename: str) -> bool:
    try:
        parse_filename(filename)
    except ValueError:
        return False
    return True


def parse_filename_safe(filename: str) -> Dict:
    try:
        return parse_filename(filename)
    except ValueError:
        return {"filename": filename, "is_valid": False, "skip_reason": "parse_error"}
