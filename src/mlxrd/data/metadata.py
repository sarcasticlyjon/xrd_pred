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
            ParsingRule(
                r'^(?P<sample_number>\d+)\s+(?P<material>[A-Za-z0-9.]+)(?:\s+(?P<substrate>[A-Za-z0-9.]+))?',
                ['sample_number', 'material', 'substrate'],
                11,
            ),
            ParsingRule(r'^(?P<material>[A-Z][A-Za-z0-9.]+)_(?P<substrate>[a-z]+)_(?P<sample_number>\d+)', ['material', 'substrate', 'sample_number'], 10),
            ParsingRule(r'^(?P<material>[A-Z][A-Za-z0-9.]+)_(?P<sample_number>\d+)_(?P<substrate>[a-z]+)', ['material', 'sample_number', 'substrate'], 9),
            ParsingRule(r'^(?P<material>[A-Z][A-Za-z0-9.]+)_(?P<substrate>[a-z]+)', ['material', 'substrate'], 8),
            ParsingRule(r'^(?P<material>[A-Z][A-Za-z0-9.]+)', ['material'], 1),
        ]
    
    def parse(self, filename: str) -> Optional[Dict]:
        self.stats['total'] += 1
        name = filename.rsplit('.', 1)[0]

        parsed = parse_filename_safe(filename)
        if parsed.get("is_valid"):
            self.stats['success'] += 1
            parsed.pop("is_valid", None)
            parsed.pop("skip_reason", None)
            parsed.pop("filename", None)
            return parsed

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

    normalized = re.sub(r'\s+', ' ', text)
    normalized = normalized.replace(' = ', '=').replace(' / ', '/')

    legacy_match = re.search(r'([A-Za-z/]+)\s*=\s*([\d.]+)\s*/\s*([\d.]+)', text)
    if legacy_match:
        gases = legacy_match.group(1).split("/")
        ratios = [float(legacy_match.group(2)), float(legacy_match.group(3))]
        if len(gases) == 2 and sum(ratios) > 0:
            total = sum(ratios)
            return {
                f"{gases[0]}_percent": ratios[0] / total * 100,
                f"{gases[1]}_percent": ratios[1] / total * 100,
            }

    legacy_match = re.search(r'([A-Za-z/]+)\s*=\s*([\d.]+)\s*/\s*([\d.]+)', normalized)
    if legacy_match:
        gases = legacy_match.group(1).split("/")
        ratios = [float(legacy_match.group(2)), float(legacy_match.group(3))]
        if len(gases) == 2 and sum(ratios) > 0:
            total = sum(ratios)
            return {
                f"{gases[0]}_percent": ratios[0] / total * 100,
                f"{gases[1]}_percent": ratios[1] / total * 100,
            }

    parts = text.split()
    gases = parts[0].replace("/", ":").split(":")
    if len(parts) == 1:
        if len(gases) == 1:
            return {f"{gases[0]}_percent": 100.0}
        return {}

    if len(parts) >= 3 and parts[1] == '=':
        text = f"{parts[0]} {parts[2]}"
        parts = text.split()
        gases = parts[0].split(":")

    ratio_token = parts[1]
    if ratio_token.startswith('='):
        ratio_token = ratio_token.lstrip('=')
    if "/" in ratio_token:
        ratios = [float(item) for item in ratio_token.split("/")]
    else:
        ratios = [float(item) for item in ratio_token.split(":")]
    if len(ratios) != len(gases) or sum(ratios) <= 0:
        return {}

    total = sum(ratios)
    return {f"{gas}_percent": ratio / total * 100 for gas, ratio in zip(gases, ratios)}


def _match_sample_number(token: str) -> bool:
    return bool(re.match(r'^\d{4}$|^B-\d+(?:-\d+)?$|^\d+$', token))


def parse_filename(filename: str) -> Dict:
    name = filename.rsplit(".", 1)[0]
    _reject_invalid_patterns(name)

    tokens = [t for t in re.split(r'[\s_]+', name) if t]
    if not tokens:
        raise ValueError("Пустое имя файла")

    data = {
        "sample_number": None,
        "material": None,
        "substrate": None,
        "annealed": False,
        "extra": None,
        "is_valid": False,
    }

    if tokens and _match_sample_number(tokens[0]):
        data["sample_number"] = tokens.pop(0)
    elif tokens and _match_sample_number(tokens[-1]):
        data["sample_number"] = tokens.pop(-1)

    if tokens:
        material_match = re.match(r'[A-Za-z][A-Za-z0-9.()]*', tokens[0])
        if material_match:
            data["material"] = tokens.pop(0)

    if tokens and tokens[0].isdigit() is False:
        data["substrate"] = _normalize_substrate(tokens.pop(0))

    remaining = []
    for token in tokens:
        if "anneal" in token.lower():
            data["annealed"] = True
            cleaned = token.lower().replace("anneal", "").strip()
            if cleaned:
                remaining.append(cleaned)
        else:
            remaining.append(token)

    data["extra"] = "_".join(remaining).strip() if remaining else None

    if data["sample_number"] and data["material"]:
        data["is_valid"] = True

    return data


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
