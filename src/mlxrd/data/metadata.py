"""Metadata Extraction v2.0"""
import re, logging
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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

def parse_filename(filename: str):
    return MetadataExtractor().parse(filename)
