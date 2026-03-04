"""
TEK17 corpus extraction subpackage.

Handles downloading and parsing the authoritative TEK17 root-print
snapshot from DiBK into structured per-provision JSONL records.
"""

from .download import DEFAULT_ROOT_PRINT_URL, run_download_root_print
from .parse import run_parse_root_print

__all__ = [
    "DEFAULT_ROOT_PRINT_URL",
    "run_download_root_print",
    "run_parse_root_print",
]
