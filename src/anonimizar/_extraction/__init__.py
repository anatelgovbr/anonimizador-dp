"""Módulo de extração de entidades.

Este módulo fornece funções para extrair entidades sensíveis de diferentes
formatos de texto, incluindo tabelas markdown e texto regular.
"""

from anonimizar._extraction.markdown import (
    extract_entities_from_markdown_tables,
)
from anonimizar._extraction.model import extract_from_model
from anonimizar._extraction.pipeline import extract_entities
from anonimizar._extraction.regex import extract_entities_regex_re

__all__ = [
    "extract_entities",
    "extract_entities_from_markdown_tables",
    "extract_entities_regex_re",
    "extract_from_model",
]
