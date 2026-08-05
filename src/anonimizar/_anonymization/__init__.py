"""Módulo de anonimização de texto.

Este módulo fornece funções para substituir entidades sensíveis detectadas
no texto por tags de anonimização.
"""

from anonimizar._anonymization.text import anonymize_text

__all__ = ["anonymize_text"]
