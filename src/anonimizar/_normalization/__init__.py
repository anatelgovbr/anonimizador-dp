"""Normalização de entidades (GT e predições).

Fornece funções offset-safe para remover prefixos (``RG:``, ``Latitude:``,
``CPF:``) e sufixos (``SSP/DF``, ``DETRAN/RJ``) de entidades detectadas,
ajustando os spans correspondentes.
"""

from anonimizar._normalization.normalize import (
    normalize_entity,
    remover_prefixo,
    remover_prefixo_sufixo,
    remover_sufixo,
)

__all__ = [
    "normalize_entity",
    "remover_prefixo",
    "remover_prefixo_sufixo",
    "remover_sufixo",
]
