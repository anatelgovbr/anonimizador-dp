"""Módulo de padrões regex para detecção de entidades.

Este módulo fornece funções para registrar padrões regex embutidos
e customizados para detecção de diversos tipos de entidades sensíveis.
"""

from anonimizar._patterns.builtin import (
    PATTERN_ADDERS,
    add_pattern_cid,
    add_pattern_cnh,
    add_pattern_cpf,
    add_pattern_dados_bancarios,
    add_pattern_data_nascimento,
    add_pattern_email,
    add_pattern_endereco,
    add_pattern_geo_coord,
    add_pattern_passaporte,
    add_pattern_passaporte_est,
    add_pattern_rg,
    add_pattern_siape,
    add_pattern_telefone,
    add_pattern_titulo_eleitor,
)
from anonimizar._patterns.custom import add_custom_pattern
from anonimizar._patterns.registry import (
    add_apply_patterns,
    get_active_labels,
    list_patterns,
)

__all__ = [
    "PATTERN_ADDERS",
    "add_apply_patterns",
    "add_custom_pattern",
    "add_pattern_cid",
    "add_pattern_cnh",
    "add_pattern_cpf",
    "add_pattern_dados_bancarios",
    "add_pattern_data_nascimento",
    "add_pattern_email",
    "add_pattern_endereco",
    "add_pattern_geo_coord",
    "add_pattern_passaporte",
    "add_pattern_passaporte_est",
    "add_pattern_rg",
    "add_pattern_siape",
    "add_pattern_telefone",
    "add_pattern_titulo_eleitor",
    "get_active_labels",
    "list_patterns",
]
