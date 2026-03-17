"""Módulo de validadores de documentos brasileiros.

Este módulo fornece funções de validação para diversos tipos de documentos
oficiais brasileiros, incluindo CPF, CNPJ, CNH, e Título de Eleitor.
"""

# Import context validators to trigger registration
import anonimizar._validators.context  # noqa: F401
from anonimizar._validators.documents import (
    valida_cnh,
    valida_cnpj,
    valida_cpf,
    valida_titulo_eleitor,
)
from anonimizar._validators.registry import (
    VALIDATOR_REGISTRY,
    get_validator,
    register_validator,
)
from anonimizar._validators.unified import verify_entities_unified

__all__ = [
    "VALIDATOR_REGISTRY",
    "get_validator",
    "register_validator",
    "valida_cnh",
    "valida_cnpj",
    "valida_cpf",
    "valida_titulo_eleitor",
    "verify_entities_unified",
]
