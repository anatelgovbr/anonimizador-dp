"""Constantes de validação de documentos brasileiros.

Este módulo contém constantes numéricas usadas na validação de CPF, CNPJ,
CNH, Título de Eleitor, passaporte e outros documentos, eliminando magic numbers.
"""

# =============================================================================
# CONTAGEM DE DÍGITOS
# =============================================================================

CPF_DIGIT_COUNT = 11
CNPJ_DIGIT_COUNT = 14
TITULO_ELEITOR_DIGIT_COUNT = 12
CNH_DIGIT_COUNT = 11
CNH_ESPELHO_DIGIT_COUNT = 10

# =============================================================================
# LIMITES DE VALIDAÇÃO
# =============================================================================

MAX_UF_CODE = 28
DV_REMAINDER_THRESHOLD = 2
DV_MODULO = 10
MIN_CPF_LENGTH = 9
MAX_CPF_LENGTH = 16
MIN_PASSPORT_LENGTH = 6
MAX_PASSPORT_LENGTH = 9
MIN_GEO_COORD_LENGTH = 3
MIN_TABLE_PIPE_COUNT = 3
MIN_TABLE_LINES = 3
MIN_INIT_SAMPLE_SIZE = 50
MIN_SPLIT_EXAMPLES = 2
MAX_HOLDOUT_TEST_SIZE = 0.5

# =============================================================================
# CONSTANTES TEMPORAIS
# =============================================================================

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600

# =============================================================================
# PESOS DE VALIDAÇÃO
# =============================================================================

CNPJ_WEIGHTS_1 = [6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9]

__all__ = [
    "CNH_DIGIT_COUNT",
    "CNH_ESPELHO_DIGIT_COUNT",
    "CNPJ_DIGIT_COUNT",
    "CNPJ_WEIGHTS_1",
    "CPF_DIGIT_COUNT",
    "DV_MODULO",
    "DV_REMAINDER_THRESHOLD",
    "MAX_CPF_LENGTH",
    "MAX_HOLDOUT_TEST_SIZE",
    "MAX_PASSPORT_LENGTH",
    "MAX_UF_CODE",
    "MIN_CPF_LENGTH",
    "MIN_GEO_COORD_LENGTH",
    "MIN_INIT_SAMPLE_SIZE",
    "MIN_PASSPORT_LENGTH",
    "MIN_SPLIT_EXAMPLES",
    "MIN_TABLE_LINES",
    "MIN_TABLE_PIPE_COUNT",
    "SECONDS_PER_HOUR",
    "SECONDS_PER_MINUTE",
    "TITULO_ELEITOR_DIGIT_COUNT",
]
