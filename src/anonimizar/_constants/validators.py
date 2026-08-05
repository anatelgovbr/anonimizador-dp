"""Constantes de validação de documentos brasileiros.

Este módulo contém constantes numéricas usadas na validação de CPF, CNPJ,
CNH, Título de Eleitor, passaporte e outros documentos, eliminando magic numbers.
"""

# =============================================================================
# CONTAGEM DE DÍGITOS
# =============================================================================

#: Quantidade de dígitos de um CPF sem pontuação.
CPF_DIGIT_COUNT = 11
#: Quantidade de dígitos de um CNPJ sem pontuação.
CNPJ_DIGIT_COUNT = 14
#: Quantidade de dígitos de um Título de Eleitor sem pontuação.
TITULO_ELEITOR_DIGIT_COUNT = 12
#: Quantidade de dígitos de uma CNH padrão.
CNH_DIGIT_COUNT = 11
#: Quantidade de dígitos da CNH no formato espelho.
CNH_ESPELHO_DIGIT_COUNT = 10

# =============================================================================
# LIMITES DE VALIDAÇÃO
# =============================================================================

#: Limite superior exclusivo dos códigos numéricos de UF aceitos.
MAX_UF_CODE = 28
#: Resto mínimo que altera a regra de cálculo de dígito verificador.
DV_REMAINDER_THRESHOLD = 2
#: Valor de módulo usado nas regras de dígito verificador aplicáveis.
DV_MODULO = 10

# Aliases semânticos para validação de Título de Eleitor (usa módulo 11, mesmo que CPF)
TITULO_MODULO = CPF_DIGIT_COUNT  # Módulo 11 conforme TSE
TITULO_DV_THRESHOLD = DV_MODULO  # Resto >= 10 → dígito = 0

#: Quantidade de dígitos de PIS/PASEP/NIT sem pontuação.
PIS_DIGIT_COUNT = 11
#: Pesos, da esquerda para a direita, do dígito verificador de PIS.
PIS_WEIGHTS = [3, 2, 9, 8, 7, 6, 5, 4, 3, 2]

#: Quantidade de dígitos de um Cartão Nacional de Saúde.
CNS_DIGIT_COUNT = 15
#: Prefixos iniciais aceitos para números de CNS.
CNS_VALID_PREFIXES = (1, 2, 7, 8, 9)

#: Comprimento mínimo aceito por padrões candidatos a CPF.
MIN_CPF_LENGTH = 9
#: Comprimento máximo aceito por padrões candidatos a CPF.
MAX_CPF_LENGTH = 16
#: Comprimento mínimo aceito por padrões candidatos a passaporte.
MIN_PASSPORT_LENGTH = 6
#: Comprimento máximo aceito por padrões candidatos a passaporte.
MAX_PASSPORT_LENGTH = 9
#: Comprimento mínimo de uma coordenada geográfica candidata.
MIN_GEO_COORD_LENGTH = 3
#: Número mínimo de caracteres ``|`` para reconhecer uma tabela Markdown.
MIN_TABLE_PIPE_COUNT = 3
#: Número mínimo de linhas para reconhecer uma tabela Markdown.
MIN_TABLE_LINES = 3
#: Tamanho mínimo da amostra inicial em rotinas de treinamento.
MIN_INIT_SAMPLE_SIZE = 50
#: Número mínimo de exemplos para dividir um conjunto de dados.
MIN_SPLIT_EXAMPLES = 2
#: Fração máxima exclusiva permitida para o conjunto holdout.
MAX_HOLDOUT_TEST_SIZE = 0.5

# =============================================================================
# CONSTANTES TEMPORAIS
# =============================================================================

#: Quantidade de segundos em um minuto.
SECONDS_PER_MINUTE = 60
#: Quantidade de segundos em uma hora.
SECONDS_PER_HOUR = 3600

# =============================================================================
# PESOS DE VALIDAÇÃO
# =============================================================================

#: Pesos, da esquerda para a direita, do primeiro dígito verificador de CNPJ.
#: A lista é compartilhada e não deve ser alterada in place.
CNPJ_WEIGHTS_1 = [6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9]

__all__ = [
    "CNH_DIGIT_COUNT",
    "CNH_ESPELHO_DIGIT_COUNT",
    "CNPJ_DIGIT_COUNT",
    "CNPJ_WEIGHTS_1",
    "CNS_DIGIT_COUNT",
    "CNS_VALID_PREFIXES",
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
    "PIS_DIGIT_COUNT",
    "PIS_WEIGHTS",
    "SECONDS_PER_HOUR",
    "SECONDS_PER_MINUTE",
    "TITULO_DV_THRESHOLD",
    "TITULO_ELEITOR_DIGIT_COUNT",
    "TITULO_MODULO",
]
