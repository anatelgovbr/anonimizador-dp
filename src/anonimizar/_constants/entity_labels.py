"""Constantes de labels de entidades para NER.

Este módulo contém todas as definições de labels de entidades suportadas,
prioridades de processamento e mapeamentos de avaliação.
"""

# =============================================================================
# LABELS SUPORTADOS
# =============================================================================

#: Labels aceitos pelas APIs de extração e treinamento; trate a lista como imutável.
DEFAULT_SUPPORTED_LABELS = [
    "CPF",
    "RG",
    "SIAPE",
    "ENDEREÇO",
    "TELEFONE",
    "EMAIL",
    "DADOS_BANCARIOS",
    "CNH",
    "GEO_COORD",
    "CID",
    "PASSAPORTE",
    "TITULO_ELEITOR",
    "DATA_NASCIMENTO",
    "PIS",
    "CNS",
    "RESERVISTA",
]

#: Conjunto de labels que um modelo spaCy treinado pelo pacote pode emitir.
ALL_MODEL_LABELS = {
    "CPF",
    "RG",
    "TITULO_ELEITOR",
    "PASSAPORTE",
    "SIAPE",
    "DADOS_BANCARIOS",
    "EMAIL",
    "TELEFONE",
    "DATA_NASCIMENTO",
    "CNH",
    "ENDEREÇO",
    "GEO_COORD",
    "CID",
    "PIS",
    "CNS",
    "RESERVISTA",
}

# Labels do modelo spaCy pré-treinado que devem ser ignorados
#: Labels genéricos de modelos spaCy que são descartados na extração.
IGNORE_LABELS = ["LOC", "MISC", "ORG", "PER"]

# =============================================================================
# PRIORIDADE DE PROCESSAMENTO
# =============================================================================

#: Ordem de precedência para resolver spans de entidades sobrepostos.
ENTITY_PRIORITY_LIST = [
    "PASSAPORTE",
    "CPF",
    "ENDEREÇO",
    "CNH",
    "SIAPE",
    "RG",
    "TELEFONE",
    "DADOS_BANCARIOS",
    "EMAIL",
    "DATA_NASCIMENTO",
    "TITULO_ELEITOR",
    "GEO_COORD",
    "CID",
    "PIS",
    "CNS",
    "RESERVISTA",
]

# =============================================================================
# MAPEAMENTO DE AVALIAÇÃO
# =============================================================================

#: Normalização de labels externos para os labels canônicos de avaliação.
DEFAULT_ENTITY_MAPPING = {
    "CPF/FISTEL": "CPF",
    "FISTEL": "CPF",
    "RG": "RG",
    "CNH": "CNH",
    "Título de eleitor": "TITULO_ELEITOR",
    "Passaporte": "PASSAPORTE",
    "PASSAPORTE": "PASSAPORTE",
    "SIAPE": "SIAPE",
    "Matrícula SIAPE": "SIAPE",
    "Data de Nascimento": "DATA_NASCIMENTO",
    "Dados Bancários": "DADOS_BANCARIOS",
    "E-mail": "EMAIL",
    "Telefone": "TELEFONE",
    "Endereço": "ENDEREÇO",
    "TITULO_ELEITOR": "TITULO_ELEITOR",
    "DATA_NASCIMENTO": "DATA_NASCIMENTO",
    "DADOS_BANCARIOS": "DADOS_BANCARIOS",
    "EMAIL": "EMAIL",
    "TELEFONE": "TELEFONE",
    "ENDERECO": "ENDEREÇO",
    "ENDEREÇO": "ENDEREÇO",
    "PIS": "PIS",
    "PASEP": "PIS",
    "NIT": "PIS",
    "CNS": "CNS",
    "RESERVISTA": "RESERVISTA",
}

__all__ = [
    "ALL_MODEL_LABELS",
    "DEFAULT_ENTITY_MAPPING",
    "DEFAULT_SUPPORTED_LABELS",
    "ENTITY_PRIORITY_LIST",
    "IGNORE_LABELS",
]
