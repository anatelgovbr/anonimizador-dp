"""Constantes de formatos e colunas para avaliação e treinamento.

Este módulo contém constantes de formato de saída, nomes de colunas
obrigatórias, chaves de relatório e sufixos de merge.
"""

# =============================================================================
# EVALUATION — COLUNAS E CHAVES
# =============================================================================

REQUIRED_TEXT_COLUMNS = ["id", "text"]
REQUIRED_GT_COLUMNS = ["id", "tp_entidade", "start_entidade", "end_entidade"]
REQUIRED_PREDICTION_COLUMNS = ["id", "tp_entidade", "start_entidade", "end_entidade"]
REQUIRED_REPORT_COLUMNS = ["tp_entidade", "fbeta", "precision", "recall"]

MERGE_SUFFIXES = ("_true", "_pred")
COMPARISON_SUFFIXES = ("_atual", "_anterior")

ALL_ENTITIES_KEY = "TODAS"
REPORT_HEADER = "=== RELATÓRIO DE AVALIAÇÃO ==="
REMOVE_FLAG_VALUE = 1

# =============================================================================
# TRAINING — COLUNAS
# =============================================================================

ROW_PER_ENTITY_COLUMNS = ["texto", "start", "end", "entidade"]
STANDARD_FORMAT_COLUMNS = ["text", "entities"]
CV_ENTITY_COLUMNS = ["id", "start", "end", "entidade", "texto"]

# =============================================================================
# ANONIMIZAÇÃO
# =============================================================================

ANON_TAG_FORMAT = "<|{label}|>"
DEFAULT_RETURN_TYPE = "label_position"

__all__ = [
    "ALL_ENTITIES_KEY",
    "ANON_TAG_FORMAT",
    "COMPARISON_SUFFIXES",
    "CV_ENTITY_COLUMNS",
    "DEFAULT_RETURN_TYPE",
    "MERGE_SUFFIXES",
    "REMOVE_FLAG_VALUE",
    "REPORT_HEADER",
    "REQUIRED_GT_COLUMNS",
    "REQUIRED_PREDICTION_COLUMNS",
    "REQUIRED_REPORT_COLUMNS",
    "REQUIRED_TEXT_COLUMNS",
    "ROW_PER_ENTITY_COLUMNS",
    "STANDARD_FORMAT_COLUMNS",
]
