"""Constantes de formatos e colunas para avaliação e treinamento.

Este módulo contém constantes de formato de saída, nomes de colunas
obrigatórias, chaves de relatório e sufixos de merge.
"""

# =============================================================================
# EVALUATION — COLUNAS E CHAVES
# =============================================================================

#: Colunas mínimas, na ordem esperada, para DataFrames de textos de avaliação.
REQUIRED_TEXT_COLUMNS = ["id", "text"]
#: Colunas mínimas, na ordem esperada, para DataFrames de ground truth.
REQUIRED_GT_COLUMNS = ["id", "tp_entidade", "start_entidade", "end_entidade"]
#: Colunas mínimas, na ordem esperada, para DataFrames de predições.
REQUIRED_PREDICTION_COLUMNS = ["id", "tp_entidade", "start_entidade", "end_entidade"]
#: Colunas de métricas necessárias para comparar dois relatórios.
REQUIRED_REPORT_COLUMNS = ["tp_entidade", "fbeta", "precision", "recall"]

#: Sufixos de colunas homônimas ao unir ground truth e predições.
MERGE_SUFFIXES = ("_true", "_pred")
#: Sufixo de colunas homônimas ao comparar relatório atual e anterior.
COMPARISON_SUFFIXES = ("_atual", "_anterior")

#: Chave usada em relatórios para as métricas agregadas de todas as entidades.
ALL_ENTITIES_KEY = "TODAS"
#: Cabeçalho textual do relatório resumido de avaliação.
REPORT_HEADER = "=== RELATÓRIO DE AVALIAÇÃO ==="
#: Valor numérico que marca uma anotação para remoção durante o carregamento.
REMOVE_FLAG_VALUE = 1

# =============================================================================
# TRAINING — COLUNAS
# =============================================================================

#: Schema de entrada tabular com uma entidade anotada por linha.
ROW_PER_ENTITY_COLUMNS = ["texto", "start", "end", "entidade"]
#: Schema do formato de treinamento spaCy serializado em DataFrame.
STANDARD_FORMAT_COLUMNS = ["text", "entities"]
#: Schema de entidades usado pela validação cruzada.
CV_ENTITY_COLUMNS = ["id", "start", "end", "entidade", "texto"]

# =============================================================================
# ANONIMIZAÇÃO
# =============================================================================

#: Template imutável da tag que substitui uma entidade anonimizada.
ANON_TAG_FORMAT = "<|{label}|>"
#: Formato de retorno padrão das APIs de extração de entidades.
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
