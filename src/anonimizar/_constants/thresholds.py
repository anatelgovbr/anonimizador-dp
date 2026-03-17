"""Constantes de thresholds e configurações padrão numéricas.

Este módulo contém thresholds de sobreposição, parâmetros de treinamento,
configurações de validação cruzada e outros valores numéricos padrão.
"""

# =============================================================================
# DEFAULTS COMPARTILHADOS
# =============================================================================

DEFAULT_OVERLAP_THRESHOLD = 0.8
DEFAULT_BETA = 2.0
DEFAULT_N_ITER = 20
DEFAULT_DROP = 0.35
DEFAULT_BATCH_SIZE = 8
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_INITIAL_BASE_FRAC = 1.0
DEFAULT_N_JOBS = 1
DEFAULT_OUTPUT_DIR = "./trained_model"
MAX_NLP_DOCUMENT_LENGTH = 3_000_000
SPACY_LANGUAGE = "pt"

# =============================================================================
# EVALUATION
# =============================================================================

DEFAULT_OVERLAP_THRESHOLDS = [0.7, 0.8, 0.9]
DEFAULT_BETA_VALUES = [1, 2]
OVERLAP_BINS = [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
OVERLAP_BIN_LABELS = ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
PERCENT_MULTIPLIER = 100
MAX_ERROR_EXAMPLES = 10

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_BETA",
    "DEFAULT_BETA_VALUES",
    "DEFAULT_DROP",
    "DEFAULT_INITIAL_BASE_FRAC",
    "DEFAULT_N_ITER",
    "DEFAULT_N_JOBS",
    "DEFAULT_N_SPLITS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_OVERLAP_THRESHOLD",
    "DEFAULT_OVERLAP_THRESHOLDS",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VALIDATION_SPLIT",
    "MAX_ERROR_EXAMPLES",
    "MAX_NLP_DOCUMENT_LENGTH",
    "OVERLAP_BINS",
    "OVERLAP_BIN_LABELS",
    "PERCENT_MULTIPLIER",
    "SPACY_LANGUAGE",
]
