"""Constantes de thresholds e configurações padrão numéricas.

Este módulo contém thresholds de sobreposição, parâmetros de treinamento,
configurações de validação cruzada e outros valores numéricos padrão.
"""

# =============================================================================
# DEFAULTS COMPARTILHADOS
# =============================================================================

#: Sobreposição mínima, de 0 a 1, para considerar uma predição correspondente.
DEFAULT_OVERLAP_THRESHOLD = 0.8
#: Peso padrão de recall na métrica F-beta.
DEFAULT_BETA = 2.0
#: Número padrão de épocas de treinamento.
DEFAULT_N_ITER = 20
#: Taxa padrão de dropout do treinamento.
DEFAULT_DROP = 0.35
#: Tamanho padrão de lote de treinamento.
DEFAULT_BATCH_SIZE = 8
#: Semente padrão para operações reprodutíveis.
DEFAULT_RANDOM_STATE = 42
#: Número padrão de partições da validação cruzada.
DEFAULT_N_SPLITS = 5
#: Fração padrão reservada à validação quando o fluxo a suporta.
DEFAULT_VALIDATION_SPLIT = 0.2
#: Fração padrão destinada ao conjunto de treinamento.
DEFAULT_TRAIN_RATIO = 0.8
#: Fração inicial padrão da base de treinamento.
DEFAULT_INITIAL_BASE_FRAC = 1.0
#: Número padrão de processos paralelos.
DEFAULT_N_JOBS = 1
#: Diretório padrão para modelos treinados.
DEFAULT_OUTPUT_DIR = "./trained_model"
#: Comprimento máximo, em caracteres, aceito pelo pipeline spaCy.
MAX_NLP_DOCUMENT_LENGTH = 3_000_000
#: Código ISO 639-1 do idioma configurado para spaCy.
SPACY_LANGUAGE = "pt"

# =============================================================================
# EVALUATION
# =============================================================================

#: Thresholds padrão para avaliação em lote; não modifique a lista in place.
DEFAULT_OVERLAP_THRESHOLDS = [0.7, 0.8, 0.9]
#: Valores padrão de beta para avaliação em lote; não modifique a lista in place.
DEFAULT_BETA_VALUES = [1, 2]
#: Limites inclusivos de histogramas de sobreposição; não modifique a lista in place.
OVERLAP_BINS = [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
#: Rótulos alinhados aos intervalos consecutivos de ``OVERLAP_BINS``.
OVERLAP_BIN_LABELS = ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
#: Fator para converter proporções em percentuais nos relatórios.
PERCENT_MULTIPLIER = 100
#: Número máximo de exemplos por classe incluído na análise de erros.
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
