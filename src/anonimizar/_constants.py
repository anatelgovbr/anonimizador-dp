"""Compatibilidade legada para constantes compartilhadas.

.. deprecated:: 1.0.2
   Use :mod:`anonimizar._constants`, o pacote canônico. Este arquivo é
   mantido apenas para instalações que o referenciem diretamente e não deve
   receber novas constantes.
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
# LABELS SUPORTADOS
# =============================================================================

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
]

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
}

IGNORE_LABELS = ["LOC", "MISC", "ORG", "PER"]

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
]

# =============================================================================
# VALIDAÇÃO DE DOCUMENTOS (elimina PLR2004 - magic numbers)
# =============================================================================

CPF_DIGIT_COUNT = 11
CNPJ_DIGIT_COUNT = 14
TITULO_ELEITOR_DIGIT_COUNT = 12
CNH_DIGIT_COUNT = 11
CNH_ESPELHO_DIGIT_COUNT = 10
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
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
CNPJ_WEIGHTS_1 = [6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9]

# =============================================================================
# CONTEXT WINDOWS (back_offset, forward_offset)
# =============================================================================

CONTEXT_WINDOWS: dict[str, tuple[int, int]] = {
    "CPF": (-80, 0),
    "FISTEL": (-80, 0),
    "ENDERECO": (-20, 0),
    "ENDEREÇO": (-20, 0),
    "RG": (-100, 50),
    "CNH": (-80, 20),
    "SIAPE": (-80, 0),
    "TELEFONE": (-100, 0),
    "DATA_NASCIMENTO": (-80, 10),
    "PASSAPORTE": (-100, 0),
    "TITULO_ELEITOR": (-50, 20),
    "DADOS_BANCARIOS": (-80, 10),
    "GEO_COORD": (-100, 50),
    "CID": (-100, 50),
}

# =============================================================================
# KEYWORD LISTS
# =============================================================================

KEYWORDS_CPF = ["cpf", "fistel"]
KEYWORDS_CNH_IN_CPF = ["cnh"]
KEYWORDS_ENDERECO = ["cep", "postal", "rua", "avenida", "av."]

KEYWORDS_RG = [
    "rg",
    "identidade",
    "registro geral",
    "carteira de identidade",
    "rg número",
    "número do rg",
    "rg nº",
    "rg n",
    "documento de identidade",
    "identidade número",
    "mg-",
    "carteira de identidade nacional",
    "nova identidade",
    "rg digital",
    "rg eletrônico",
    "rg físico",
    "registro civil",
    "identificação civil",
]

KEYWORDS_CNH = ["cnh", "habilita", "carteira", "condutor"]
KEYWORDS_SIAPE = ["siape"]
KEYWORDS_SEI_EXCLUSION = ["sei"]
KEYWORDS_TELEFONE = ["telefone", "whatsapp"]
KEYWORDS_DATA_NASCIMENTO = ["nascimento", "emissor"]

KEYWORDS_PASSAPORTE = [
    "passaporte",
    "passport",
    "documento de viagem",
    "travel document",
    "número do passaporte",
    "passport number",
    "passport no",
    "passaporte n",
    "passaporte nº",
]

KEYWORDS_TITULO_ELEITOR = ["título", "eleitor", "inscrição", "eleitoral"]

KEYWORDS_DADOS_BANCARIOS = [
    "banco",
    "(sem",
    "dv)",
    "dados",
    "bancarios**",
    "conta",
    "bancários**",
    "agencia",
    "agência",
]

KEYWORDS_GEO_COORD = [
    "coordenada",
    "latitude",
    "longitude",
    "lat",
    "long",
    "posição",
    "localização",
    "gps",
    "geográfica",
    "geográfico",
    "UTM",
    "graus",
    "minutos",
    "segundos",
    "norte",
    "sul",
    "leste",
    "oeste",
    "coordenadas",
    "posicionamento",
]

KEYWORDS_CID = [
    "cid",
    "diagnóstico",
    "doença",
    "patologia",
    "enfermidade",
    "diagnóstico médico",
    "classificação",
    "código médico",
    "laudo",
    "atestado",
    "relatório médico",
    "prontuário",
    "história clínica",
    "quadro clínico",
    "condição médica",
    "morbidade",
    "nosologia",
    "diagnostic",
    "disease",
]

CID_FALSE_POSITIVES = ["A12", "B12", "C12", "D12", "E12", "F12", "G12", "H12"]

SENSITIVE_TABLE_KEYWORDS = [
    "cpf",
    "rg",
    "titulo",
    "documento",
    "passaporte",
    "cnh",
    "siape",
]

# =============================================================================
# EVALUATION
# =============================================================================

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
}

REQUIRED_TEXT_COLUMNS = ["id", "text"]
REQUIRED_GT_COLUMNS = ["id", "tp_entidade", "start_entidade", "end_entidade"]
REQUIRED_PREDICTION_COLUMNS = ["id", "tp_entidade", "start_entidade", "end_entidade"]
REQUIRED_REPORT_COLUMNS = ["tp_entidade", "fbeta", "precision", "recall"]

OVERLAP_BINS = [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
OVERLAP_BIN_LABELS = ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]

DEFAULT_OVERLAP_THRESHOLDS = [0.7, 0.8, 0.9]
DEFAULT_BETA_VALUES = [1, 2]

MERGE_SUFFIXES = ("_true", "_pred")
COMPARISON_SUFFIXES = ("_atual", "_anterior")

ALL_ENTITIES_KEY = "TODAS"
REPORT_HEADER = "=== RELATÓRIO DE AVALIAÇÃO ==="
REMOVE_FLAG_VALUE = 1
MAX_ERROR_EXAMPLES = 10
PERCENT_MULTIPLIER = 100

# =============================================================================
# TRAINING
# =============================================================================

ROW_PER_ENTITY_COLUMNS = ["texto", "start", "end", "entidade"]
STANDARD_FORMAT_COLUMNS = ["text", "entities"]
CV_ENTITY_COLUMNS = ["id", "start", "end", "entidade", "texto"]

# =============================================================================
# ANONIMIZAÇÃO
# =============================================================================

ANON_TAG_FORMAT = "<|{label}|>"
DEFAULT_RETURN_TYPE = "label_position"
