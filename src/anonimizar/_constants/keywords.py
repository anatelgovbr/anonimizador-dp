"""Constantes de keywords para validação contextual de entidades.

Este módulo contém as listas de keywords utilizadas pelos validadores
de contexto para confirmar ou rejeitar entidades detectadas.
"""

# =============================================================================
# KEYWORDS POR TIPO DE ENTIDADE
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
    "documento",
    "número",
]

KEYWORDS_CNH = ["cnh", "habilita", "carteira", "condutor"]
KEYWORDS_SIAPE = ["siape"]
KEYWORDS_SEI_EXCLUSION = ["sei"]
KEYWORDS_TELEFONE = ["telefone", "whatsapp", "fone", "ramal", "celular", "fax"]
KEYWORDS_DATA_NASCIMENTO = ["nascimento", "emissor"]
KEYWORDS_DATA_NASCIMENTO_EXCLUDE = [
    "início",
    "termino",
    "vigência",
    "validade",
    "contrato",
    "térrino",
    "processo",
    "protocolo",
    "publicação",
    "publicado",
    "publicada",
    "diário",
    "ofício",
    "certidão",
    "certame",
    "edital",
    "inicio",  # Sem acento
    "fim",
]

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
    "cid-10",
    "cid10",
    "diagnóstico",
    "doença",
    "patologia",
    "enfermidade",
    "diagnóstico médico",
    "classificação",
    "código médico",
    "código cid",
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
    "identificacao",
    "email",
    "fistel",
    "telefone",
]

# =============================================================================
# JANELAS DE CONTEXTO (back_offset, forward_offset)
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

__all__ = [
    "CID_FALSE_POSITIVES",
    "CONTEXT_WINDOWS",
    "KEYWORDS_CID",
    "KEYWORDS_CNH",
    "KEYWORDS_CNH_IN_CPF",
    "KEYWORDS_CPF",
    "KEYWORDS_DADOS_BANCARIOS",
    "KEYWORDS_DATA_NASCIMENTO",
    "KEYWORDS_ENDERECO",
    "KEYWORDS_GEO_COORD",
    "KEYWORDS_PASSAPORTE",
    "KEYWORDS_RG",
    "KEYWORDS_SEI_EXCLUSION",
    "KEYWORDS_SIAPE",
    "KEYWORDS_TELEFONE",
    "KEYWORDS_TITULO_ELEITOR",
    "SENSITIVE_TABLE_KEYWORDS",
]
