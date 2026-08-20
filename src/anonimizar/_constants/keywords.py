"""Constantes de keywords para validação contextual de entidades.

Este módulo contém as listas de keywords utilizadas pelos validadores
de contexto para confirmar ou rejeitar entidades detectadas.
"""

# =============================================================================
# KEYWORDS POR TIPO DE ENTIDADE
# =============================================================================

#: Keywords de contexto para CPF/FISTEL; listas de keywords não devem ser alteradas in place.
KEYWORDS_CPF = ["cpf", "fistel"]
KEYWORDS_CNH_IN_CPF = ["cnh"]
KEYWORDS_ENDERECO = [
    "cep",
    "postal",
    "rua",
    "avenida",
    "av.",
    "endereço",
    "endereco",
    "logradouro",
    "bairro",
    "residência",
    "residencia",
    "domicílio",
    "domicilio",
    "alameda",
    "travessa",
    "rodovia",
    "estrada",
    "praça",
    "quadra",
    "complemento",
    "cidade",
]

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

# Padrão regex para órgãos emissores de RG (SSP/UF, DETRAN/UF, etc.)
# Formato: SIGLA/UF ou SIGLA UF
RG_ORGAO_EMISSOR_PATTERN = (
    r"(?:SSP|SDS|DETRAN|IFP|PC|SESP|SEJUSP|ITEP|IGP|DGPC|SSDS?"
    r"|CRM|CRO|CRP|CREA|CRF|CRESS|CRECI|COREN|CRMV|OAB)"
    r"[-\s/]*[A-Z]{2}"
)

KEYWORDS_CNH = [
    "cnh",
    "habilitação",
    "habilitacao",
    "habilita",
    "condutor",
    "condutora",
    "carteira de motorista",
    "carteira nacional",
    "registro de habilitação",
    "registro de habilitacao",
]
KEYWORDS_SIAPE = ["siape", "matrícula", "matricula", "mat.", "mat:"]
KEYWORDS_SEI_EXCLUSION = ["sei"]
KEYWORDS_TELEFONE = [
    "telefone",
    "whatsapp",
    "fone",
    "ramal",
    "celular",
    "fax",
    "tel.",
    "tel:",
    "tel ",
    "contato",
    "fixo",
    "móvel",
    "movel",
    "wpp",
    "ddd",
]
KEYWORDS_DATA_NASCIMENTO = [
    "nascimento",
    "nascido",
    "nascida",
    "nasc.",
    "nasc:",
    "dt nasc",
    "dt. nasc",
    "d.n.",
    "dn:",
    "data de nascimento",
]
KEYWORDS_DATA_NASCIMENTO_EXCLUDE = [
    "início",
    "termino",
    "vigência",
    "validade",
    "contrato",
    "término",
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
    "reunião",
    "reuniao",
    "pagamento",
    "vencimento",
    "audiência",
    "audiencia",
    "prazo",
    "assinatura",
    "despacho",
    "emissão",
    "emissao",
    "expedição",
    "expedicao",
    "admissão",
    "admissao",
    "aposentadoria",
    "nomeação",
    "nomeacao",
    "posse",
    "exercício",
    "exercicio",
    "contribuição",
    "contribuicao",
    "ingresso",
    "vacância",
    "vacancia",
    "férias",
    "ferias",
    "licença",
    "licenca",
    "frequência",
    "frequencia",
    "período",
    "periodo",
    "gravação",
    "gravacao",
    "programação",
    "programacao",
    "centenário",
    "centenario",
]

KEYWORDS_RNE = [
    "rne",
    "crnm",
    "registro nacional de estrangeiro",
    "registro nacional de migratório",
    "registro nacional migratório",
    "estrangeiro",
    "migratório",
    "migrante",
    "imigrante",
    "documento migratório",
    "carteira de registro nacional migratório",
    "número do rne",
    "número do crnm",
    "nº do rne",
    "nº do crnm",
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
    "conta",
    "agencia",
    "agência",
    "bancário",
    "bancario",
    "bancários",
    "bancarios",
    "pix",
    "poupança",
    "poupanca",
    "corrente",
    "c/c",
    "operação",
    "operacao",
    "código do banco",
    "codigo do banco",
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
    "utm",
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

KEYWORDS_PIS = [
    "pis",
    "pasep",
    "nit",
    "inscrição do trabalhador",
    "inscricao do trabalhador",
    "programa de integração social",
    "programa de integracao social",
    "número do pis",
    "numero do pis",
    "pis/pasep",
]

KEYWORDS_CNS = [
    "cns",
    "cartão nacional de saúde",
    "cartao nacional de saude",
    "número do cns",
    "numero do cns",
    "cns:",
    "sus",
    "sistema único de saúde",
    "sistema unico de saude",
]

KEYWORDS_RESERVISTA = [
    "reservista",
    "certificado de reservista",
    "certificado de alistamento militar",
    "certificado de dispensa de incorporação",
    "cam",
    "cda",
    "cdi",
    "alistamento militar",
    "serviço militar",
    "servico militar",
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

PERSONAL_DATA_TABLE_KEYWORDS = [
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

#: Janela ``(início, fim)`` em caracteres, relativa à entidade, para cada validador contextual.
#: O mapeamento é configuração compartilhada e deve ser tratado como imutável.
CONTEXT_WINDOWS: dict[str, tuple[int, int]] = {
    "CPF": (-80, 20),
    "FISTEL": (-80, 20),
    "ENDEREÇO": (-80, 20),
    # reduzir janela de contexto para RG para diminuir falsos positivos por termos distantes
    "RG": (-100, 50),
    "RNE": (-100, 50),
    "CNH": (-80, 20),
    "SIAPE": (-80, 20),
    "TELEFONE": (-100, 20),
    "DATA_NASCIMENTO": (-80, 10),
    "PASSAPORTE": (-100, 20),
    "TITULO_ELEITOR": (-50, 20),
    "DADOS_BANCARIOS": (-80, 10),
    "GEO_COORD": (-100, 50),
    "CID": (-100, 50),
    "PIS": (-80, 20),
    "CNS": (-80, 20),
    "RESERVISTA": (-80, 50),
}

__all__ = [
    "CID_FALSE_POSITIVES",
    "CONTEXT_WINDOWS",
    "KEYWORDS_CID",
    "KEYWORDS_CNH",
    "KEYWORDS_CNH_IN_CPF",
    "KEYWORDS_CNS",
    "KEYWORDS_CPF",
    "KEYWORDS_DADOS_BANCARIOS",
    "KEYWORDS_DATA_NASCIMENTO",
    "KEYWORDS_DATA_NASCIMENTO_EXCLUDE",
    "KEYWORDS_ENDERECO",
    "KEYWORDS_GEO_COORD",
    "KEYWORDS_PASSAPORTE",
    "KEYWORDS_PIS",
    "KEYWORDS_RESERVISTA",
    "KEYWORDS_RG",
    "KEYWORDS_RNE",
    "KEYWORDS_SEI_EXCLUSION",
    "KEYWORDS_SIAPE",
    "KEYWORDS_TELEFONE",
    "KEYWORDS_TITULO_ELEITOR",
    "PERSONAL_DATA_TABLE_KEYWORDS",
    "RG_ORGAO_EMISSOR_PATTERN",
]
