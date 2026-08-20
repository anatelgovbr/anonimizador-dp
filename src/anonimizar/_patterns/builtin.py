"""Funções para registro de padrões regex embutidos.

Este módulo fornece funções standalone para adicionar padrões regex
de detecção de entidades. Cada função modifica uma lista de padrões in-place.
"""

import logging
from collections.abc import Callable

#: Regex interna com siglas de UFs brasileiras, reutilizada por padrões de RG.
BRAZIL_UF_PATTERN = r"(?:AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)"


def add_pattern_cid(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex para detecção de códigos CID.

    Adiciona múltiplos padrões para capturar códigos CID (Classificação
    Internacional de Doenças) em diferentes formatos.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [
        {"label": "CID", "pattern": {"REGEX": r"\b[A-TV-Za-tv-z]\d{2}\b"}},
        {"label": "CID", "pattern": {"REGEX": r"\b[A-TV-Za-tv-z]\d{2}\.[0-9Xx]{1,3}\b"}},
        {
            "label": "CID",
            "pattern": {"REGEX": r"(?i)\bcid[-\s]*(?:10|11)?[:\s]*[A-TV-Za-tv-z]\d{2}(?:\.[0-9Xx]{1,3})?\b"},
        },
        {
            "label": "CID",
            "pattern": {
                "REGEX": (
                    r"(?i)(?:diagnóstico|diagnostic[ao]?|código|code)"
                    r"[:\s-]*[A-TV-Za-tv-z]\d{2}(?:\.[0-9Xx]{1,3})?\b"
                )
            },
        },
        {"label": "CID", "pattern": {"REGEX": r"\b[A-TV-Za-tv-z]\d{2}(?:\.[0-9Xx]{1,3})?\b"}},
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de CID.")


def add_pattern_cpf(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex para detecção de CPF.

    Adiciona múltiplos padrões para capturar CPF em diversos formatos,
    incluindo formatações padrão e com mascaramento parcial.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [
        # CPF Separado corretamente
        {"label": "CPF", "pattern": {"REGEX": r"(?<!\d)\d{3}\.\d{3}\.\d{3}[-]\d{2}(?!\d)"}},
        {"label": "CPF", "pattern": {"REGEX": r"(?<!\d)\d{9}[-\/]\d{2}(?!\d)"}},
        {"label": "CPF", "pattern": {"REGEX": r"(?<!\d)\d{11}(?!\d)"}},
        # CPF junto com o n
        {"label": "CPF", "pattern": {"REGEX": r"\bnº:?\s*\d{9}[-\/]\d{2}(?!\d)"}},
        {"label": "CPF", "pattern": {"REGEX": r"\bnº:?\s*\d{11}(?!\d)"}},
        # CPF com asteriscos na ponta
        {"label": "CPF", "pattern": {"REGEX": r"(?<!\*)\*{3}\.\d{3}\.\d{3}[-\/]\*{2}(?!\*)"}},
        {"label": "CPF", "pattern": {"REGEX": r"(?<!\*)\*{3}\d{6}[-\/]\*{2}(?!\*)"}},
        {"label": "CPF", "pattern": {"REGEX": r"(?<!\*)\*{3}\d{6}\*{2}(?!\*)"}},
        # CPF com asteriscos no meio
        {"label": "CPF", "pattern": {"REGEX": r"(?<!\d)\d{3}\.\*{3}\.\*{3}[-\/]\d{2}(?!\d)"}},
        {"label": "CPF", "pattern": {"REGEX": r"(?<!\d)\d{3}\*{6}[-\/]\d{2}(?!\d)"}},
        {"label": "CPF", "pattern": {"REGEX": r"(?<!\d)\d{3}\*{6}\d{2}(?!\d)"}},
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de CPF.")


def add_pattern_endereco(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex de CEP classificados como ``ENDEREÇO``.

    Adiciona padrões para identificar CEPs brasileiros em diferentes formatos.
    Embora o conteúdo detectado seja um CEP, o label registrado é ``ENDEREÇO``.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [
        {"label": "ENDEREÇO", "pattern": {"REGEX": r"(?<!\d)\d{5}[-]\d{3}(?!\d)"}},
        {"label": "ENDEREÇO", "pattern": {"REGEX": r"(?<!\d)\d{2}\.\d{3}[-]\d{3}(?!\d)"}},
        {"label": "ENDEREÇO", "pattern": {"REGEX": r"(?<!\d)\d{8}(?!\d)"}},
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padroes de Endereços.")


def add_pattern_geo_coord(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex para detecção de coordenadas geográficas.

    Adiciona múltiplos padrões para capturar coordenadas em diferentes formatos.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [
        {"label": "GEO_COORD", "pattern": {"REGEX": r"(?<!\d)\d{1,3}°\d{1,2}\.?\d*'?\s*[NSEWnsew](?!\d)"}},
        {"label": "GEO_COORD", "pattern": {"REGEX": r"(?:Lat|Long)\.?\d{1,3}[NSEWnsew]\d{6}[a-z]?"}},
        {"label": "GEO_COORD", "pattern": {"REGEX": r"(?<!\d)-?\d{1,3},\d+\s+-?\d{1,3},\d+(?!\d)"}},
        {"label": "GEO_COORD", "pattern": {"REGEX": r"(?<!\d)\d{1,3}°\d{1,2}\'?\d{1,2}\.?\d*\"?[NSEWnsew](?!\d)"}},
        {
            "label": "GEO_COORD",
            "pattern": {"REGEX": r"(?<!\d)\d{1,3}°[NSEWnsew]\s+\d{1,2}\'?\s*\d{1,2}\.?\d*\"?(?!\d)"},
        },
        {
            "label": "GEO_COORD",
            "pattern": {"REGEX": r"(?i)(?:coordenada|latitude|longitude|lat|long|log)[\s:]*-?\s?\d{1,3}[°.,]\d+"},
        },
        # P7: hemisfério entre graus e minutos sem espaço, segundos com vírgula/ponto decimal
        # Ex: 21°S11'43,0'' | 48°W47'21,0'' | 20°S38'43.4"
        {
            "label": "GEO_COORD",
            "pattern": {"REGEX": r"(?<!\d)\d{1,3}[°º][NSEWnsew]\d{1,2}'?\d{1,2}[.,]\d+['\"]{0,2}(?!\d)"},
        },
        # P8: graus decimais com ponto + símbolo grau + hemisfério sufixo obrigatório
        # Ex: -10.16281°S | -48.87431°W | -10.175774°S
        {
            "label": "GEO_COORD",
            "pattern": {"REGEX": r"(?<!\d)-?\d{1,3}\.\d{3,}[°º][NSEWnsew](?!\w)"},
        },
        # P9: coordenada compacta sem prefixo — dígitos + hemisfério + dígitos
        # Ex: 02S380200, 47W561530
        {
            "label": "GEO_COORD",
            "pattern": {"REGEX": r"(?<!\d)\d{2}[NSEWnsew]\d{6}(?!\d)"},
        },
        # P10: par decimal (ponto) separado por vírgula — lat, lon na mesma entidade
        # Ex: -10.296940, -48.358915
        {
            "label": "GEO_COORD",
            "pattern": {"REGEX": r"(?<!\d)-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+(?!\d)"},
        },
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de GEO_COORD.")


def add_pattern_rg(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex para detecção de RG.

    Adiciona múltiplos padrões para capturar RG em diferentes formatos
    utilizados pelos estados brasileiros.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [
        # RG com sufixo UF por hífen (ex: 1993018 -DF, 282591227 -SP)
        {"label": "RG", "pattern": {"REGEX": rf"(?<!\d)\d{{6,9}}\s*-\s*{BRAZIL_UF_PATTERN}\b"}},
        # RG pontuado com UF no próprio span (ex: 58.896.796-3  MA)
        {
            "label": "RG",
            "pattern": {"REGEX": rf"(?<!\d)\d{{1,2}}\.\d{{3}}\.\d{{3}}-[\dXx]\s+{BRAZIL_UF_PATTERN}(?!\w)"},
        },
        {"label": "RG", "pattern": {"REGEX": r"(?<!\d)\d{1,2}\.\d{3}\.\d{3}[-]?\w{1,2}(?!\d)"}},
        {"label": "RG", "pattern": {"REGEX": r"(?<!\d)\d{1,2}\.\d{3}\.\d{3}-[\dXx](?!\d)"}},
        # Ceará com barra
        {"label": "RG", "pattern": {"REGEX": r"(?<!\d)\d{2}\.\d{3}\.\d{3}-\d\/[\dXx](?!\d)"}},
        # Minas Gerais sem DV
        {"label": "RG", "pattern": {"REGEX": r"MG-\d{2}\.\d{3}\.\d{3}(?!\d)"}},
        # AM/RR legados com 2 DV
        {"label": "RG", "pattern": {"REGEX": r"(?<!\d)\d{2}\.\d{3}\.\d{3}-[\dXx]{2}(?!\d)"}},
        # RG com prefixo "RG" explícito (B-13)
        {"label": "RG", "pattern": {"REGEX": r"(?i)\brg\s*n?[ºo°]?\s*[:\-]?\s*\d{5,10}(?!\d)"}},
        # RG com órgão emissor SSP/UF (B-04)
        {
            "label": "RG",
            "pattern": {
                "REGEX": (
                    rf"(?<!\d)\d{{1,2}}\.?\d{{3}}\.?\d{{3}}[-.]?\d?\s*[-/]?\s*"
                    rf"(?:SSP|SDS|DETRAN|IFP|PC|SESP|SEJUSP|ITEP|IGP|DGPC|SSDS?)\s*/?\s*{BRAZIL_UF_PATTERN}(?!\w)"
                )
            },
        },
        # Conselhos profissionais (CRM, CRO, CRP, CREA, OAB, etc.) como RG
        {
            "label": "RG",
            "pattern": {
                "REGEX": (
                    rf"(?i)(?:CRM|CRO|CRP|CREA|CRF|CRESS|CRECI|COREN|CRMV|OAB)"
                    rf"[-/\s]*{BRAZIL_UF_PATTERN}[-/\s]*\d{{2,10}}"
                )
            },
        },
        {
            "label": "RG",
            "pattern": {
                "REGEX": (
                    rf"(?i)\d{{2,10}}\s*[-/]\s*"
                    rf"(?:CRM|CRO|CRP|CREA|CRF|CRESS|CRECI|COREN|CRMV|OAB)"
                    rf"\s*/?\s*{BRAZIL_UF_PATTERN}"
                )
            },
        },
        {
            "label": "RG",
            "pattern": {
                "REGEX": (
                    r"(?i)(?:CRM|CRO|CRP|CREA|CRF|CRESS|CRECI|COREN|CRMV|OAB)"
                    r"[-:\s]*\d{2,10}(?!\d)"
                )
            },
        },
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de RG.")


def add_pattern_rg_estrangeiro(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex para detecção de RNE/CRNM (RG de estrangeiro).

    Todos os padrões exigem prefixo contextual (RNE, CRNM ou nome por extenso)
    para evitar falsos positivos com números soltos.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [
        # Prefixo RNE/CRNM + 6 a 9 dígitos
        {"label": "RG", "pattern": {"REGEX": r"\b(?:RNE|CRNM)\s*[:\-]?\s*\d{6,9}\b"}},
        # Prefixo RNE/CRNM + número com pontos (1a3.3.1a2)
        {"label": "RG", "pattern": {"REGEX": r"\b(?:RNE|CRNM)\s*[:\-]?\s*\d{1,3}(?:\.\d{3}){1,2}\b"}},
        # Nome completo "Registro Nacional de Estrangeiro/Migratório" + dígitos
        {
            "label": "RG",
            "pattern": {
                "REGEX": (
                    r"\b(?:Registro\s+Nacional\s+(?:de\s+Estrangeiro|Migrat[oó]rio))"
                    r"\s*(?:[:\-]|[Nn][ºo°]|[Nn]úmero)?\s*\d{6,9}\b"
                )
            },
        },
        # Nome completo + número com pontos
        {
            "label": "RG",
            "pattern": {
                "REGEX": (
                    r"\b(?:Registro\s+Nacional\s+(?:de\s+Estrangeiro|Migrat[oó]rio))"
                    r"\s*[:\-]?\s*\d{1,3}(?:\.\d{3}){1,2}\b"
                )
            },
        },
        # "Nº/Número do RNE/CRNM" + dígitos
        {"label": "RG", "pattern": {"REGEX": r"\b(?:N[ºo]|Número)\s*(?:do\s*)?(?:RNE|CRNM)\s*[:\-]?\s*\d{6,9}\b"}},
        # Tolerante a ruído (reduzido: até 3 chars entre prefixo e número)
        {"label": "RG", "pattern": {"REGEX": r"\b(?:RNE|CRNM)[^\d]{0,3}\d{6,9}\b"}},
        # Tolerante + pontos (ruído reduzido)
        {"label": "RG", "pattern": {"REGEX": r"\b(?:RNE|CRNM)[^\d]{0,3}\d{1,3}(?:\.\d{3}){1,2}\b"}},
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de RG estrangeiro (RNE/CRNM).")


def add_pattern_titulo_eleitor(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex para detecção de Título de Eleitor.

    Adiciona padrões para identificar títulos de eleitor no formato brasileiro.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [
        {
            "label": "TITULO_ELEITOR",
            "pattern": {"REGEX": r"(?i)\bt[íi]tulo(?:\s+de)?\s+eleitor\s*[:\-]?\s*\d{10,12}(?!\d)"},
        },
        {"label": "TITULO_ELEITOR", "pattern": {"REGEX": r"(?<!\d):?\d{4}\s\d{4}\s\d{4}(?!\d)"}},
        {"label": "TITULO_ELEITOR", "pattern": {"REGEX": r"(?<!\d)\d{4}\s?\d{4}\s?\d{2}[\s\-]?\d{2}(?!\d)"}},
        {"label": "TITULO_ELEITOR", "pattern": {"REGEX": r"(?<!\d):?\d{12}(?!\d)"}},
        {
            "label": "TITULO_ELEITOR",
            "pattern": {
                "REGEX": r"""(?ix)(?:t[íi]tulo(?:\s+de)?\s+eleitor\s*[:\-]?\s*|n[ºo°]\s*)?
                (
                    (?:\d{4}\s?\d{4}\s?\d{4})
                    |
                    \d{12}
                    |
                    \d{4}\s?\d{4}\s?\d{2}[\s\-]?\d{2}
                )
            """
            },
        },
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de TITULO_ELEITOR.")


def add_pattern_passaporte(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex para detecção de passaportes brasileiros.

    Adiciona padrão para identificar passaportes no formato brasileiro:
    2 letras seguidas de 5-6 dígitos.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [
        {"label": "PASSAPORTE", "pattern": {"REGEX": r"[a-zA-Z]{2}\d{5,6}\s?"}},
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de PASSAPORTE.")


def add_pattern_passaporte_est(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex para detecção de passaportes estrangeiros.

    Adiciona múltiplos padrões para identificar passaportes de diversos países.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [
        # EUA moderno: 1 letra + 8 dígitos
        {"label": "PASSAPORTE", "pattern": {"REGEX": r"(?<!\w)[A-Z]\d{8}(?!\w)"}},
        # EUA antigo / Reino Unido: 9 dígitos
        {"label": "PASSAPORTE", "pattern": {"REGEX": r"(?<!\w)\d{9}(?!\w)"}},
        # Alemanha: 1 letra específica + 8 alfanuméricos
        {"label": "PASSAPORTE", "pattern": {"REGEX": r"(?<!\w)[CFGHJK][A-Z0-9]{8}(?!\w)"}},
        # Europa geral: 1 letra + 6 dígitos (Portugal, Espanha, Itália)
        {"label": "PASSAPORTE", "pattern": {"REGEX": r"(?<!\w)[A-Z]\d{6}(?!\w)"}},
        # França: 2 dígitos + 2 letras + 5 dígitos
        {"label": "PASSAPORTE", "pattern": {"REGEX": r"(?<!\w)\d{2}[A-Z]{2}\d{5}(?!\w)"}},
        # Austrália: 1 ou 2 letras + 7 dígitos
        {"label": "PASSAPORTE", "pattern": {"REGEX": r"(?<!\w)[A-Z]{1,2}\d{7}(?!\w)"}},
        # Padrão geral ICAO: até 9 caracteres alfanuméricos
        {"label": "PASSAPORTE", "pattern": {"REGEX": r"(?<!\w)[A-Z0-9]{6,9}(?!\w)"}},
        # Com referência explícita "Passaporte" ou "Passport"
        {
            "label": "PASSAPORTE",
            "pattern": {"REGEX": r"(?i)(?:passaporte?|passport)\s*n[ºo°]?\s*[:\-]?\s*([A-Z0-9]{6,9})"},
        },
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de PASSAPORTE estrangeiro.")


def add_pattern_siape(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex para detecção de códigos SIAPE.

    Adiciona múltiplos padrões para identificar códigos SIAPE de
    servidores públicos federais.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [
        {"label": "SIAPE", "pattern": {"REGEX": r"(?<!\d)\d{7}(?!\d)"}},
        {"label": "SIAPE", "pattern": {"REGEX": r"\bSIAPE[\s-]?[nº°o]?\s*\d{7}(?!\d)"}},
        {"label": "SIAPE", "pattern": {"REGEX": r"(?<!\d)\d{6}[/-]\d{1}(?!\d)"}},
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de SIAPE.")


def add_pattern_cnh(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex para detecção de CNH.

    Adiciona padrões para identificar CNH (Carteira Nacional de Habilitação)
    em formatos de 10 ou 11 dígitos.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [
        # CNH formato completo (Registro Nacional - 11 dígitos)
        {"label": "CNH", "pattern": {"REGEX": r"(?<!\d)\d{11}(?!\d)"}},
        # CNH Espelho (10 dígitos)
        {"label": "CNH", "pattern": {"REGEX": r"(?<!\d)\d{10}(?!\d)"}},
        # CNH com formatação (espaços/hífens)
        {"label": "CNH", "pattern": {"REGEX": r"(?<!\d)\d{3}\s?\d{3}\s?\d{3}\s?[-]?\d{2}(?!\d)"}},
        # CNH com prefixos explícitos
        {"label": "CNH", "pattern": {"REGEX": r"(?i)\bcnh\b[\s:°º-]*\d{10,11}(?!\d)"}},
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de CNH.")


def add_pattern_dados_bancarios(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex para detecção de dados bancários.

    Adiciona múltiplos padrões para identificar contas, agências e
    códigos IBAN em diferentes formatos.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [
        {"label": "DADOS_BANCARIOS", "pattern": {"REGEX": r"(?<!\d)\d{4,6}[-]\d{1}(?!\d)"}},
        {"label": "DADOS_BANCARIOS", "pattern": {"REGEX": r"(?<!\d)\d{3,5}[-]\d{6,9}(?:[-]\d{1})?(?!\d)"}},
        {"label": "DADOS_BANCARIOS", "pattern": {"REGEX": r"\b[A-Za-z]{2}\d{2}(?:\s?\d{4}){4,6}\b"}},
        {"label": "DADOS_BANCARIOS", "pattern": {"REGEX": r"(?<!\d)\d{8,12}[-]\d{1}(?!\d)"}},
        {"label": "DADOS_BANCARIOS", "pattern": {"REGEX": r"\b\d{3}-\d{6}-\d\b"}},
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de DADOS_BANCARIOS.")


def add_pattern_email(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex para detecção de e-mails.

    Adiciona padrão para identificar endereços de e-mail no formato padrão.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [{"label": "EMAIL", "pattern": {"REGEX": r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"}}]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de EMAIL.")


def add_pattern_telefone(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex para detecção de telefones brasileiros.

    Adiciona múltiplos padrões para capturar telefones em formatos
    nacionais e internacionais.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [
        {"label": "TELEFONE", "pattern": {"REGEX": r"(?<!\d)\(\d{2}\)\s?\d{4,5}[-\s]?\d{4}(?!\d)"}},
        {"label": "TELEFONE", "pattern": {"REGEX": r"(?<!\d)\d{2}\s\d{4,5}[-\s]?\d{4}(?!\d)"}},
        {"label": "TELEFONE", "pattern": {"REGEX": r"\+\d{1,3}\s?\(?\d{2}\)?\s?\d{4,5}[-\s]?\d{4}(?!\d)"}},
        {"label": "TELEFONE", "pattern": {"REGEX": r"(?<!\d)\d{10,11}(?!\d)"}},
        {"label": "TELEFONE", "pattern": {"REGEX": r"(?<!\d)(?:\+?55)?\s?\(?\d{2}\)?[\s-]?\d{4,5}[\s-]?\d{4}(?!\d)"}},
        {"label": "TELEFONE", "pattern": {"REGEX": r"(?<!\d)\(\d{2,3}\)\s?\d{4,5}[\s.-]?\d{4}(?!\d)"}},
        {"label": "TELEFONE", "pattern": {"REGEX": r"(?<!\d)\(\d{2,3}\)\s?\d{1}\s?\d{4}[\s.-]?\d{4}(?!\d)"}},
        {"label": "TELEFONE", "pattern": {"REGEX": r"(?<!\d)\d{2,3}\s\d{4,5}[\s.-]?\d{4}(?!\d)"}},
        {"label": "TELEFONE", "pattern": {"REGEX": r"\+55[\s-]?\d{2,3}[\s-]?\d{4,5}[\s.-]?\d{4}(?!\d)"}},
        {"label": "TELEFONE", "pattern": {"REGEX": r"\+55\d{10,11}(?!\d)"}},
        {
            "label": "TELEFONE",
            "pattern": {"REGEX": r"(?<!\d)(?:\+?55[\s-]?)?\(?(\d{2,3})\)?[\s.-]?\d{1}?\s?\d{4}[\s.-]?\d{4}(?!\d)"},
        },
        {"label": "TELEFONE", "pattern": {"REGEX": r"(?<!\d)\d{2}\s?\d{4}\s?\d{7}(?!\d)"}},
        {
            "label": "TELEFONE",
            "pattern": {"REGEX": r"(?<!\d)(?:\+?55[\s-]?)?\(?(\d{2,3})\)?[\s.-]?\d{4,5}[\s.-]\d{4}(?!\d)"},
        },
        {"label": "TELEFONE", "pattern": {"REGEX": r"(?<!\d)\d{4}[-.\s]?\d{4}(?!\d)"}},
        {"label": "TELEFONE", "pattern": {"REGEX": r"(?<!\d)\(\d{2}\)\d{4,5}[-.\s]?\d{4}(?!\d)"}},
        {"label": "TELEFONE", "pattern": {"REGEX": r"(?<!\d)\(\d{2}\)\d{9}(?!\d)"}},
        {"label": "TELEFONE", "pattern": {"REGEX": r"(?<!\d)\(\+?55\)\s?\d{2}\s?\d{4,5}[-.\s]?\d{4}(?!\d)"}},
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de TELEFONE.")


def add_pattern_data_nascimento(patterns: list[dict], logger: logging.Logger) -> None:
    """Registra padrões regex para detecção de datas de nascimento.

    Adiciona múltiplos padrões para identificar datas em formatos
    numéricos e por extenso.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.
    """
    pattern_list = [
        {"label": "DATA_NASCIMENTO", "pattern": {"REGEX": r"\d{2}[-\/\.]\d{2}[-\/\.]\d{2}(?:\d{2})?\b"}},
        {"label": "DATA_NASCIMENTO", "pattern": {"REGEX": r"\d{4}-\d{2}-\d{2}\b"}},
        {
            "label": "DATA_NASCIMENTO",
            "pattern": {
                "REGEX": (
                    r"(?i)\b(0?[1-9]|[12][0-9]|3[01])\s+de\s+"
                    r"(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|"
                    r"setembro|outubro|novembro|dezembro)\s+de\s+(19|20)\d{2}\b"
                )
            },
        },
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de DATA_NASCIMENTO.")


def add_pattern_pis(patterns: list[dict], logger: logging.Logger) -> None:
    r"""Registra padrões regex para detecção de PIS/PASEP/NIT.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.

    Note:
        Apenas o valor numérico é capturado (sem prefixo textual como "PIS/PASEP:").
        O validador contextual verifica as palavras-chave no entorno.
    """
    pattern_list = [
        {"label": "PIS", "pattern": {"REGEX": r"(?<!\d)\d{11}(?!\d)"}},
        {"label": "PIS", "pattern": {"REGEX": r"(?<!\d)\d{10}-\d(?!\d)"}},
        {"label": "PIS", "pattern": {"REGEX": r"(?<!\d)\d{3}\.\d{5}\.\d{2}-\d\b"}},
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de PIS.")


def add_pattern_cns(patterns: list[dict], logger: logging.Logger) -> None:
    r"""Registra padrões regex para detecção de CNS (Cartão Nacional de Saúde).

    CNS sem formatação tem 15 dígitos e começa com 1, 2, 7, 8 ou 9. Também é
    aceito o formato de 16 dígitos, com hífen antes do último dígito.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.

    Note:
        Apenas o valor numérico é capturado (sem prefixo textual como "CNS:").
        O validador contextual verifica as palavras-chave no entorno.
    """
    pattern_list = [
        {"label": "CNS", "pattern": {"REGEX": r"(?<!\d)(?:[12789]\d{13}\d)(?!\d)"}},
        {"label": "CNS", "pattern": {"REGEX": r"(?<!\d)(?:[12789]\d{13}\d-\d)(?!\d)"}},
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de CNS.")


def add_pattern_reservista(patterns: list[dict], logger: logging.Logger) -> None:
    r"""Registra padrões regex para detecção de Certificado de Reservista.

    Formato: XXXXXX - XXXXXXXXXXXX ou XXXXXXXXXXXX/XXXXXX

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        logger: Logger para mensagens de debug.

    Note:
        Apenas o valor numérico é capturado (sem prefixo textual como
        "CERTIFICADO RESERVISTA:"). O validador contextual verifica
        as palavras-chave no entorno.
    """
    pattern_list = [
        {
            "label": "RESERVISTA",
            "pattern": {
                "REGEX": (
                    r"(?<!\d)(?:\d{6}\s*[-/]\s*\d{12,13}"
                    r"|\d{12,13}\s*[-/]\s*\d{6})(?!\d)"
                )
            },
        },
    ]
    patterns.extend(pattern_list)
    logger.debug("Adicionando padrões de RESERVISTA.")


#: Registro dos labels com regex built-in para suas funções de adição. As
#: funções alteram a lista de padrões recebida in-place; labels do modelo sem
#: entrada neste mapa não possuem regex built-in.
PATTERN_ADDERS: dict[str, Callable[[list[dict], logging.Logger], None]] = {
    "CPF": add_pattern_cpf,
    "RG": add_pattern_rg,
    "TITULO_ELEITOR": add_pattern_titulo_eleitor,
    "PASSAPORTE": add_pattern_passaporte,
    "SIAPE": add_pattern_siape,
    "DADOS_BANCARIOS": add_pattern_dados_bancarios,
    "EMAIL": add_pattern_email,
    "TELEFONE": add_pattern_telefone,
    "DATA_NASCIMENTO": add_pattern_data_nascimento,
    "CNH": add_pattern_cnh,
    "ENDEREÇO": add_pattern_endereco,
    "GEO_COORD": add_pattern_geo_coord,
    "CID": add_pattern_cid,
    "PIS": add_pattern_pis,
    "CNS": add_pattern_cns,
    "RESERVISTA": add_pattern_reservista,
}
