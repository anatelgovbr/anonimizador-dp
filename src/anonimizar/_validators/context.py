"""Validadores contextuais para dados pessoais.

Este módulo fornece funções de validação que verificam o contexto textual
ao redor de entidades detectadas para determinar se são válidas.
"""

import re
from logging import Logger

from anonimizar._constants import (
    CID_FALSE_POSITIVES,
    CONTEXT_WINDOWS,
    KEYWORDS_CID,
    KEYWORDS_CNH,
    KEYWORDS_CNH_IN_CPF,
    KEYWORDS_CNS,
    KEYWORDS_CPF,
    KEYWORDS_DADOS_BANCARIOS,
    KEYWORDS_DATA_NASCIMENTO,
    KEYWORDS_DATA_NASCIMENTO_EXCLUDE,
    KEYWORDS_ENDERECO,
    KEYWORDS_GEO_COORD,
    KEYWORDS_PASSAPORTE,
    KEYWORDS_PIS,
    KEYWORDS_RESERVISTA,
    KEYWORDS_RG,
    KEYWORDS_RNE,
    KEYWORDS_SIAPE,
    KEYWORDS_TELEFONE,
    KEYWORDS_TITULO_ELEITOR,
    MAX_CPF_LENGTH,
    MAX_PASSPORT_LENGTH,
    MIN_CPF_LENGTH,
    MIN_GEO_COORD_LENGTH,
    MIN_PASSPORT_LENGTH,
    RG_ORGAO_EMISSOR_PATTERN,
)
from anonimizar._validators.documents import (
    valida_cnh,
    valida_cns,
    valida_cpf,
    valida_pis,
    valida_titulo_eleitor,
)
from anonimizar._validators.registry import register_validator

BRAZIL_UF_PATTERN = r"(?:AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)"
MIN_RG_DIGITS_WITH_ISSUER = 6
MIN_CONSELHO_DIGITS = 2
RG_CONTEXT_HINT_TERMS = (
    "rg",
    "identidade",
    "documento",
    "identificacao",
    "identificação",
    "orgao",
    "órgão",
    "emissor",
)
RG_DIGIT_CONTEXT_TERMS = (
    "rg",
    "identidade",
    "nº",
    "n°",
    "numero",
    "n.",
    "nº do",
    "número",
    "rg:",
    "rg n",
    "rg nº",
    "ssp",
    "detran",
)
RG_AUX_CONTEXT_TERMS = ("documento", "identificacao", "orgao", "emissor")
RG_SIX_DIGIT_CONTEXT_TERMS = ("rg", "identidade", "registro geral", "documento de identidade")


def _get_context_window(text: str, start: int, end: int, label: str) -> str:
    """Extrai a janela de contexto para um label.

    Args:
        text (str): Texto completo
        start (int): Posição inicial da entidade
        end (int): Posição final da entidade
        label (str): Label da entidade

    Returns:
        str: Texto da janela de contexto em lowercase
    """
    back, forward = CONTEXT_WINDOWS.get(label, (-100, 0))
    ctx_start = max(0, start + back)
    ctx_end = min(len(text), end + forward) if forward else end
    return text[ctx_start:ctx_end].lower()


def _has_rg_context_term(text: str, start: int, end: int, terms: tuple[str, ...]) -> bool:
    contexto_rg = _get_context_window(text, start, end, "RG")
    short_before = text[max(0, start - 60) : end].lower()
    return any(term in contexto_rg or term in short_before for term in terms)


CONSELHO_PROFISSIONAL_PATTERN = r"(?:CRM|CRO|CRP|CREA|CRF|CRESS|CRECI|COREN|CRMV)"


def _validate_rg_issuer(rg_text: str, logger: Logger) -> bool | None:
    is_conselho = re.search(CONSELHO_PROFISSIONAL_PATTERN, rg_text) is not None
    if is_conselho:
        digits_only = re.sub(r"\D", "", rg_text)
        if len(digits_only) >= MIN_CONSELHO_DIGITS:
            logger.debug("RG aceito por conselho profissional: %s", rg_text)
            return True

    if not re.search(rf"\b(?:{RG_ORGAO_EMISSOR_PATTERN})\b", rg_text):
        return None

    uf_at_end = re.search(r"[A-Z]{2}$", rg_text) is not None
    uf_after_slash = re.search(r"/[A-Z]{2}", rg_text) is not None
    uf_before_digits = re.search(r"[A-Z]{2}\s+\d", rg_text) is not None

    if not (uf_at_end or uf_after_slash or uf_before_digits):
        logger.debug("Órgão emissor detectado mas sem UF explícito, não aceitar automaticamente: %s", rg_text)
        return None

    digits_only = re.sub(r"\D", "", rg_text)
    min_digits = MIN_RG_DIGITS_WITH_ISSUER
    if len(digits_only) >= min_digits:
        logger.debug("RG aceito por órgão emissor estrito no texto (com número): %s", rg_text)
        return True

    logger.debug("Órgão emissor com UF encontrado mas sem número plausível: %s", rg_text)
    return None


def _validate_rg_punctuated(
    original_span: str, text: str, start: int, end: int, rg_text: str, logger: Logger
) -> bool | None:
    if re.match(rf"^\d{{1,2}}\.\d{{3}}\.\d{{3}}-[\dXx]\s+{BRAZIL_UF_PATTERN}$", original_span.strip()):
        logger.debug("RG aceito por formato pontuado com UF: %s", rg_text)
        return True

    if not re.search(r"\d{1,3}(?:\.\d{3}){1,2}[-/]?\w?$", original_span.replace(" ", "")):
        return None

    if not _has_rg_context_term(text, start, end, RG_CONTEXT_HINT_TERMS):
        logger.debug("RG rejeitado: formato pontuado sem contexto explícito: %s", rg_text)
        return False

    logger.debug("RG aceito por formato com pontuação: %s", rg_text)
    return True


def _validate_rg_uf_suffix(
    original_span: str, text: str, start: int, end: int, rg_text: str, logger: Logger
) -> bool | None:
    if not re.match(rf"^\d{{6,9}}\s*[-/]\s*{BRAZIL_UF_PATTERN}$", original_span):
        return None

    if _has_rg_context_term(text, start, end, RG_CONTEXT_HINT_TERMS):
        logger.debug("RG aceito por sufixo UF com contexto: %s", rg_text)
        return True

    logger.debug("RG rejeitado: sufixo UF sem contexto explícito: %s", rg_text)
    return False


def _validate_rg_digits(rg_clean: str, text: str, start: int, end: int, rg_text: str, logger: Logger) -> bool | None:
    if re.match(r"^\d{7,9}$", rg_clean):
        if _has_rg_context_term(text, start, end, RG_DIGIT_CONTEXT_TERMS):
            logger.debug("RG aceito por dígitos plausíveis (7-9) com contexto explícito: %s", rg_text)
            return True

        larger_ctx = _get_context_window(text, start, end, "RG").lower()
        if any(term in larger_ctx for term in RG_AUX_CONTEXT_TERMS):
            logger.debug(
                "RG aceito por dígitos (7-9) por presença de termos auxiliares na janela maior: %s | ctx=%s",
                rg_text,
                larger_ctx,
            )
            return True

        logger.debug("RG rejeitado: 7-9 dígitos sem contexto explícito: %s", rg_text)
        return False

    if re.match(r"^\d{6}$", rg_clean):
        contexto_rg = _get_context_window(text, start, end, "RG")
        if any(term in contexto_rg for term in RG_SIX_DIGIT_CONTEXT_TERMS):
            logger.debug("RG aceito: 6 dígitos com contexto explícito: %s", rg_text)
            return True

        logger.debug("RG rejeitado: 6 dígitos sem contexto explícito: %s", rg_text)
        return False

    return None


@register_validator("CPF")
@register_validator("FISTEL")
def validate_cpf_context(
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,
    logger: Logger,
    *,
    use_cpf_validator: bool = True,
) -> bool | str:
    """Valida entidade CPF/FISTEL por contexto e algoritmo.

    Args:
        text (str): Texto completo
        start_char (int): Posição inicial
        end_char (int): Posição final
        text_entity (str): Texto da entidade
        logger (Logger): Logger
        use_cpf_validator (bool): Se deve usar validador algorítmico

    Returns:
        bool | str: True (válido), False (inválido), ou "CNH" (reclassificação)
    """
    contexto = _get_context_window(text, start_char, end_char, "CPF")

    has_cpf_fistel = any(p in contexto for p in KEYWORDS_CPF)
    has_cnh = any(p in contexto for p in KEYWORDS_CNH_IN_CPF)

    if has_cpf_fistel:
        if (end_char - start_char > MAX_CPF_LENGTH) or (end_char - start_char < MIN_CPF_LENGTH):
            logger.debug("CPF rejeitado por tamanho inválido no span (%d chars)", end_char - start_char)
            return False
        if use_cpf_validator and not valida_cpf(text_entity):
            logger.debug("CPF rejeitado por validação algorítmica: %s", text_entity)
            return False
        logger.debug("CPF aceito (contexto e validação ok)")
        return True

    if has_cnh:
        logger.debug("Reclassificando de CPF para CNH por contexto")
        return "CNH"

    if use_cpf_validator and not valida_cpf(text_entity):
        logger.debug("CPF rejeitado por validação algorítmica (sem contexto explícito): %s", text_entity)
        return False

    logger.debug("CPF aceito (sem contexto explícito, mas válido)")
    return True


@register_validator("ENDEREÇO")
def validate_endereco_context(
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,  # noqa: ARG001
    logger: Logger,
) -> bool:
    """Valida entidade ENDEREÇO por contexto.

    Args:
        text (str): Texto completo
        start_char (int): Posição inicial
        end_char (int): Posição final
        text_entity (str): Texto da entidade (não usado)
        logger (Logger): Logger

    Returns:
        bool: True se válido, False caso contrário
    """
    contexto = _get_context_window(text, start_char, end_char, "ENDEREÇO")
    ok = any(p in contexto for p in KEYWORDS_ENDERECO)
    logger.debug("ENDEREÇO contexto_ok=%s", ok)
    return ok


@register_validator("RG")
def validate_rg_context(
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,
    logger: Logger,
) -> bool:
    """Valida entidade RG por contexto.

    Args:
        text (str): Texto completo
        start_char (int): Posição inicial
        end_char (int): Posição final
        text_entity (str): Texto da entidade
        logger (Logger): Logger

    Returns:
        bool: True se válido, False caso contrário
    """
    contexto = _get_context_window(text, start_char, end_char, "RG")
    ok = any(p in contexto for p in KEYWORDS_RG)

    if not ok:
        rne_contexto = _get_context_window(text, start_char, end_char, "RNE")
        ok = any(p in rne_contexto for p in KEYWORDS_RNE)

    if ok:
        logger.debug("RG aceito por contexto: %s", ok)
        return True

    original_span = text[start_char:end_char] if text is not None else ""
    rg_text = text_entity.strip().upper() if text_entity else original_span.strip().upper()
    rg_clean = re.sub(r"[\s.\-/]", "", original_span)

    for validator_result in (
        _validate_rg_issuer(rg_text, logger),
        _validate_rg_punctuated(original_span, text, start_char, end_char, rg_text, logger),
        _validate_rg_uf_suffix(original_span, text, start_char, end_char, rg_text, logger),
        _validate_rg_digits(rg_clean, text, start_char, end_char, rg_text, logger),
    ):
        if validator_result is not None:
            return validator_result

    logger.debug("RG contexto_ok=%s", ok)
    return ok


@register_validator("CNH")
def validate_cnh_context(
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,
    logger: Logger,
    *,
    use_cnh_validator: bool = True,
) -> bool:
    """Valida entidade CNH por contexto e algoritmo.

    Args:
        text (str): Texto completo
        start_char (int): Posição inicial
        end_char (int): Posição final
        text_entity (str): Texto da entidade
        logger (Logger): Logger
        use_cnh_validator (bool): Se deve usar validador algorítmico

    Returns:
        bool: True se válido, False caso contrário
    """
    contexto = _get_context_window(text, start_char, end_char, "CNH")
    has_context = any(p in contexto for p in KEYWORDS_CNH)

    if has_context:
        logger.debug("CNH aceita por contexto explícito")
        return True

    if use_cnh_validator:
        cnh_num = re.sub(r"\D", "", text_entity)
        if len(cnh_num) == 11 and not valida_cnh(cnh_num):  # noqa: PLR2004
            logger.debug("CNH rejeitada por validação algorítmica (sem contexto): %s", cnh_num)
            return False

    logger.debug("CNH contexto_ok=%s", has_context)
    return has_context


@register_validator("SIAPE")
def validate_siape_context(
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,  # noqa: ARG001
    logger: Logger,
) -> bool:
    """Valida entidade SIAPE por contexto (verifica presença de SIAPE vs SEI).

    Args:
        text (str): Texto completo
        start_char (int): Posição inicial
        end_char (int): Posição final
        text_entity (str): Texto da entidade (não usado)
        logger (Logger): Logger

    Returns:
        bool: True se válido, False caso contrário
    """
    contexto = _get_context_window(text, start_char, end_char, "SIAPE")

    has_siape = any(p in contexto for p in KEYWORDS_SIAPE)
    ok = has_siape

    logger.debug("SIAPE contexto_ok=%s (has_siape=%s)", ok, has_siape)
    return ok


@register_validator("TELEFONE")
def validate_telefone_context(
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,  # noqa: ARG001
    logger: Logger,
) -> bool:
    """Valida entidade TELEFONE por contexto.

    Args:
        text (str): Texto completo
        start_char (int): Posição inicial
        end_char (int): Posição final
        text_entity (str): Texto da entidade (não usado)
        logger (Logger): Logger

    Returns:
        bool: True se válido, False caso contrário
    """
    contexto = _get_context_window(text, start_char, end_char, "TELEFONE")
    ok = any(p in contexto for p in KEYWORDS_TELEFONE)
    logger.debug("TELEFONE contexto_ok=%s", ok)
    return ok


@register_validator("DATA_NASCIMENTO")
def validate_data_nascimento_context(
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,  # noqa: ARG001
    logger: Logger,
) -> bool:
    """Valida entidade DATA_NASCIMENTO por contexto.

    Args:
        text (str): Texto completo
        start_char (int): Posição inicial
        end_char (int): Posição final
        text_entity (str): Texto da entidade (não usado)
        logger (Logger): Logger

    Returns:
        bool: True se válido, False caso contrário
    """
    contexto = _get_context_window(text, start_char, end_char, "DATA_NASCIMENTO")
    has_exclude = any(p in contexto for p in KEYWORDS_DATA_NASCIMENTO_EXCLUDE)
    if has_exclude:
        logger.debug("DATA_NASCIMENTO contexto_ok=False (palavra de exclusão)")
        return False
    ok = any(p in contexto for p in KEYWORDS_DATA_NASCIMENTO)
    logger.debug("DATA_NASCIMENTO contexto_ok=%s", ok)
    return ok


@register_validator("PASSAPORTE")
def validate_passaporte_context(
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,
    logger: Logger,
) -> bool:
    """Valida entidade PASSAPORTE por comprimento e contexto.

    Args:
        text (str): Texto completo
        start_char (int): Posição inicial
        end_char (int): Posição final
        text_entity (str): Texto da entidade
        logger (Logger): Logger

    Returns:
        bool: True se válido, False caso contrário
    """
    passaporte_text = text_entity.strip()
    if len(passaporte_text) < MIN_PASSPORT_LENGTH or len(passaporte_text) > MAX_PASSPORT_LENGTH:
        logger.debug("PASSAPORTE rejeitado por comprimento: %d", len(passaporte_text))
        return False

    contexto = _get_context_window(text, start_char, end_char, "PASSAPORTE")
    ok = any(p in contexto for p in KEYWORDS_PASSAPORTE)
    logger.debug("PASSAPORTE contexto_ok=%s", ok)
    return ok


@register_validator("TITULO_ELEITOR")
def validate_titulo_eleitor_context(
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,
    logger: Logger,
    *,
    use_titulo_validator: bool = True,
) -> bool:
    """Valida entidade TITULO_ELEITOR por contexto e algoritmo.

    Args:
        text (str): Texto completo
        start_char (int): Posição inicial
        end_char (int): Posição final
        text_entity (str): Texto da entidade
        logger (Logger): Logger
        use_titulo_validator (bool): Se deve usar validador algorítmico

    Returns:
        bool: True se válido, False caso contrário
    """
    contexto = _get_context_window(text, start_char, end_char, "TITULO_ELEITOR")
    ok = any(p in contexto for p in KEYWORDS_TITULO_ELEITOR)

    if use_titulo_validator and not valida_titulo_eleitor(text_entity):
        digits = re.sub(r"\D", "", text_entity)
        if ok and 10 <= len(digits) <= 12 and len(set(digits)) > 1:  # noqa: PLR2004
            logger.debug("TITULO_ELEITOR aceito por contexto explícito: %s", text_entity)
            return True
        logger.debug("TITULO_ELEITOR rejeitado por validação algorítmica: %s", text_entity)
        return False

    logger.debug("TITULO_ELEITOR contexto_ok=%s", ok)
    return ok


@register_validator("DADOS_BANCARIOS")
def validate_dados_bancarios_context(
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,  # noqa: ARG001
    logger: Logger,
) -> bool:
    """Valida entidade DADOS_BANCARIOS por contexto.

    Args:
        text (str): Texto completo
        start_char (int): Posição inicial
        end_char (int): Posição final
        text_entity (str): Texto da entidade (não usado)
        logger (Logger): Logger

    Returns:
        bool: True se válido, False caso contrário
    """
    contexto = _get_context_window(text, start_char, end_char, "DADOS_BANCARIOS")
    ok = any(p in contexto for p in KEYWORDS_DADOS_BANCARIOS)
    logger.debug("DADOS_BANCARIOS contexto_ok=%s", ok)
    return ok


@register_validator("GEO_COORD")
def validate_geo_coord_context(  # noqa: PLR0911
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,
    logger: Logger,
) -> bool:
    """Valida entidade GEO_COORD por símbolos, formato numérico e contexto.

    Args:
        text (str): Texto completo
        start_char (int): Posição inicial
        end_char (int): Posição final
        text_entity (str): Texto da entidade
        logger (Logger): Logger

    Returns:
        bool: True se válido, False caso contrário
    """
    coord_text = text_entity.strip()

    if len(coord_text) < MIN_GEO_COORD_LENGTH:
        logger.debug("GEO_COORD rejeitado: muito curto")
        return False

    # Aceitar por presença de símbolos de coordenada
    if any(symbol in coord_text for symbol in ["°", "'", '"', "Lat", "Long", "Log"]):
        logger.debug("GEO_COORD aceito: presença de símbolos de coordenadas")
        return True

    # Aceitar formato compacto (2 dígitos + NS/WE + 6 dígitos)
    if re.match(r"^\d{2}[NSEWnsew]\d{6}$", coord_text):
        logger.debug("GEO_COORD aceito: formato compacto")
        return True

    # Aceitar par decimal separado por vírgula
    if re.match(r"^-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+$", coord_text):
        logger.debug("GEO_COORD aceito: par decimal")
        return True

    contexto = _get_context_window(text, start_char, end_char, "GEO_COORD")
    has_context = any(p in contexto for p in KEYWORDS_GEO_COORD)

    if re.match(r"^-?\d+[.,]\d+$", coord_text):
        logger.debug("GEO_COORD numérico aceito? contexto=%s", has_context)
        return has_context

    # Decimal com espaço após o sinal — aceitar apenas com contexto
    if re.match(r"^-\s\d{1,3}\.\d+$", coord_text):
        logger.debug("GEO_COORD decimal com espaço? contexto=%s", has_context)
        return has_context

    logger.debug("GEO_COORD aceito? contexto=%s", has_context)
    return has_context


@register_validator("CID")
def validate_cid_context(
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,
    logger: Logger,
) -> bool:
    """Valida entidade CID por contexto médico e formato.

    Args:
        text (str): Texto completo
        start_char (int): Posição inicial
        end_char (int): Posição final
        text_entity (str): Texto da entidade
        logger (Logger): Logger

    Returns:
        bool: True se válido, False caso contrário
    """
    contexto = _get_context_window(text, start_char, end_char, "CID")
    has_medical_context = any(p in contexto for p in KEYWORDS_CID)

    cid_text = text_entity.strip().upper() if text_entity else ""

    if cid_text in CID_FALSE_POSITIVES and not has_medical_context:
        logger.debug("CID rejeitado por falso positivo: %s", cid_text)
        return False

    if has_medical_context:
        logger.debug("CID aceito por contexto médico: %s", cid_text)
        return True

    if re.match(r"^[A-TV-Z]\d{2}(\.\d+)?([X\d])?$", cid_text):
        logger.debug("CID aceito por formato com letra e dígitos: %s", cid_text)
        return True

    if re.match(r"^[A-TV-Z]\s?\d{2}(\.\d+)?(-[A-Z\d]+)?$", cid_text):
        logger.debug("CID aceito por formato alternativo: %s", cid_text)
        return True

    if re.match(r"^[A-TV-Z]\d{2}$", cid_text):
        logger.debug("CID aceito por formato básico: %s", cid_text)
        return True

    logger.debug("CID rejeitado: %s", cid_text)
    return False


@register_validator("EMAIL")
def validate_email_context(  # noqa: PLR0911
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,
    logger: Logger,
) -> bool:
    """Valida entidade EMAIL para reduzir falsos positivos com URLs e links.

    Args:
        text (str): Texto completo
        start_char (int): Posição inicial
        end_char (int): Posição final
        text_entity (str): Texto da entidade
        logger (Logger): Logger

    Returns:
        bool: True se válido, False caso contrário
    """
    email_text = text_entity.strip().rstrip(".,;:!?)")

    # Regra 1: deve conter @
    if "@" not in email_text:
        logger.debug("EMAIL rejeitado: sem @: %s", email_text)
        return False

    # Regra 2: rejeitar se parece URL (contém / ou ://)
    if "/" in email_text:
        logger.debug("EMAIL rejeitado: contém barra (possível URL): %s", email_text)
        return False

    # Verificar contexto para URLs
    context_start = max(0, start_char - 50)
    context_end = min(len(text), end_char + 50)
    context_before = text[context_start:start_char].lower()
    context_after = text[end_char:context_end].lower()

    if (
        any(marker in context_before for marker in ["://", "http", "https", "www."])
        or email_text.startswith("www.")
        or context_after.startswith("/")
    ):
        logger.debug("EMAIL rejeitado: contexto ou prefixo/sufixo indica URL: %s", email_text)
        return False

    # Regra 3: validar formato local@dominio.tld
    parts = email_text.split("@")
    if len(parts) != 2:  # noqa: PLR2004
        logger.debug("EMAIL rejeitado: múltiplos @: %s", email_text)
        return False

    local_part, domain = parts
    if not local_part or not domain:
        logger.debug("EMAIL rejeitado: parte local ou domínio vazio: %s", email_text)
        return False

    # Domínio deve ter pelo menos um ponto e TLD com 2+ letras
    if "." not in domain:
        logger.debug("EMAIL rejeitado: domínio sem ponto: %s", email_text)
        return False

    tld = domain.rsplit(".", 1)[-1]
    if len(tld) < 2 or not tld.isalpha():  # noqa: PLR2004
        logger.debug("EMAIL rejeitado: TLD inválido: %s", tld)
        return False

    logger.debug("EMAIL aceito: %s", email_text)
    return True


@register_validator("PIS")
def validate_pis_context(
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,
    logger: Logger,
) -> bool:
    """Valida PIS/PASEP/NIT por contexto e dígito verificador.

    Args:
        text: Texto completo que contém a entidade.
        start_char: Offset inicial da entidade em ``text``.
        end_char: Offset final exclusivo da entidade em ``text``.
        text_entity: Valor candidato a PIS, PASEP ou NIT.
        logger: Logger usado para registrar a decisão.

    Returns:
        bool: ``True`` somente quando o valor tem dígito verificador válido e
        há palavra-chave de PIS, PASEP ou NIT no contexto.
    """
    contexto = _get_context_window(text, start_char, end_char, "PIS")
    has_keyword = any(p in contexto for p in KEYWORDS_PIS)
    valido = valida_pis(text_entity)
    if has_keyword and valido:
        logger.debug("PIS aceito (contexto + DV)")
        return True
    if has_keyword and not valido:
        logger.debug("PIS rejeitado (contexto ok, DV inválido): %s", text_entity)
        return False
    # PIS válido sem contexto explícito — rejeitar para evitar falsos positivos
    logger.debug("PIS rejeitado (sem contexto explícito)")
    return False


@register_validator("CNS")
def validate_cns_context(
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,
    logger: Logger,
) -> bool:
    """Valida CNS por contexto e validação algorítmica.

    Args:
        text: Texto completo que contém a entidade.
        start_char: Offset inicial da entidade em ``text``.
        end_char: Offset final exclusivo da entidade em ``text``.
        text_entity: Valor candidato a Cartão Nacional de Saúde.
        logger: Logger usado para registrar a decisão.

    Returns:
        bool: ``True`` somente quando o valor é estruturalmente válido e há
        palavra-chave de CNS no contexto.
    """
    contexto = _get_context_window(text, start_char, end_char, "CNS")
    has_cns = any(p in contexto for p in KEYWORDS_CNS)
    valido = valida_cns(text_entity)
    if has_cns and valido:
        logger.debug("CNS aceito (contexto + DV)")
        return True
    if has_cns and not valido:
        logger.debug("CNS rejeitado (contexto ok, DV inválido): %s", text_entity)
        return False
    # CNS válido sem contexto explícito — rejeitar para evitar falsos positivos
    logger.debug("CNS rejeitado (sem contexto explícito)")
    return False


@register_validator("RESERVISTA")
def validate_reservista_context(
    text: str,
    start_char: int,
    end_char: int,
    text_entity: str,  # noqa: ARG001
    logger: Logger,
) -> bool:
    """Valida RESERVISTA por contexto textual.

    Args:
        text: Texto completo que contém a entidade.
        start_char: Offset inicial da entidade em ``text``.
        end_char: Offset final exclusivo da entidade em ``text``.
        text_entity: Valor candidato; o formato já foi filtrado pelo padrão
            regex e não é revalidado nesta função.
        logger: Logger usado para registrar a decisão.

    Returns:
        bool: ``True`` se houver palavra-chave de certificado de reservista no
        contexto da entidade.
    """
    contexto = _get_context_window(text, start_char, end_char, "RESERVISTA")
    has_keyword = any(p in contexto for p in KEYWORDS_RESERVISTA)
    if not has_keyword:
        logger.debug("RESERVISTA rejeitado (sem contexto)")
        return False
    logger.debug("RESERVISTA aceito (contexto ok)")
    return True


def validate_default_context(
    text: str,  # noqa: ARG001
    start_char: int,  # noqa: ARG001
    end_char: int,  # noqa: ARG001
    text_entity: str,  # noqa: ARG001
    logger: Logger,
    label: str = "UNKNOWN",
) -> bool:
    """Validador padrão que aceita qualquer entidade sem regra específica.

    Args:
        text (str): Texto completo (não usado)
        start_char (int): Posição inicial (não usado)
        end_char (int): Posição final (não usado)
        text_entity (str): Texto da entidade (não usado)
        logger (Logger): Logger
        label (str): Label da entidade para logging

    Returns:
        bool: Sempre True
    """
    logger.debug("Entidade %s aceita por padrão (sem regra específica)", label)
    return True
