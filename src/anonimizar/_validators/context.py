"""Validadores contextuais para entidades sensíveis.

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
    KEYWORDS_CPF,
    KEYWORDS_DADOS_BANCARIOS,
    KEYWORDS_DATA_NASCIMENTO,
    KEYWORDS_DATA_NASCIMENTO_EXCLUDE,
    KEYWORDS_ENDERECO,
    KEYWORDS_GEO_COORD,
    KEYWORDS_PASSAPORTE,
    KEYWORDS_RG,
    KEYWORDS_SEI_EXCLUSION,
    KEYWORDS_SIAPE,
    KEYWORDS_TELEFONE,
    KEYWORDS_TITULO_ELEITOR,
    MAX_CPF_LENGTH,
    MAX_PASSPORT_LENGTH,
    MIN_CPF_LENGTH,
    MIN_GEO_COORD_LENGTH,
    MIN_PASSPORT_LENGTH,
)
from anonimizar._validators.documents import (
    valida_cnh,
    valida_cpf,
    valida_titulo_eleitor,
)
from anonimizar._validators.registry import register_validator


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

    if ok:
        logger.debug("RG aceito por contexto: %s", ok)
        return True

    rg_text = text_entity.strip().upper() if text_entity else ""
    if re.match(r"^\d{5,9}$", rg_text.replace(".", "").replace("-", "")):
        logger.debug("RG aceito por formato numérico: %s", rg_text)
        return True

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
    # Validar CNH apenas com algoritmo e contexto (validação de dígitos muito agressiva)
    contexto = _get_context_window(text, start_char, end_char, "CNH")

    if use_cnh_validator:
        cnh_num = re.sub(r"\D", "", text_entity)
        if len(cnh_num) == 11 and not valida_cnh(cnh_num):  # noqa: PLR2004
            logger.debug("CNH rejeitada por validacao algoritmica: %s", cnh_num)
            return False

    ok = any(p in contexto for p in KEYWORDS_CNH)
    logger.debug("CNH contexto_ok=%s", ok)
    return ok


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
    has_sei = any(p in contexto for p in KEYWORDS_SEI_EXCLUSION)
    ok = has_siape and not has_sei

    logger.debug("SIAPE contexto_ok=%s (has_siape=%s, has_sei=%s)", ok, has_siape, has_sei)
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

    if use_titulo_validator and not valida_titulo_eleitor(text_entity):
        logger.debug("TITULO_ELEITOR rejeitado por validação algorítmica: %s", text_entity)
        return False

    ok = any(p in contexto for p in KEYWORDS_TITULO_ELEITOR)
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
def validate_geo_coord_context(
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

    if any(symbol in coord_text for symbol in ["°", "'", '"', "Lat", "Long"]):
        logger.debug("GEO_COORD aceito: presença de símbolos de coordenadas")
        return True

    contexto = _get_context_window(text, start_char, end_char, "GEO_COORD")
    has_context = any(p in contexto for p in KEYWORDS_GEO_COORD)

    if re.match(r"^-?\d+[.,]\d+$", coord_text):
        logger.debug("GEO_COORD numérico aceito? contexto=%s", has_context)
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
