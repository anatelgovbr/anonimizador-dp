"""Validação unificada de entidades usando Strategy Pattern.

Este módulo fornece o dispatcher que usa o VALIDATOR_REGISTRY para
delegar a validação ao validator correto baseado no label da entidade.
"""

from logging import Logger
from typing import Any

import anonimizar._validators.context  # noqa: F401
from anonimizar._validators.documents import valida_cnpj
from anonimizar._validators.registry import get_validator


def verify_entities_unified(
    entity: Any,  # noqa: ANN401
    text: str,
    logger: Logger,
    *,
    use_cpf_validator: bool = True,
    use_cnh_validator: bool = True,
    use_titulo_validator: bool = True,
) -> bool | str:
    """Valida uma entidade detectada usando contexto textual e regras por tipo.

    Utiliza o VALIDATOR_REGISTRY para delegar a validação ao validator
    apropriado baseado no label da entidade. Aplica uma guarda global
    para descartar CNPJs detectados erroneamente como outros labels.

    Args:
        entity (Any): Dicionário ou objeto que fornece ``label_``, ``start_char``
            e ``end_char``. Os offsets devem ser inteiros utilizáveis para
            indexar ``text``.
        text (str): Texto completo onde a entidade foi encontrada
        logger (Logger): Logger para debug
        use_cpf_validator (bool): Se True, aplica validação algorítmica de CPF
        use_cnh_validator (bool): Se True, aplica validação algorítmica de CNH
        use_titulo_validator (bool): Se True, aplica validação algorítmica de Título

    Returns:
        bool | str:
            - True: Se a entidade é válida
            - False: Se deve ser descartada
            - str: Novo label para reclassificação (ex: "CNH")

    Raises:
        AttributeError: Se ``label_`` estiver ausente ou não for uma string.
        TypeError: Se os offsets não puderem ser usados para indexar ``text``.
    """
    label = _get_attr(entity, "label_")
    start_char = _get_attr(entity, "start_char")
    end_char = _get_attr(entity, "end_char")
    text_entity = text[start_char:end_char]

    logger.debug("Verificando entidade: label=%s span=(%d,%d) text='%s'", label, start_char, end_char, text_entity)

    # Guarda global: descartar CNPJs detectados como outro label
    if label.lower() != "cnpj" and valida_cnpj(text_entity.lower()):
        logger.debug("Descartando por ser CNPJ válido no span (não esperado para label=%s)", label)
        return False

    validator = get_validator(label)
    if validator is None:
        # Label sem validator registrado — aceitar incondicionalmente
        logger.debug("Entidade %s aceita por padrão (sem regra específica)", label)
        return True

    # Preparar kwargs com os validadores habilitados
    kwargs = {}
    if "use_cpf_validator" in validator.__code__.co_varnames:
        kwargs["use_cpf_validator"] = use_cpf_validator
    if "use_cnh_validator" in validator.__code__.co_varnames:
        kwargs["use_cnh_validator"] = use_cnh_validator
    if "use_titulo_validator" in validator.__code__.co_varnames:
        kwargs["use_titulo_validator"] = use_titulo_validator

    return validator(
        text,
        start_char,
        end_char,
        text_entity,
        logger,
        **kwargs,
    )


def _get_attr(entity: Any, attr: str) -> Any:  # noqa: ANN401
    """Obtém atributo de dict ou objeto.

    Args:
        entity (Any): Entidade (dict ou objeto)
        attr (str): Nome do atributo

    Returns:
        Any: Valor do atributo ou ``None`` se não encontrado. A ausência dos
        campos obrigatórios é tratada pela operação de indexação do chamador.
    """
    if isinstance(entity, dict):
        return entity.get(attr)
    return getattr(entity, attr, None)
