"""Funções para registro de padrões regex customizados.

Este módulo fornece funcionalidade para adicionar padrões personalizados
para detecção de entidades específicas não cobertas pelos padrões embutidos.
"""

import logging
import re


def add_custom_pattern(
    patterns: list[dict],
    label: str,
    regex_pattern: str,
    description: str,
    logger: logging.Logger,
) -> None:
    r"""Adiciona um padrão REGEX customizado para detecção de entidades.

    Permite criar padrões personalizados para detectar entidades que não são
    cobertas pelos padrões pré-definidos do sistema.

    Args:
        patterns: Lista de padrões a ser modificada in-place.
        label: Nome/rótulo da entidade (será convertido para maiúsculas).
        regex_pattern: Expressão regular para detectar a entidade.
        description: Descrição opcional do padrão para documentação.
        logger: Logger para mensagens de debug e erro.

    Raises:
        ValueError: Se regex_pattern for vazio ou inválido.

    Examples:
        >>> patterns = []
        >>> logger = logging.getLogger()
        >>> add_custom_pattern(
        ...     patterns, "CODIGO_PRODUTO", r"PROD-\\d{4}", "Código", logger
        ... )
    """
    try:
        if not isinstance(regex_pattern, str) or not regex_pattern.strip():
            msg = "O padrão REGEX deve ser uma string não vazia"
            logger.exception("Padrão inválido para label '%s': %s", label, msg)
            raise ValueError(msg)  # noqa: TRY301

        re.compile(regex_pattern)

        patterns.append(
            {
                "label": label.upper(),
                "pattern": {"REGEX": regex_pattern},
                "description": description,
            }
        )
        logger.debug("Padrão customizado adicionado: %s - %s", label, description)

    except re.error as e:
        msg = f"Regex inválido para label '{label}': {e}"
        logger.exception(msg)
        raise ValueError(msg) from e
    except Exception as e:
        logger.exception("Erro inesperado ao adicionar padrão '%s': %s", label, e)  # noqa: TRY401
        raise
