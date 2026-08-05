"""Funções para anonimização de texto.

Este módulo fornece funções para substituir entidades sensíveis detectadas
no texto por tags de anonimização, preservando a estrutura do texto original.
"""

from logging import Logger

from anonimizar._constants import ANON_TAG_FORMAT, IGNORE_LABELS

_ANON_LOGGER_NAME = "anonimizar._anonymization.text"


def _build_substitution_list(entities_list: list[dict]) -> list[tuple[int, int, str]]:
    """Constrói lista ordenada de substituições a partir das entidades.

    Filtra entidades ignoráveis e ordena em ordem reversa (do fim para o início)
    para preservar posições corretas durante as substituições.

    Args:
        entities_list (list[dict]): Lista de entidades com start_position, end_position, label

    Returns:
        list[tuple[int, int, str]]: Lista de tuplas (start, end, label) ordenada em ordem reversa
    """
    return sorted(
        [
            (ent["start_position"], ent["end_position"], ent["label"])
            for ent in entities_list
            if ent["label"] not in IGNORE_LABELS
        ],
        key=lambda x: x[0],
        reverse=True,
    )


def _apply_substitutions(text: str, substitutions: list[tuple[int, int, str]], logger: Logger) -> str:
    """Aplica substituições de entidades no texto.

    Substitui cada entidade por sua tag de anonimização, processando em ordem
    reversa para preservar as posições corretas.

    Args:
        text (str): Texto original
        substitutions (list[tuple[int, int, str]]): Lista de (start, end, label) em ordem reversa
        logger (Logger): Logger para registrar avisos

    Returns:
        str: Texto com substituições aplicadas
    """
    for start_char, end_char, label in substitutions:
        if start_char < 0 or end_char > len(text) or start_char >= end_char:
            logger.warning(
                "Posição inválida ignorada: start=%d end=%d len=%d label=%s", start_char, end_char, len(text), label
            )
            continue
        anon_tag = ANON_TAG_FORMAT.format(label=label)
        text = text[:start_char] + anon_tag + text[end_char:]
    return text


def anonymize_text(text: str, entities_list: list[dict], logger: Logger) -> str:
    """Substitui entidades detectadas no texto por tags de anonimização.

    Processa uma lista de entidades extraídas e substitui cada ocorrência no texto
    original por uma tag formatada. As substituições são feitas em ordem reversa
    para preservar as posições corretas dos caracteres.

    Args:
        text (str): Texto original a ser anonimizado
        entities_list (list[dict]): Lista de entidades extraídas usando return_type
            "label_position" ou "label_detail". Cada entidade deve conter:
            - 'start_position': posição inicial no texto
            - 'end_position': posição final no texto
            - 'label': tipo da entidade
        logger (Logger): Logger para registrar informações de debug

    Returns:
        str: Texto anonimizado onde entidades são substituídas por tags no formato
            ``<|TIPO_ENTIDADE|>``. Exemplo: ``<|CPF|>``, ``<|EMAIL|>``.

    Note:
        - Substituições são feitas em ordem reversa (do fim para o início)
        - Entidades com labels em IGNORE_LABELS são puladas
        - Preserva formatação e espaçamento do texto original
        - Tags seguem padrão definido em ANON_TAG_FORMAT

    Examples:
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> text = "João, CPF 123.456.789-00, email: joao@email.com"
        >>> entities = [
        ...     {"label": "CPF", "start_position": 10, "end_position": 24},
        ...     {"label": "EMAIL", "start_position": 33, "end_position": 47},
        ... ]
        >>> result = anonymize_text(text, entities, logger)
        >>> result
        'João, CPF <|CPF|>, email: <|EMAIL|>'
    """
    logger.debug("Anonimizando texto: entidades recebidas=%d", len(entities_list))

    substitutions = _build_substitution_list(entities_list)
    text = _apply_substitutions(text, substitutions, logger)

    logger.debug("Anonimização concluída: substituições aplicadas=%d", len(substitutions))
    return text
