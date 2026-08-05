"""Extração de entidades usando padrões regex.

Este módulo fornece funções para extrair entidades de texto usando
padrões de expressão regular (regex) configurados.
"""

import re
from collections.abc import Callable
from logging import Logger


def _adjust_positions(match: re.Match) -> tuple[int, int]:
    """Ajusta posições para remover espaços nas bordas.

    Args:
        match (re.Match): Match object do regex

    Returns:
        tuple[int, int]: Posições (start, end) ajustadas
    """
    start = match.span()[0]
    end = match.span()[1]

    if match.group().startswith(" "):
        start += 1
    if match.group().endswith(" "):
        end -= 1

    return start, end


def _create_entity_dict(
    label: str,
    start: int,
    end: int,
    text: str,
    regex_obj: re.Pattern,
    return_type: str,
) -> dict:
    """Cria dicionário de entidade no formato solicitado.

    Args:
        label (str): Label da entidade
        start (int): Posição inicial
        end (int): Posição final
        text (str): Texto da entidade
        regex_obj (re.Pattern): Objeto regex usado
        return_type (str): Formato de retorno

    Returns:
        dict: Dicionário de entidade no formato solicitado
    """
    if return_type == "label_text":
        return {"label": label, "text": text}
    if return_type == "label_position":
        return {"label": label, "start_position": start, "end_position": end}
    # label_detail
    return {
        "label": label,
        "start_position": start,
        "end_position": end,
        "text": text,
        "detected_by": regex_obj,
    }


def extract_entities_regex_re(
    text: str,
    patterns: list[dict],
    return_type: str,
    verify_fn: Callable[[dict, str], bool | str],
    logger: Logger,
) -> list[dict]:
    """Extrai entidades do texto usando padrões regex configurados.

    Aplica todos os padrões regex ao texto e valida cada entidade encontrada
    através da função de verificação fornecida.

    Args:
        text (str): Texto a ser processado
        patterns (list[dict]): Lista de padrões regex com estrutura:
            [{"label": str, "pattern": {"REGEX": str}}]
        return_type (str): Formato de retorno ("label_text", "label_position", "label_detail")
        verify_fn (Callable): Função de verificação contextual
        logger (Logger): Logger para debug

    Returns:
        list[dict]: Lista de entidades encontradas e validadas

    Note:
        - Cada padrão regex é aplicado independentemente
        - Validação contextual é aplicada via verify_fn
        - Entidades inválidas são automaticamente filtradas
        - Posições são ajustadas para remover espaços extras
    """
    resultado = []
    total_by_label: dict[str, int] = {}

    for pattern in patterns:
        entidades_regex = []
        for match in re.finditer(pattern["pattern"]["REGEX"], text):
            start, end = _adjust_positions(match)
            entidades_regex.append(
                {
                    "label": pattern["label"],
                    "start_char": start,
                    "end_char": end,
                    "text": match.group().strip(),
                    "regex": match.re,
                }
            )

        if entidades_regex:
            total_by_label[pattern["label"]] = total_by_label.get(pattern["label"], 0) + len(entidades_regex)

        for entidade_regex in entidades_regex:
            verifica_entidade = verify_fn(
                {
                    "label_": entidade_regex["label"],
                    "start_char": entidade_regex["start_char"],
                    "end_char": entidade_regex["end_char"],
                },
                text,
            )

            if str(verifica_entidade) != "False":
                # Usar label reclassificado se houver, senão manter original
                final_label = entidade_regex["label"] if str(verifica_entidade) == "True" else str(verifica_entidade)

                entity_dict = _create_entity_dict(
                    final_label,
                    entidade_regex["start_char"],
                    entidade_regex["end_char"],
                    entidade_regex["text"],
                    entidade_regex["regex"],
                    return_type,
                )
                resultado.append(entity_dict)

    for k, v in total_by_label.items():
        logger.debug("Regex label=%s matches=%d", k, v)

    logger.debug("Total de entidades via regex: %d", len(resultado))
    return resultado
