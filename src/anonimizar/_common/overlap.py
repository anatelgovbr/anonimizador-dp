"""Funções para resolução de overlaps entre entidades detectadas."""

import logging
from typing import Any

from anonimizar._constants import ENTITY_PRIORITY_LIST


def spans_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Verifica se dois spans se sobrepõem.

    Args:
        start1: Posição inicial do primeiro span.
        end1: Posição final do primeiro span.
        start2: Posição inicial do segundo span.
        end2: Posição final do segundo span.

    Returns:
        True se os spans se sobrepõem, False caso contrário.
    """
    return not (end1 <= start2 or start1 >= end2)


def remove_overlap_positions(
    entities: list[dict[str, Any]],
    priority_list: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Remove as entidades com overlap deixando apenas a entidade com maior abrangência.

    Regras de desempate:
        1. Entidades totalmente contidas por outra são descartadas.
        2. Se start/end coincidirem, usa a ordem definida em `priority_list`.
        3. Empates fora da lista de prioridade → mantém a primeira.

    Args:
        entities: Lista com as entidades detectadas. Cada entidade deve ter
            'start_position', 'end_position' e 'label'.
        priority_list: Lista de prioridade de labels. Se None, usa ENTITY_PRIORITY_LIST.
        logger: Logger para debug. Se None, usa logger nulo.

    Returns:
        Lista de entidades sem overlaps.
    """
    if priority_list is None:
        priority_list = ENTITY_PRIORITY_LIST
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.debug("Removendo overlaps: recebidas=%d", len(entities))

    entities_with_positions = [e for e in entities if "start_position" in e and "end_position" in e]
    entities_without_positions = [e for e in entities if "start_position" not in e or "end_position" not in e]

    entities_with_positions.sort(
        key=lambda x: (
            x["start_position"],
            -x["end_position"],
            priority_list.index(x["label"]) if x["label"] in priority_list else float("inf"),
        )
    )

    resultado = []

    for ent in entities_with_positions:
        sobreposicao_encontrada = False
        for existente in resultado:
            if _spans_overlap_check(ent, existente):
                action = _resolve_overlap(ent, existente, priority_list, logger)
                if action == "skip_ent":
                    sobreposicao_encontrada = True
                    break
                if action == "remove_existing":
                    resultado.remove(existente)
                    break

        if not sobreposicao_encontrada:
            resultado.append(ent)

    kept = len(resultado) + len(entities_without_positions)
    removed = len(entities) - kept
    logger.debug("Overlaps resolvidos: mantidas=%d removidas=%d", kept, removed)
    return resultado + entities_without_positions


def _spans_overlap_check(ent: dict[str, Any], existente: dict[str, Any]) -> bool:
    """Verifica se duas entidades se sobrepõem."""
    return not (
        ent["end_position"] <= existente["start_position"] or ent["start_position"] >= existente["end_position"]
    )


def _resolve_overlap(
    ent: dict[str, Any],
    existente: dict[str, Any],
    priority_list: list[str],
    logger: logging.Logger,
) -> str:
    """Resolve o overlap entre duas entidades.

    Returns:
        "skip_ent": pular a entidade ent
        "remove_existing": remover existente e continuar processando ent
        "continue": continuar processando sem ação
    """
    # Total overlap - spans iguais
    if _is_equal_span(ent, existente):
        return _handle_equal_spans(ent, existente, priority_list, logger)

    # ent contido em existente
    if _is_contained(ent, existente):
        logger.debug("Descartando ent contido em existente: ent=%s existente=%s", ent, existente)
        return "skip_ent"

    # existente contido em ent
    if _is_contained(existente, ent):
        logger.debug("Removendo existente contido em ent: existente=%s", existente)
        return "remove_existing"

    # ajuste de início para consolidar (só entre labels iguais)
    if _should_merge(ent, existente):
        if ent.get("label") == existente.get("label"):
            logger.debug("Ajustando início de ent para início de existente: ent=%s existente=%s", ent, existente)
            ent["start_position"] = existente["start_position"]
            if "text" in ent:
                ent["text"] = ent.get("text", "")  # será recalculado pelo chamador se necessário
            return "remove_existing"
        # Labels diferentes: usar prioridade
        return _handle_equal_spans(ent, existente, priority_list, logger)

    return "continue"


def _is_equal_span(ent: dict[str, Any], existente: dict[str, Any]) -> bool:
    """Verifica se dois spans são iguais."""
    return existente["start_position"] == ent["start_position"] and existente["end_position"] == ent["end_position"]


def _is_contained(inner: dict[str, Any], outer: dict[str, Any]) -> bool:
    """Verifica se inner está contido em outer."""
    return outer["start_position"] <= inner["start_position"] and outer["end_position"] >= inner["end_position"]


def _should_merge(ent: dict[str, Any], existente: dict[str, Any]) -> bool:
    """Verifica se ent deve ser mesclado com existente."""
    return ent["start_position"] >= existente["start_position"] and ent["end_position"] > existente["end_position"]


def _handle_equal_spans(
    ent: dict[str, Any],
    existente: dict[str, Any],
    priority_list: list[str],
    logger: logging.Logger,
) -> str:
    """Lida com spans iguais usando lista de prioridade."""
    ent_in_priority = ent["label"] in priority_list
    existente_in_priority = existente["label"] in priority_list

    # Ambos na lista de prioridade
    if ent_in_priority and existente_in_priority:
        if priority_list.index(existente["label"]) <= priority_list.index(ent["label"]):
            logger.debug("Mantendo existente e descartando ent por prioridade: %s vs %s", existente, ent)
            return "skip_ent"
        logger.debug("Removendo existente por prioridade: %s", existente)
        return "remove_existing"

    # Apenas existente na lista de prioridade
    if existente_in_priority:
        logger.debug("Descartando ent por existente em prioridade: %s", existente)
        return "skip_ent"

    # Nenhum ou apenas ent na lista de prioridade
    logger.debug("Removendo existente sem prioridade: %s", existente)
    return "remove_existing"
