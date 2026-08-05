"""Extração de entidades usando modelo spaCy treinado.

Este módulo fornece funções para extrair entidades de texto usando
um modelo spaCy treinado.
"""

from collections.abc import Callable
from logging import Logger
from typing import Any

from anonimizar._constants import IGNORE_LABELS


def _create_entity_from_model(
    ent: Any,  # noqa: ANN401
    text: str,  # noqa: ARG001
    return_type: str,
    verification: bool | str,
) -> dict:
    """Cria dicionário de entidade extraída do modelo no formato solicitado.

    Args:
        ent (Any): Entidade do spaCy (Span object)
        text (str): Texto completo
        return_type (str): Formato de retorno
        verification (bool | str): Resultado da verificação contextual

    Returns:
        dict: Dicionário de entidade no formato solicitado
    """
    final_label = ent.label_ if str(verification) == "True" else verification

    if return_type == "label_position":
        return {
            "label": final_label,
            "start_position": ent.start_char,
            "end_position": ent.end_char,
        }
    if return_type == "label_text":
        return {
            "label": final_label,
            "text": ent.text,
        }
    # label_detail
    return {
        "label": final_label,
        "start_position": ent.start_char,
        "end_position": ent.end_char,
        "text": ent.text,
        "detected_by": "modelo",
    }


def extract_from_model(
    nlp_trained: Any,  # noqa: ANN401
    text: str,
    labels: set[str],
    return_type: str,
    verify_fn: Callable[[Any, str], bool | str],
    logger: Logger,
) -> list[dict]:
    """Extrai entidades do texto usando modelo spaCy treinado.

    Args:
        nlp_trained (Any): Modelo spaCy treinado
        text (str): Texto a ser processado
        labels (set[str]): Labels ativos para filtrar entidades
        return_type (str): Formato de retorno ("label_position", "label_text", "label_detail")
        verify_fn (Callable): Função de verificação contextual
        logger (Logger): Logger para debug

    Returns:
        list[dict]: Lista de entidades extraídas pelo modelo

    Note:
        - Filtra por labels ativos e ignora IGNORE_LABELS
        - Aplica verificação contextual a cada entidade
        - Descarta entidades que falham na verificação
    """
    doc_trained = nlp_trained(text)

    entities_from_model = []
    for ent in doc_trained.ents:
        if ent.label_ in labels and ent.label_ not in IGNORE_LABELS:
            verification = verify_fn(ent, text)
            if str(verification) != "False":
                entity_dict = _create_entity_from_model(ent, text, return_type, verification)
                entities_from_model.append(entity_dict)

    logger.debug("Entidades detectadas pelo modelo: %d", len(entities_from_model))
    return entities_from_model
