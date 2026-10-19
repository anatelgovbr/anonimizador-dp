"""Extração opcional de nomes de pessoas usando um modelo spaCy genérico.

O modelo NER treinado pelo pacote cobre documentos e contatos, mas não nomes.
Este módulo converte as entidades ``PER`` de um modelo spaCy genérico em português
(por exemplo ``pt_core_news_sm``) no label ``NOME``.
"""

from collections.abc import Callable
from logging import Logger
from typing import Any

from anonimizar._constants import NAME_LABEL

#: Label de pessoa nos modelos spaCy genéricos em português.
PERSON_LABEL = "PER"


def _parece_nome(ent: Any) -> bool:  # noqa: ANN401
    """Descarta falsos positivos típicos: palavra comum no início de frase marcada como pessoa.

    Nomes com mais de um token são aceitos. Um único token só é aceito quando a classe
    gramatical é nome próprio (``PROPN``). Se o modelo não fornece classe gramatical
    (``pos_`` vazio), o token é mantido: na dúvida, anonimiza.

    Args:
        ent (Any): Span de entidade do spaCy com label ``PER``.

    Returns:
        bool: True se o span deve ser tratado como nome de pessoa.
    """
    if len(ent) > 1:
        return True
    pos = ent[0].pos_
    return pos in ("PROPN", "")


def extract_names(
    nlp_names: Callable[[str], Any],
    text: str,
    logger: Logger,
) -> list[dict]:
    """Extrai nomes de pessoas do texto como entidades ``NOME``.

    Args:
        nlp_names (Callable): Modelo spaCy (ou callable equivalente) que retorna um ``Doc``
            com ``ents`` contendo spans ``PER``.
        text (str): Texto a ser processado.
        logger (Logger): Logger para debug.

    Returns:
        list[dict]: Entidades no formato ``label_detail``, com ``detected_by="modelo_nomes"``.
    """
    doc = nlp_names(text)

    entities = []
    descartadas = 0
    for ent in doc.ents:
        if ent.label_ != PERSON_LABEL:
            continue
        if not _parece_nome(ent):
            descartadas += 1
            continue
        entities.append(
            {
                "label": NAME_LABEL,
                "start_position": ent.start_char,
                "end_position": ent.end_char,
                "text": ent.text,
                "detected_by": "modelo_nomes",
            }
        )

    logger.debug("Nomes detectados: %d (descartados por filtro de classe gramatical: %d)", len(entities), descartadas)
    return entities
