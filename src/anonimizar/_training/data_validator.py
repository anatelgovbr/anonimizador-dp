"""Validação de dados para treinamento NER.

Este módulo contém funções para validar dados de treinamento,
incluindo validação BILUO, detecção de conflitos e limpeza de entidades.
"""

import logging
from typing import Any

import spacy
from spacy.training.iob_utils import offsets_to_biluo_tags


def validate_entities(
    text: str,
    entities: list[tuple[int, int, str]],
    supported_labels: list[str],
    nlp: spacy.Language,
    logger: logging.Logger,
) -> bool:
    """Valida entidades anotadas com base no texto.

    Retorna True se todas estiverem corretas,
    ou False se houver qualquer erro (offset inválido, label não suportado, desalinhamento BILUO etc.).

    Args:
        text: Texto completo a ser inspecionado.
        entities: Lista de tuplas (start, end, label).
        supported_labels: Lista de labels suportados.
        nlp: Pipeline spaCy para validação.
        logger: Logger para mensagens.

    Returns:
        bool: True se todas as entidades forem válidas, False caso contrário.
    """
    text_len = len(text)
    doc = nlp.make_doc(text)

    for start, end, label in entities:
        if start >= end:
            logger.debug("validate_entities=False: posição inicial >= final (%d >= %d)", start, end)
            return False
        if start < 0:
            logger.debug("validate_entities=False: posição inicial negativa (%d)", start)
            return False
        if start >= text_len or end > text_len:
            logger.debug("validate_entities=False: offset fora do texto (%d-%d) len=%d", start, end, text_len)
            return False

        entity_text = text[start:end]

        if entity_text.startswith(" ") or entity_text.endswith(" "):
            logger.debug(
                "validate_entities=False: espaços nas extremidades da entidade '%s' (%d-%d)",
                entity_text,
                start,
                end,
            )
            return False

        if label not in supported_labels:
            logger.debug("validate_entities=False: label não suportado '%s'", label)
            return False

        try:
            biluo = offsets_to_biluo_tags(doc, [(start, end, label)])
            if "-" in biluo:
                logger.debug("validate_entities=False: BILUO inválido para '%s' (%d-%d)", entity_text, start, end)
                return False
        except Exception as e:
            logger.debug("validate_entities=False: exceção na validação BILUO (%s)", e)
            return False

    return True


def validate_data(
    data_to_validate: list[tuple[str, dict[str, Any]]],
    supported_labels: list[str],
    blank_nlp: spacy.Language,
    logger: logging.Logger,
    *,
    errors: str = "raise",
    keep_empty_entities: bool = False,
    skip_biluo: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    """Valida os dados de treinamento fornecidos.

    Args:
        data_to_validate: Lista de tuplas (text, annotations) para validar.
        supported_labels: Lista de labels suportados.
        blank_nlp: Pipeline spaCy em branco para validação.
        logger: Logger para mensagens.
        errors: 'raise' para lançar erro, 'coerce' para corrigir, 'ignore' para ignorar.
        keep_empty_entities: Se True, mantém textos mesmo quando todas as entidades são inválidas.
        skip_biluo: Se True, não faz validação de BILUO.

    Returns:
        list: Lista de dados validados no formato [(text, {"entities": [...]}), ...]

    Raises:
        ValueError: Quando errors="raise" e é encontrada inconsistência.
    """
    valid_data = []

    for text, annotations in data_to_validate:
        entities = annotations.get("entities", [])
        original_entities_count = len(entities)
        valid_entities = []

        for start, end, label in entities:
            if label not in supported_labels:
                if errors == "raise":
                    msg = f"Label inválido: {label}"
                    logger.exception(msg)
                    raise ValueError(msg)
                if errors in {"coerce", "ignore"}:
                    logger.debug("Entidade descartada (label inválido) [%s] em modo %s.", label, errors)
                    continue

            if start < 0 or end > len(text) or start >= end:
                if errors == "raise":
                    msg = f"Posições inválidas: {start}-{end} em texto de tamanho {len(text)}"
                    logger.exception(msg)
                    raise ValueError(msg)
                if errors == "coerce":
                    logger.debug("Entidade descartada (offset inválido %d-%d) em modo coerce.", start, end)
                    continue
                if errors == "ignore":
                    logger.debug("Entidade mantida apesar de offset inválido (modo ignore): %d-%d.", start, end)
                    valid_entities.append((start, end, label))
                    continue

            valid_entities.append((start, end, label))

        if valid_entities:
            before = len(valid_entities)
            if not skip_biluo:
                valid_entities = validate_biluo_tags(text, valid_entities, blank_nlp, logger, errors)
            removed = before - len(valid_entities)
            if removed > 0 and errors in {"coerce", "ignore"}:
                logger.debug("Entidades removidas por BILUO (%d) em modo %s.", removed, errors)

        if original_entities_count == 0 or valid_entities or keep_empty_entities:
            valid_data.append((text, {"entities": valid_entities}))
        else:
            logger.debug("Texto descartado por não manter entidades após validação: '%.50s...'", text)

    logger.debug(f"Validação concluída. {len(valid_data)} de {len(data_to_validate)} exemplos mantidos.")
    return valid_data


def validate_biluo_tags(
    text: str,
    entities: list[tuple[int, int, str]],
    blank_nlp: spacy.Language,
    logger: logging.Logger,
    errors: str = "raise",
) -> list[tuple[int, int, str]]:
    """Valida entidades usando o esquema BILUO do spaCy.

    Args:
        text: Texto a ser validado.
        entities: Lista de tuplas (start, end, label).
        blank_nlp: Pipeline spaCy em branco.
        logger: Logger para mensagens.
        errors: "raise" - lança exceção, "coerce" - remove entidade problemática, "ignore" - mantém todas.

    Returns:
        list: Lista de entidades válidas.

    Raises:
        ValueError: Se errors="raise" e algum problema BILUO é encontrado.
    """
    if not entities:
        return entities

    doc = blank_nlp.make_doc(text)
    valid_entities = []

    for _i, entity in enumerate(entities):
        start, end, label = entity
        single_entity_list = [(start, end, label)]
        biluo_tags = None

        try:
            biluo_tags = offsets_to_biluo_tags(doc, single_entity_list)
        except Exception as e:
            if errors == "raise":
                msg = f"Erro BILUO na entidade {entity} no texto: '{text[:50]}...'"
                logger.exception(msg)
                raise ValueError(msg) from e
            if errors == "coerce":
                logger.debug(f"Entidade descartada (erro BILUO): {start}-{end} '{text[start:end]}' - {e!s}")
                continue
            if errors == "ignore":
                logger.debug(f"Entidade descartada (erro BILUO - ignore): {start}-{end} '{text[start:end]}' - {e!s}")
                continue

        if "-" in biluo_tags:
            if errors == "raise":
                msg = f"Erro BILUO na entidade {entity} no texto: '{text[:50]}...'"
                logger.exception(msg)
                raise ValueError(msg)
            if errors == "coerce":
                logger.debug(f"Entidade descartada (BILUO inválido): {start}-{end} '{text[start:end]}'")
                continue
            if errors == "ignore":
                logger.debug(
                    f"Entidade mantida apesar de BILUO inválido (modo ignore): %d-%d '%s'", start, end, text[start:end]
                )
                valid_entities.append(entity)
        else:
            valid_entities.append(entity)

    return valid_entities


def detect_entity_conflicts(entities: list[tuple[int, int, str]]) -> dict[str, Any]:
    """Detecta conflitos entre entidades (duplicatas e sobreposições).

    Args:
        entities: Lista de entidades (start, end, label).

    Returns:
        dict: Informações sobre conflitos encontrados:
            - 'duplicates': lista de índices de entidades duplicadas
            - 'overlaps': lista de pares de índices com sobreposição
            - 'has_conflicts': bool indicando se há conflitos
    """
    duplicates = []
    overlaps = []

    seen = {}
    if len(entities) != len(list(set(entities))):
        for i, (start, end, label) in enumerate(entities):
            key = (start, end, label)
            if key in seen:
                duplicates.append(i)
            else:
                seen[key] = i

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            start1, end1, _ = entities[i]
            start2, end2, _ = entities[j]

            if not (end1 <= start2 or end2 <= start1):
                if i not in duplicates and j not in duplicates:
                    overlaps.append((i, j))

    return {
        "duplicates": duplicates,
        "overlaps": overlaps,
        "has_conflicts": len(duplicates) > 0 or len(overlaps) > 0,
    }


def resolve_entity_conflicts(
    entities: list[tuple[int, int, str]],
    conflicts: dict[str, Any],
    logger: logging.Logger,
) -> list[tuple[int, int, str]]:
    """Tenta resolver conflitos de entidades automaticamente.

    Args:
        entities: Lista original de entidades.
        conflicts: Informações sobre conflitos detectados.
        logger: Logger para mensagens.

    Returns:
        list: Lista de entidades sem conflitos ou lista vazia se não conseguir resolver.
    """
    resolved_entities = entities.copy()

    if conflicts["duplicates"]:
        for idx in sorted(conflicts["duplicates"], reverse=True):
            logger.debug(f"Removendo entidade duplicada: {resolved_entities[idx]}")
            resolved_entities.pop(idx)

    if conflicts["overlaps"]:
        overlaps_after_dedup = detect_entity_conflicts(resolved_entities)["overlaps"]
        if overlaps_after_dedup:
            logger.debug(f"Sobreposições persistem após remoção de duplicatas: {len(overlaps_after_dedup)}")
            for i, j in overlaps_after_dedup:
                entity_i = resolved_entities[i]
                entity_j = resolved_entities[j]
                logger.warning(
                    f"Sobreposição detectada:\n"
                    f"  Entidade 1: {entity_i[2]:15} [{entity_i[0]:4}-{entity_i[1]:4}]\n"
                    f"  Entidade 2: {entity_j[2]:15} [{entity_j[0]:4}-{entity_j[1]:4}]"
                )
            return []

    return resolved_entities


def clean_entities(
    text: str,
    entities: list[tuple[int, int, str]],
    supported_labels: list[str],
    nlp: spacy.Language,
    logger: logging.Logger,
    strict: bool = True,
    resolve_conflicts: str = "coerce",
    errors: str = "coerce",
) -> list[tuple[int, int, str]]:
    """Limpa e corrige entidades automaticamente, incluindo resolução de conflitos.

    Args:
        text: Texto completo onde as entidades estão localizadas.
        entities: Lista de entidades no formato (start, end, label).
        supported_labels: Lista de labels suportados.
        nlp: Pipeline spaCy.
        logger: Logger para mensagens.
        strict: Se True, descarta exemplos onde alguma entidade foi removida.
        resolve_conflicts: Como lidar com conflitos:
            - 'raise': lança erro quando há conflitos
            - 'ignore': retorna lista vazia (remove documento)
            - 'coerce': tenta resolver conflitos automaticamente
        errors: Política de erros para validação BILUO:
            - 'raise': lança erro quando há problemas BILUO
            - 'coerce': remove entidades problemáticas
            - 'ignore': mantém entidades como estão (sem correção)

    Returns:
        list: Lista de entidades limpas e válidas.

    Raises:
        ValueError: Se resolve_conflicts='raise' e conflitos forem encontrados.
    """
    if not entities:
        return entities

    conflicts = detect_entity_conflicts(entities)

    if conflicts["has_conflicts"]:
        logger.debug(
            f"Conflitos detectados: {len(conflicts['duplicates'])} duplicatas, {len(conflicts['overlaps'])} sobreposições"
        )

        if resolve_conflicts == "raise":
            duplicates_info = [entities[i] for i in conflicts["duplicates"]]
            overlaps_info = [(entities[i], entities[j]) for i, j in conflicts["overlaps"]]
            error_msg = (
                f"Conflitos de entidades detectados. Duplicatas: {duplicates_info}, Sobreposições: {overlaps_info}"
            )
            logger.exception(error_msg)
            raise ValueError(error_msg)

        if resolve_conflicts == "ignore":
            logger.debug("Documento removido devido a conflitos de entidades")
            return []

        if resolve_conflicts == "coerce":
            entities = resolve_entity_conflicts(entities, conflicts, logger)
            if not entities:
                logger.debug("Não foi possível resolver conflitos - documento removido")
                return []

    cleaned_entities = []
    doc = nlp.make_doc(text)

    for start, end, label in entities:
        if start >= len(text) or end > len(text) or start >= end:
            if errors == "raise":
                msg = f"Posições inválidas: {start}-{end} em texto de tamanho {len(text)}"
                logger.exception(msg)
                raise ValueError(msg)
            if errors == "ignore":
                logger.debug("Entidade mantida apesar de offset inválido (modo ignore): %d-%d", start, end)
                cleaned_entities.append((start, end, label))
                continue
            continue

        entity_text = text[start:end]

        stripped = entity_text.strip()
        if stripped != entity_text:
            if errors == "ignore":
                cleaned_entities.append((start, end, label))
                logger.debug(f"Entidade mantida com espaços (ignore): '{entity_text}'")
            else:
                left_spaces = len(entity_text) - len(entity_text.lstrip())
                right_spaces = len(entity_text) - len(entity_text.rstrip())
                new_start = start + left_spaces
                new_end = end - right_spaces

                if new_start < new_end <= len(text):
                    cleaned_entities.append((new_start, new_end, label))
                    logger.debug(f"Entidade corrigida: '{entity_text}' -> '{text[new_start:new_end]}'")
        else:
            if errors == "ignore":
                cleaned_entities.append((start, end, label))
            else:
                try:
                    biluo = offsets_to_biluo_tags(doc, [(start, end, label)])
                    if "-" not in biluo:
                        cleaned_entities.append((start, end, label))
                    else:
                        if errors == "raise":
                            msg = f"Offsets desalinhados aos tokens (BILUO inválido): {start}-{end} '{entity_text}'"
                            logger.exception(msg)
                            raise ValueError(msg)
                        logger.debug(f"Entidade descartada por desalinhamento BILUO: '{entity_text}' {start}-{end}")
                except Exception as e:
                    if errors == "raise":
                        logger.exception("Erro BILUO na entidade %d-%d '%s': %s", start, end, entity_text, e)
                        raise
                    logger.debug(f"Entidade descartada por erro de validação: '{entity_text}' {start}-{end} - {e!s}")

    final_conflicts = detect_entity_conflicts(cleaned_entities)
    if final_conflicts["has_conflicts"]:
        if resolve_conflicts in ["raise", "ignore"]:
            logger.warning("Conflitos persistem após limpeza - aplicando política configurada")
            if resolve_conflicts == "raise":
                raise ValueError("Conflitos persistem após limpeza automática")
            return []
        logger.warning("Conflitos persistem após limpeza - removendo documento")
        return []

    if strict and len(cleaned_entities) != len(entities):
        return []

    return cleaned_entities


def debug_entities(
    text: str,
    entities: list[tuple[int, int, str]],
    supported_labels: list[str],
    nlp: spacy.Language,
    logger: logging.Logger,
) -> None:
    """Mostra detalhes de offsets/labels e verifica consistência BILUO.

    Args:
        text: Texto completo a ser inspecionado.
        entities: Lista (start, end, label).
        supported_labels: Lista de labels suportados.
        nlp: Pipeline spaCy.
        logger: Logger para mensagens.
    """
    logger.info(f"Texto: '{text}'")
    text_len = len(text)
    logger.info(f"Tamanho do texto: {text_len}")
    logger.info("-" * 50)

    doc = nlp.make_doc(text)

    for i, (start, end, label) in enumerate(entities):
        entity_text = text[start:end]
        logger.info(f"Entidade {i}: ({start}, {end}, '{label}')")
        logger.info(f"  Texto da entidade: '{entity_text}'")
        if start >= text_len:
            logger.info("  PROBLEMA: Entidade começa após o final do texto")
        else:
            logger.info(f"  Caractere inicial: '{text[start]}' (posição {start})")

        if end > text_len:
            logger.info("  PROBLEMA: Entidade termina após o final do texto")
        else:
            logger.info(f"  Caractere final: '{text[end - 1]}' (posição {end - 1})")

        if entity_text.startswith(" "):
            logger.info("  PROBLEMA: Entidade começa com espaço")
        if entity_text.endswith(" "):
            logger.info("  PROBLEMA: Entidade termina com espaço")
        if label not in supported_labels:
            logger.info("  PROBLEMA: Entidade com label não suportado")
        try:
            biluo = offsets_to_biluo_tags(doc, [(start, end, label)])
            if "-" in biluo:
                logger.info("  PROBLEMA: Offsets desalinhados aos tokens (BILUO inválido)")
        except Exception as e:  # noqa: BLE001
            logger.info(f"  PROBLEMA: Falha ao validar BILUO ({e!s})")
