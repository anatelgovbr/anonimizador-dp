"""Gerenciamento de registro de padrões de detecção.

Este módulo fornece funções para adicionar, aplicar e listar padrões de detecção
de entidades, utilizando o registro PATTERN_ADDERS dos padrões pré-definidos.
"""

from collections import Counter
from logging import Logger

from anonimizar._patterns.builtin import (
    PATTERN_ADDERS,
    add_pattern_passaporte_est,
)
from anonimizar._patterns.custom import add_custom_pattern


def get_active_labels(patterns: list[dict], labels: list[str]) -> dict:
    """Retorna informações detalhadas sobre labels ativos.

    Args:
        patterns (list[dict]): Lista de padrões regex registrados
        labels (list[str]): Lista de labels ativos para detecção por modelo

    Returns:
        dict: Dicionário contendo:
            - "model_labels": Lista de labels ativos para detecção por modelo spaCy
            - "regex_summary": Dicionário com labels e quantidade de padrões regex por label
            - "regex_patterns": Número total de padrões regex registrados
            - "total_patterns": Soma de labels do modelo com total de padrões regex
    """
    regex_counter = Counter([p["label"] for p in patterns])
    return {
        "model_labels": list(labels),
        "regex_summary": dict(regex_counter),
        "regex_patterns": sum(regex_counter.values()),
        "total_patterns": len(labels) + sum(regex_counter.values()),
    }


def add_apply_patterns(  # noqa: C901, PLR0912
    patterns: list[dict],
    labels: list[str] | None,
    model_labels: list[str],
    logger: Logger,
    *,
    replace_patterns: bool = False,
    use_model_labels: bool = False,
    foreign_passport: bool = False,
    custom_patterns: list[dict] | None = None,
) -> dict:
    """Adiciona e aplica padrões de regex para detecção de entidades.

    Utiliza o registro PATTERN_ADDERS para aplicar padrões pré-definidos de forma
    simplificada, além de suportar padrões customizados.

    Args:
        patterns (list[dict]): Lista de padrões a ser modificada (in-place)
        labels (list[str] | None): Lista de rótulos dos padrões pré-definidos
        model_labels (list[str]): Labels ativos no modelo spaCy
        logger (Logger): Logger para registrar informações
        replace_patterns (bool): Se True, limpa padrões anteriores antes de adicionar
        use_model_labels (bool): Se True, combina labels do modelo com os fornecidos
        foreign_passport (bool): Se True, adiciona padrões de passaporte estrangeiro
        custom_patterns (list[dict] | None): Lista de padrões customizados

    Returns:
        dict: Informações sobre labels ativos (via get_active_labels)

    Raises:
        ValueError: Se labels for vazio/None quando use_model_labels=False
        ValueError: Se algum label não for suportado
        ValueError: Se padrão customizado não tiver label/regex
    """
    logger.info(
        "Aplicando padrões: labels=%s, replace_patterns=%s, use_model_labels=%s, foreign_passport=%s",
        labels,
        replace_patterns,
        use_model_labels,
        foreign_passport,
    )

    if replace_patterns:
        logger.debug("replace_patterns=True: limpando padrões anteriores (qtd=%d)", len(patterns))
        patterns.clear()

    if not use_model_labels:
        if not labels:
            msg = "Você deve fornecer ao menos um label quando 'use_model_labels' for False."
            logger.error(msg)
            raise ValueError(msg)
    else:
        combined = list(set(model_labels or []) | set(labels or []))
        logger.debug("Labels combinados (modelo + entrada): %s", combined)
        labels = combined

    for label in labels:
        if label in PATTERN_ADDERS:
            if label == "PASSAPORTE" and foreign_passport:
                logger.debug("Adicionando padrões de PASSAPORTE (estrangeiro)")
                add_pattern_passaporte_est(patterns, logger)
            before = len(patterns)
            PATTERN_ADDERS[label](patterns, logger)
            after = len(patterns)
            logger.debug("Padrões adicionados para '%s': novos=%d (total=%d)", label, after - before, after)
        elif label == "FISTEL" and ("CPF" not in labels):
            logger.debug("Label 'FISTEL' ativa CPF por compatibilidade")
            before = len(patterns)
            PATTERN_ADDERS["CPF"](patterns, logger)
            after = len(patterns)
            logger.debug("Padrões adicionados para 'CPF' via 'FISTEL': novos=%d (total=%d)", after - before, after)
        elif label == "FISTEL" and ("CPF" in labels):
            logger.debug("Label 'FISTEL' ignorada (CPF já presente)")
        else:
            supported = [*list(PATTERN_ADDERS.keys()), "FISTEL"]
            msg = f"Rótulo não suportado: {label}. Suportados: {supported}"
            logger.error(msg)
            raise ValueError(msg)

    if custom_patterns:
        for custom in custom_patterns:
            if not custom.get("label") or not custom.get("regex"):
                msg = f"Padrão customizado inválido: {custom}"
                logger.error(msg)
                raise ValueError(msg)
            add_custom_pattern(
                patterns,
                label=custom.get("label"),
                regex_pattern=custom.get("regex"),
                description=custom.get("description", ""),
                logger=logger,
            )
            logger.debug("Padrão customizado aplicado: %s", custom.get("label"))

    summary = get_active_labels(patterns, model_labels)
    logger.debug(
        "Total de padrões ativos: %d | regex_patterns=%d | resumo=%s",
        summary["total_patterns"],
        summary["regex_patterns"],
        summary["regex_summary"],
    )

    return summary


def list_patterns(patterns: list[dict], model_labels: list[str]) -> dict:
    """Retorna todos os padrões regex registrados, organizados por categoria.

    Separa padrões pré-definidos (built-in) de padrões customizados.

    Args:
        patterns (list[dict]): Lista de padrões registrados
        model_labels (list[str]): Labels ativos no modelo (para identificar built-in)

    Returns:
        dict: Dicionário com duas chaves:
            - "builtin": Lista de padrões pré-definidos do sistema
            - "custom": Lista de padrões customizados adicionados pelo usuário
    """
    builtin = [p for p in patterns if p["label"] in model_labels]
    custom = [p for p in patterns if p["label"] not in model_labels]
    return {"builtin": builtin, "custom": custom}
