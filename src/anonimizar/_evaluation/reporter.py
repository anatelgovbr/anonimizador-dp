"""Módulo para geração de relatórios e análises de erros NER.

Este módulo contém funções para gerar relatórios detalhados, análise de erros,
e exportação de casos de classificação (TP, FP, FN, TN).
"""

import logging
from pathlib import Path

import pandas as pd

from anonimizar._constants import (
    ALL_ENTITIES_KEY,
    MAX_ERROR_EXAMPLES,
    OVERLAP_BIN_LABELS,
    OVERLAP_BINS,
)


def classify_case(y_true: int, y_pred: int) -> str:
    """Classifica um caso como TP, FP, FN ou TN.

    Args:
        y_true: Label verdadeiro (0 ou 1)
        y_pred: Label predito (0 ou 1)

    Returns:
        str: Classificação ('TP', 'FP', 'FN', 'TN')
    """
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 1 and y_pred == 0:
        return "FN"
    return "TN"


def get_classification_cases(
    comparison_data: pd.DataFrame,
    entity_type: str | None = None,
    case_type: str = "all",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Extrai casos individuais de TP, FP, TN, FN para análise detalhada.

    Args:
        comparison_data: DataFrame com dados de comparação
        entity_type: Tipo específico de entidade. Se None, considera todas
        case_type: Tipo de caso ('tp', 'fp', 'tn', 'fn', 'all')
        logger: Logger para mensagens

    Returns:
        DataFrame com casos individuais e suas classificações

    Raises:
        ValueError: Se case_type inválido
    """
    valid_case_types = {"tp", "fp", "tn", "fn", "all"}
    if case_type not in valid_case_types:
        msg = f"case_type deve ser um de: {valid_case_types}"
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    # Filtrar por tipo de entidade se especificado
    if entity_type is not None:
        comparison_data = comparison_data[comparison_data["tp_entidade"] == entity_type]

    # Classificar cada caso
    comparison_data = comparison_data.copy()
    comparison_data["classification"] = comparison_data.apply(
        lambda row: classify_case(row["y_true"], row["y_pred"]), axis=1
    )

    # Filtrar por tipo de caso se não for 'all'
    if case_type != "all":
        comparison_data = comparison_data[comparison_data["classification"] == case_type.upper()]

    # Adicionar flags de texto presente
    comparison_data["has_gt_text"] = pd.notna(comparison_data.get("text_entidade_true"))
    comparison_data["has_pred_text"] = pd.notna(comparison_data.get("text_entidade_pred"))

    # Ordenar por ID e tipo de entidade
    comparison_data = comparison_data.sort_values(["id", "tp_entidade"])

    if logger:
        logger.debug(
            "Extraídos %d casos do tipo '%s' para entidade '%s'",
            len(comparison_data),
            case_type,
            entity_type or ALL_ENTITIES_KEY,
        )

    return comparison_data.reset_index(drop=True)


def get_error_analysis(
    comparison_data: pd.DataFrame,
    entity_type: str | None = None,
    logger: logging.Logger | None = None,
) -> dict:
    """Gera análise detalhada de erros por tipo de entidade.

    Args:
        comparison_data: DataFrame com dados de comparação
        entity_type: Tipo específico de entidade. Se None, analisa todas
        logger: Logger para mensagens

    Returns:
        dict: Dicionário com estatísticas e exemplos de erros
    """
    # Obter todos os casos classificados
    all_cases = get_classification_cases(comparison_data, entity_type=entity_type, case_type="all", logger=logger)

    analysis = {
        "entity_type": entity_type or ALL_ENTITIES_KEY,
        "total_cases": len(all_cases),
        "summary": {},
        "tp_examples": [],
        "fp_examples": [],
        "fn_examples": [],
        "overlap_distribution": {},
    }

    # Contar casos por classificação
    classification_counts = all_cases["classification"].value_counts().to_dict()
    analysis["summary"] = {
        "TP": classification_counts.get("TP", 0),
        "FP": classification_counts.get("FP", 0),
        "FN": classification_counts.get("FN", 0),
        "TN": classification_counts.get("TN", 0),
    }

    # Extrair exemplos para cada tipo de caso
    for case_type in ["TP", "FP", "FN"]:
        cases = all_cases[all_cases["classification"] == case_type]

        examples = []
        for _, row in cases.head(MAX_ERROR_EXAMPLES).iterrows():
            example = {"id": row["id"], "tp_entidade": row["tp_entidade"], "overlap": row.get("overlap", 0.0)}

            # Adicionar informações de ground truth se presente
            if pd.notna(row.get("text_entidade_true")):
                example["texto_gt"] = row["text_entidade_true"]
                example["pos_gt"] = f"{row.get('start_entidade_true', 'N/A')}-{row.get('end_entidade_true', 'N/A')}"

            # Adicionar informações de predição se presente
            if pd.notna(row.get("text_entidade_pred")):
                example["texto_pred"] = row["text_entidade_pred"]
                example["pos_pred"] = f"{row.get('start_entidade_pred', 'N/A')}-{row.get('end_entidade_pred', 'N/A')}"

            examples.append(example)

        analysis[f"{case_type.lower()}_examples"] = examples

    # Calcular distribuição de overlap
    overlap_bins = OVERLAP_BINS
    overlap_labels = OVERLAP_BIN_LABELS

    all_cases["overlap_bin"] = pd.cut(
        all_cases["overlap"], bins=overlap_bins, labels=overlap_labels, include_lowest=True
    )

    analysis["overlap_distribution"] = all_cases["overlap_bin"].value_counts().to_dict()

    if logger:
        logger.debug("Análise de erros concluída para %s", analysis["entity_type"])

    return analysis


def save_classification_cases(
    comparison_data: pd.DataFrame,
    output_path: str | Path,
    entity_type: str | None = None,
    case_type: str = "all",
    format: str = "parquet",  # noqa: A002
    logger: logging.Logger | None = None,
) -> None:
    """Salva casos de classificação em arquivo para análise externa.

    Args:
        comparison_data: DataFrame com dados de comparação
        output_path: Caminho para salvar o arquivo
        entity_type: Tipo de entidade. Se None, salva todas
        case_type: Tipo de caso ('tp', 'fp', 'fn', 'tn', 'all')
        format: Formato do arquivo: ``'parquet'``, ``'csv'`` ou ``'json'``.
        logger: Logger para mensagens

    Raises:
        ValueError: Se formato não é suportado
    """
    # Obter casos classificados
    cases = get_classification_cases(comparison_data, entity_type=entity_type, case_type=case_type, logger=logger)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info("Salvando casos de classificação em %s (formato=%s) - %d linhas", output_path, format, len(cases))

    # Salvar arquivo no formato especificado
    if format == "parquet":
        cases.to_parquet(output_path, index=False)
    elif format == "csv":
        cases.to_csv(output_path, index=False)
    elif format == "json":
        detected_by_col = "detected_by"
        if detected_by_col in cases.columns:
            cases = cases.copy()
            cases[detected_by_col] = cases[detected_by_col].astype("str")
        cases.to_json(str(output_path), orient="records", indent=2, force_ascii=False)
    else:
        msg = f"Formato não suportado: {format}. Use 'parquet', 'csv' ou 'json'"
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    if logger:
        logger.info("Arquivo salvo com sucesso: %s", output_path)


def export_dataframe(
    df: pd.DataFrame,
    output_path: str | Path,
    format: str = "parquet",  # noqa: A002
    logger: logging.Logger | None = None,
) -> None:
    """Exporta um DataFrame para arquivo no formato especificado.

    Args:
        df: DataFrame a ser exportado.
        output_path: Caminho para salvar o arquivo.
        format: Formato do arquivo ('parquet', 'csv', 'json'). Padrão 'parquet'.
        logger: Logger para mensagens.

    Raises:
        ValueError: Se formato não é suportado.

    Examples:
        >>> report = pd.DataFrame({"tp_entidade": ["CPF"], "fbeta": [1.0]})
        >>> export_dataframe(report, "results.parquet")
    """
    output_path = Path(output_path)

    if logger:
        logger.info("Exportando DataFrame para %s (formato=%s)", output_path, format)

    if format == "parquet":
        df.to_parquet(output_path, index=False)
    elif format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "json":
        df.to_json(str(output_path), orient="records", indent=2, force_ascii=False)
    else:
        msg = f"Formato não suportado: {format}. Use 'parquet', 'csv' ou 'json'."
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    if logger:
        logger.debug("DataFrame exportado para: %s", output_path)
