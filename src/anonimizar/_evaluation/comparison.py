"""Módulo para comparação entre ground truth e predições NER.

Este módulo contém funções para comparar entidades ground truth com predições,
calculando overlap e gerando dados de comparação para avaliação de métricas.
"""

import logging

import numpy as np
import pandas as pd

from anonimizar._constants import (
    COMPARISON_SUFFIXES,
    MERGE_SUFFIXES,
    PERCENT_MULTIPLIER,
    REQUIRED_PREDICTION_COLUMNS,
    REQUIRED_REPORT_COLUMNS,
)


def calculate_overlap(start_true: int, end_true: int, start_pred: int, end_pred: int) -> float:
    """Calcula overlap normalizado entre duas entidades.

    O overlap é calculado como a razão entre a interseção e a união
    dos spans das duas entidades.

    Args:
        start_true: Posição inicial da entidade verdadeira
        end_true: Posição final da entidade verdadeira
        start_pred: Posição inicial da entidade predita
        end_pred: Posição final da entidade predita

    Returns:
        float: Valor de overlap entre 0.0 e 1.0
    """
    if pd.isna(start_pred) or pd.isna(end_pred):
        return 0.0

    # Calcular interseção
    start_inter = max(start_true, start_pred)
    end_inter = min(end_true, end_pred)

    if start_inter >= end_inter:
        return 0.0

    intersection = end_inter - start_inter

    # Calcular união
    start_union = min(start_true, start_pred)
    end_union = max(end_true, end_pred)
    union = end_union - start_union

    if union == 0:
        return 0.0

    return intersection / union


def generate_comparison_data(  # noqa: PLR0915
    df_ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    entity_mapping: dict,
    overlap_threshold: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Gera dados de comparação entre ground truth e predições.

    Compara cada entidade do ground truth com as predições correspondentes,
    calculando overlap e determinando true positives, false positives e
    false negatives.

    Args:
        df_ground_truth: DataFrame com ground truth
        predictions: DataFrame com predições
        entity_mapping: Dicionário para mapear labels
        overlap_threshold: Threshold de overlap para considerar match
        logger: Logger para mensagens

    Returns:
        DataFrame com colunas de comparação incluindo y_true, y_pred, overlap

    Raises:
        ValueError: Se colunas necessárias estão ausentes
    """
    if df_ground_truth is None or len(df_ground_truth) == 0:
        msg = "Dados de ground truth não carregados"
        logger.error(msg)
        raise ValueError(msg)

    # Validar colunas necessárias
    required_cols = REQUIRED_PREDICTION_COLUMNS
    missing_cols = set(required_cols) - set(predictions.columns)
    if missing_cols:
        msg = f"Colunas ausentes nas predições: {missing_cols}"
        logger.error(msg)
        raise ValueError(msg)

    # Normalizar labels das predições
    predictions_norm = predictions.copy()
    predictions_norm["tp_entidade"] = predictions_norm["tp_entidade"].apply(lambda x: entity_mapping.get(x, x))

    logger.debug("Gerando comparação com %d ground truth e %d predições", len(df_ground_truth), len(predictions_norm))

    # Merge left para ter todas as entidades do ground truth
    merged = df_ground_truth.merge(predictions_norm, how="left", on=["id", "tp_entidade"], suffixes=MERGE_SUFFIXES)

    # Calcular overlap para cada par
    merged["overlap"] = merged.apply(
        lambda row: calculate_overlap(
            row["start_entidade_true"],
            row["end_entidade_true"],
            row["start_entidade_pred"] if pd.notna(row["start_entidade_pred"]) else 0,
            row["end_entidade_pred"] if pd.notna(row["end_entidade_pred"]) else 0,
        )
        if pd.notna(row["start_entidade_pred"])
        else 0.0,
        axis=1,
    )

    # Ordenar por overlap (maior primeiro) e remover duplicatas
    merged = merged.sort_values(by=["id", "tp_entidade", "overlap"], ascending=[True, True, False])
    merged = merged.drop_duplicates(
        subset=["id", "tp_entidade", "start_entidade_pred", "end_entidade_pred"], keep="first"
    )

    # Pegar melhor match para cada ground truth
    matched = merged.groupby(["id", "tp_entidade", "start_entidade_true", "end_entidade_true"]).first().reset_index()
    matched = matched.dropna(subset=["start_entidade_pred"])

    # Determinar y_true e y_pred
    matched["y_true"] = 1
    matched["y_pred"] = (matched["overlap"] >= overlap_threshold).astype(int)

    # Identificar entidades matched
    matched_true_ids = matched[["id", "tp_entidade", "start_entidade_true", "end_entidade_true"]]
    matched_pred_ids = matched[["id", "tp_entidade", "start_entidade_pred", "end_entidade_pred"]]

    # Encontrar ground truth não matched (FN)
    unmatched_ground_truth = (
        df_ground_truth.merge(
            matched_true_ids.rename(
                columns={"start_entidade_true": "start_entidade", "end_entidade_true": "end_entidade"}
            ),
            on=["id", "tp_entidade", "start_entidade", "end_entidade"],
            how="left",
            indicator=True,
        )
        .query('_merge == "left_only"')
        .drop(columns=["_merge"])
    )

    # Encontrar predições não matched (FP)
    unmatched_predictions = (
        predictions_norm.merge(
            matched_pred_ids.rename(
                columns={"start_entidade_pred": "start_entidade", "end_entidade_pred": "end_entidade"}
            ),
            on=["id", "tp_entidade", "start_entidade", "end_entidade"],
            how="left",
            indicator=True,
        )
        .query('_merge == "left_only"')
        .drop(columns=["_merge"])
    )

    # Renomear colunas para manter consistência
    unmatched_ground_truth = unmatched_ground_truth.rename(
        columns={
            "start_entidade": "start_entidade_true",
            "end_entidade": "end_entidade_true",
            "text_entidade": "text_entidade_true",
        }
    )

    unmatched_predictions = unmatched_predictions.rename(
        columns={
            "start_entidade": "start_entidade_pred",
            "end_entidade": "end_entidade_pred",
            "text_entidade": "text_entidade_pred",
        }
    )

    # Atribuir labels para FN e FP
    unmatched_ground_truth["y_true"] = 1
    unmatched_ground_truth["y_pred"] = 0
    unmatched_ground_truth["overlap"] = 0.0

    unmatched_predictions["y_true"] = 0
    unmatched_predictions["y_pred"] = 1
    unmatched_predictions["overlap"] = 0.0

    # Concatenar todos os casos
    final_comparison = pd.concat([matched, unmatched_ground_truth, unmatched_predictions], ignore_index=True)

    # Ajustar casos problemáticos (FN com span pred presente)
    problematic_cases = final_comparison[
        (final_comparison["y_true"] == 1)
        & (final_comparison["y_pred"] == 0)
        & (pd.notna(final_comparison["start_entidade_true"]))
        & (pd.notna(final_comparison["start_entidade_pred"]))
    ]

    if len(problematic_cases) > 0:
        logger.debug("Ajustando %d casos problemáticos (FN/FP com spans presentes)", len(problematic_cases))
        ok_cases = final_comparison.loc[~final_comparison.index.isin(problematic_cases.index)]

        # Criar FN sem predição
        fn_cases = problematic_cases.copy()
        fn_cases["start_entidade_pred"] = None
        fn_cases["end_entidade_pred"] = None
        fn_cases["text_entidade_pred"] = None

        # Criar FP sem ground truth
        fp_cases = problematic_cases.copy()
        fp_cases["start_entidade_true"] = None
        fp_cases["end_entidade_true"] = None
        fp_cases["text_entidade_true"] = None
        fp_cases["y_true"] = 0
        fp_cases["y_pred"] = 1

        final_comparison = pd.concat([ok_cases, fn_cases, fp_cases], ignore_index=True)

    logger.debug("Comparação gerada: %d registros", len(final_comparison))
    return final_comparison.reset_index(drop=True)


def compare_reports(
    current_report: pd.DataFrame,
    previous_report: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Compara dois relatórios de métricas e identifica melhorias/pioras.

    Args:
        current_report: Relatório atual com métricas por entidade.
        previous_report: Relatório anterior para comparação.
        logger: Logger para mensagens.

    Returns:
        DataFrame com colunas de diferença, boolean de melhoria e percentual
        de mudança para fbeta, precision e recall.

    Raises:
        ValueError: Se os DataFrames não possuem colunas necessárias.

    Examples:
        >>> df_comp = compare_reports(current_df, previous_df, logger)
        >>> df_comp["fbeta_melhorou"].sum()
        3
    """
    required_cols = REQUIRED_REPORT_COLUMNS

    for df_name, df in [("current_report", current_report), ("previous_report", previous_report)]:
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            msg = f"Colunas ausentes em {df_name}: {missing_cols}"
            logger.error(msg)
            raise ValueError(msg)

    current_report = current_report.copy()
    previous_report = previous_report.copy()

    merge_cols = ["tp_entidade"]
    if "overlap_threshold" in current_report.columns and "beta" in current_report.columns:
        merge_cols.extend(["overlap_threshold", "beta"])

    result = current_report.merge(previous_report, on=merge_cols, suffixes=COMPARISON_SUFFIXES)

    if result.empty:
        logger.warning("Nenhuma interseção entre relatórios para comparar.")
        return result

    result["fbeta_diff"] = result["fbeta_atual"] - result["fbeta_anterior"]
    result["precision_diff"] = result["precision_atual"] - result["precision_anterior"]
    result["recall_diff"] = result["recall_atual"] - result["recall_anterior"]

    result["fbeta_melhorou"] = result["fbeta_diff"] > 0
    result["precision_melhorou"] = result["precision_diff"] > 0
    result["recall_melhorou"] = result["recall_diff"] > 0

    result["fbeta_perc_change"] = (
        (result["fbeta_atual"] - result["fbeta_anterior"])
        / result["fbeta_anterior"].replace(0, np.nan)
        * PERCENT_MULTIPLIER
    ).fillna(0)

    result["precision_perc_change"] = (
        (result["precision_atual"] - result["precision_anterior"])
        / result["precision_anterior"].replace(0, np.nan)
        * PERCENT_MULTIPLIER
    ).fillna(0)

    result["recall_perc_change"] = (
        (result["recall_atual"] - result["recall_anterior"])
        / result["recall_anterior"].replace(0, np.nan)
        * PERCENT_MULTIPLIER
    ).fillna(0)

    result["status_geral"] = result.apply(
        lambda row: "Melhorou" if row["fbeta_melhorou"] else ("Piorou" if row["fbeta_diff"] < 0 else "Estável"),
        axis=1,
    )

    logger.debug("Comparação de relatórios concluída: %d entidades comparadas", len(result))

    return result


def classify_cases(comparison_data: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna 'classification' ao DataFrame de comparação.

    Classifica cada linha como TP, FP, FN ou TN com base em y_true e y_pred.

    Args:
        comparison_data: DataFrame com colunas y_true e y_pred.

    Returns:
        Cópia do DataFrame com coluna adicional 'classification'.

    Examples:
        >>> df = classify_cases(comparison_df)
        >>> df["classification"].value_counts()
        TP    10
        FP     2
        FN     1
        TN     0
        Name: classification, dtype: int64
    """

    def _classify(row: pd.Series) -> str:
        if row["y_true"] == 1 and row["y_pred"] == 1:
            return "TP"
        if row["y_true"] == 0 and row["y_pred"] == 1:
            return "FP"
        if row["y_true"] == 1 and row["y_pred"] == 0:
            return "FN"
        return "TN"

    result = comparison_data.copy()
    result["classification"] = result.apply(_classify, axis=1)
    return result


def get_classification_cases(
    comparison_data: pd.DataFrame,
    entity_type: str | None,
    case_type: str,
    all_entities_key: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Extrai casos individuais de TP, FP, TN, FN para análise detalhada.

    Args:
        comparison_data: DataFrame de comparação pré-calculado.
        entity_type: Tipo específico de entidade a filtrar. None = todas.
        case_type: Tipo de caso ('tp', 'fp', 'tn', 'fn', 'all').
        all_entities_key: Label para todas as entidades.
        logger: Logger para mensagens.

    Returns:
        DataFrame com casos individuais classificados e ordenados.

    Raises:
        ValueError: Se case_type for inválido.

    Examples:
        >>> cases = get_classification_cases(
        ...     comparison_data=comp_df,
        ...     entity_type="CPF",
        ...     case_type="fp",
        ...     all_entities_key="TODAS",
        ...     logger=logger,
        ... )
    """
    valid_case_types = {"tp", "fp", "tn", "fn", "all"}
    if case_type not in valid_case_types:
        msg = f"case_type deve ser um de: {valid_case_types}"
        logger.error(msg)
        raise ValueError(msg)

    classified = classify_cases(comparison_data)

    if entity_type is not None:
        classified = classified[classified["tp_entidade"] == entity_type]

    if case_type != "all":
        classified = classified[classified["classification"] == case_type.upper()]

    classified["has_gt_text"] = pd.notna(classified.get("text_entidade_true"))
    classified["has_pred_text"] = pd.notna(classified.get("text_entidade_pred"))

    classified = classified.sort_values(["id", "tp_entidade"])

    logger.debug(
        "Extraídos %d casos do tipo '%s' para entidade '%s'",
        len(classified),
        case_type,
        entity_type or all_entities_key,
    )

    return classified.reset_index(drop=True)


def get_error_analysis(
    comparison_data: pd.DataFrame,
    entity_type: str | None,
    max_examples: int,
    overlap_bins: list,
    overlap_bin_labels: list,
    all_entities_key: str,
    logger: logging.Logger,
) -> dict:
    """Gera análise detalhada de erros por tipo de entidade.

    Args:
        comparison_data: DataFrame de comparação já classificado (com coluna 'classification').
        entity_type: Tipo de entidade específico. None = todas.
        max_examples: Número máximo de exemplos por categoria.
        overlap_bins: Limites dos bins para distribuição de overlap.
        overlap_bin_labels: Labels dos bins de overlap.
        all_entities_key: Label para todas as entidades.
        logger: Logger para mensagens.

    Returns:
        Dicionário com:
        - entity_type: tipo de entidade analisado
        - total_cases: contagem total
        - summary: contagem TP/FP/FN/TN
        - tp_examples: lista de exemplos TP
        - fp_examples: lista de exemplos FP
        - fn_examples: lista de exemplos FN
        - overlap_distribution: distribuição por bin de overlap

    Examples:
        >>> analysis = get_error_analysis(
        ...     comparison_data=comp_df,
        ...     entity_type=None,
        ...     max_examples=10,
        ...     overlap_bins=[0, 0.5, 1.0],
        ...     overlap_bin_labels=["baixo", "alto"],
        ...     all_entities_key="TODAS",
        ...     logger=logger,
        ... )
        >>> analysis["summary"]["TP"]
        10
    """
    all_cases = classify_cases(comparison_data)

    if entity_type is not None:
        all_cases = all_cases[all_cases["tp_entidade"] == entity_type]

    analysis: dict = {
        "entity_type": entity_type or all_entities_key,
        "total_cases": len(all_cases),
        "summary": {},
        "tp_examples": [],
        "fp_examples": [],
        "fn_examples": [],
        "overlap_distribution": {},
    }

    classification_counts = all_cases["classification"].value_counts().to_dict()
    analysis["summary"] = {
        "TP": classification_counts.get("TP", 0),
        "FP": classification_counts.get("FP", 0),
        "FN": classification_counts.get("FN", 0),
        "TN": classification_counts.get("TN", 0),
    }

    for case_type in ["TP", "FP", "FN"]:
        cases = all_cases[all_cases["classification"] == case_type]

        examples = []
        for _, row in cases.head(max_examples).iterrows():
            example = {"id": row["id"], "tp_entidade": row["tp_entidade"], "overlap": row.get("overlap", 0.0)}

            if pd.notna(row.get("text_entidade_true")):
                example["texto_gt"] = row["text_entidade_true"]
                example["pos_gt"] = f"{row.get('start_entidade_true', 'N/A')}-{row.get('end_entidade_true', 'N/A')}"

            if pd.notna(row.get("text_entidade_pred")):
                example["texto_pred"] = row["text_entidade_pred"]
                example["pos_pred"] = f"{row.get('start_entidade_pred', 'N/A')}-{row.get('end_entidade_pred', 'N/A')}"

            examples.append(example)

        analysis[f"{case_type.lower()}_examples"] = examples

    all_cases["overlap_bin"] = pd.cut(
        all_cases["overlap"], bins=overlap_bins, labels=overlap_bin_labels, include_lowest=True
    )

    analysis["overlap_distribution"] = all_cases["overlap_bin"].value_counts().to_dict()

    logger.debug("Análise de erros concluída para %s", analysis["entity_type"])

    return analysis
