"""Módulo para cálculo de métricas de avaliação NER.

Este módulo contém funções para calcular métricas de avaliação como
precision, recall, F-beta score, considerando overlap entre entidades.
"""

import logging
from collections.abc import Callable

import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score

from anonimizar._constants import ALL_ENTITIES_KEY, DEFAULT_BETA_VALUES, DEFAULT_OVERLAP_THRESHOLDS


def calculate_entity_metrics(
    y_true: list,
    y_pred: list,
    beta: float,
) -> dict:
    """Calcula métricas para uma lista de labels verdadeiros e preditos.

    Args:
        y_true: Lista de labels verdadeiros (0 ou 1)
        y_pred: Lista de labels preditos (0 ou 1)
        beta: Valor beta para F-beta score

    Returns:
        dict: Dicionário com métricas (fbeta, precision, recall, tp, fp, fn)
    """
    fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    tp = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == 1 and yp == 0)

    return {"fbeta": fbeta, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def generate_evaluation_report(
    df_ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    comparison_data: pd.DataFrame,
    overlap_threshold: float,
    beta: float,
    entity_types: list[str],
    logger: logging.Logger,
    generate_comparison_fn: Callable[..., pd.DataFrame],
) -> dict:
    """Gera relatório de avaliação com métricas detalhadas por tipo de entidade.

    Args:
        df_ground_truth: DataFrame com ground truth
        predictions: DataFrame com predições
        comparison_data: DataFrame com comparação já calculada para todas entidades
        overlap_threshold: Threshold de overlap para considerar match
        beta: Valor beta para F-beta score
        entity_types: Lista de tipos de entidade para avaliar
        logger: Logger para mensagens
        generate_comparison_fn: Função para gerar dados de comparação

    Returns:
        dict: Dicionário com métricas por tipo de entidade
    """
    results = {}

    # Calcular métricas por tipo de entidade
    for entity_type in entity_types:
        try:
            gt_subset = df_ground_truth[df_ground_truth["tp_entidade"] == entity_type]
            pred_subset = predictions[predictions["tp_entidade"] == entity_type]

            if len(gt_subset) == 0:
                logger.warning("Nenhuma entidade %s no ground truth", entity_type)
                continue

            # Gerar comparação para este tipo de entidade
            comparison = generate_comparison_fn(pred_subset, overlap_threshold)
            entity_comparison = comparison[comparison["tp_entidade"] == entity_type]

            if len(entity_comparison) == 0:
                results[entity_type] = {
                    "qtd_ids": len(gt_subset["id"].unique()),
                    "qtd_entidades": len(gt_subset),
                    "fbeta": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "tp": 0,
                    "fp": 0,
                    "fn": len(gt_subset),
                }
                continue

            y_true = entity_comparison["y_true"].tolist()
            y_pred = entity_comparison["y_pred"].tolist()

            metrics = calculate_entity_metrics(y_true, y_pred, beta)

            results[entity_type] = {
                "qtd_ids": len(gt_subset["id"].unique()),
                "qtd_entidades": len(gt_subset),
                **metrics,
            }

        except Exception as e:
            logger.exception("Erro ao processar entidade %s: %s", entity_type, e)  # noqa: TRY401
            raise

    # Calcular métricas consolidadas (todas as entidades)
    try:
        all_comparison = comparison_data

        if len(all_comparison) > 0:
            y_true_all = all_comparison["y_true"].tolist()
            y_pred_all = all_comparison["y_pred"].tolist()

            metrics_all = calculate_entity_metrics(y_true_all, y_pred_all, beta)

            results[ALL_ENTITIES_KEY] = {
                "qtd_ids": len(df_ground_truth["id"].unique()),
                "qtd_entidades": len(df_ground_truth),
                **metrics_all,
            }
        else:
            logger.warning("Sem comparação consolidada (all_comparison vazio)")

    except Exception as e:
        logger.exception("Erro ao calcular métricas consolidadas: %s", e)  # noqa: TRY401
        raise

    return results


def format_summary_report(evaluation_results: dict) -> str:
    """Formata relatório resumido das métricas calculadas.

    Args:
        evaluation_results: Dicionário com resultados da avaliação

    Returns:
        str: Relatório formatado
    """
    if not evaluation_results:
        return "Nenhuma avaliação executada ainda."

    lines = []
    lines.append("=" * 60)
    lines.append("RELATÓRIO DE AVALIAÇÃO")
    lines.append("=" * 60)

    for entity_type, metrics in evaluation_results.items():
        lines.append(f"\n{entity_type}:")
        lines.append(f"  Quantidade de IDs: {metrics.get('qtd_ids', 0)}")
        lines.append(f"  Quantidade de entidades: {metrics.get('qtd_entidades', 0)}")
        lines.append(f"  F-beta: {metrics.get('fbeta', 0):.4f}")
        lines.append(f"  Precision: {metrics.get('precision', 0):.4f}")
        lines.append(f"  Recall: {metrics.get('recall', 0):.4f}")
        lines.append(f"  TP: {metrics.get('tp', 0)}")
        lines.append(f"  FP: {metrics.get('fp', 0)}")
        lines.append(f"  FN: {metrics.get('fn', 0)}")

    lines.append("=" * 60)

    return "\n".join(lines)


def get_detailed_report(evaluation_results: dict) -> pd.DataFrame:
    """Converte resultados de avaliação para DataFrame detalhado.

    Args:
        evaluation_results: Dicionário com resultados da avaliação

    Returns:
        DataFrame com uma linha por tipo de entidade
    """
    if not evaluation_results:
        return pd.DataFrame()

    rows = []
    for entity_type, metrics in evaluation_results.items():
        row = {"tp_entidade": entity_type, **metrics}
        rows.append(row)

    return pd.DataFrame(rows)


def calculate_overlap(start_true: int, end_true: int, start_pred: int, end_pred: int) -> float:
    """Calcula a proporção de sobreposição entre entidade verdadeira e predita.

    A sobreposição é calculada como a proporção da interseção sobre o comprimento
    da entidade verdadeira (ground truth).

    Args:
        start_true: Posição inicial da entidade verdadeira.
        end_true: Posição final da entidade verdadeira.
        start_pred: Posição inicial da entidade predita.
        end_pred: Posição final da entidade predita.

    Returns:
        Taxa de overlap entre 0.0 e 1.0.

    Raises:
        ValueError: Se start > end em qualquer dos spans.

    Examples:
        >>> calculate_overlap(0, 10, 5, 15)
        0.5
        >>> calculate_overlap(0, 10, 0, 10)
        1.0
        >>> calculate_overlap(0, 10, 10, 20)
        0.0
    """
    if start_pred > end_pred or start_true > end_true:
        msg = "A entidade não pode iniciar depois do término da mesma."
        raise ValueError(msg)

    overlap = max(0, min(end_true, end_pred) - max(start_true, start_pred))
    true_length = end_true - start_true
    return overlap / true_length if true_length > 0 else 0.0


def evaluate_multiple_thresholds(
    predictions: pd.DataFrame,
    df_ground_truth: pd.DataFrame,
    generate_comparison_fn: object,
    overlap_thresholds: list[float] | None = None,
    beta_values: list[float] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Avalia modelo com múltiplos thresholds e valores de beta.

    Args:
        predictions: DataFrame com predições do modelo.
        df_ground_truth: DataFrame com ground truth.
        generate_comparison_fn: Função para gerar dados de comparação.
        overlap_thresholds: Lista de thresholds a testar.
            Se None, usa DEFAULT_OVERLAP_THRESHOLDS.
        beta_values: Lista de valores beta.
            Se None, usa DEFAULT_BETA_VALUES.
        logger: Logger para mensagens. Se None, cria um padrão.

    Returns:
        DataFrame com resultados para cada combinação threshold/beta,
        contendo colunas: tp_entidade, qtd_ids, qtd_entidades, fbeta,
        precision, recall, tp, fp, fn, overlap_threshold, beta.

    Examples:
        >>> df = evaluate_multiple_thresholds(
        ...     predictions=preds_df,
        ...     df_ground_truth=gt_df,
        ...     generate_comparison_fn=gen_fn,
        ...     overlap_thresholds=[0.5, 0.8],
        ...     beta_values=[1.0, 2.0],
        ... )
        >>> df.shape[1]
        11
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    thresholds_to_use = list(overlap_thresholds) if overlap_thresholds is not None else list(DEFAULT_OVERLAP_THRESHOLDS)
    betas_to_use = list(beta_values) if beta_values is not None else list(DEFAULT_BETA_VALUES)

    entity_types = df_ground_truth["tp_entidade"].unique().tolist()

    all_results = []

    for overlap_threshold in thresholds_to_use:
        for beta in betas_to_use:
            logger.debug("Avaliando: overlap=%.3f, beta=%.3f", overlap_threshold, beta)

            try:
                comparison_data = generate_comparison_fn(predictions, overlap_threshold)  # type: ignore[operator]

                results = generate_evaluation_report(
                    df_ground_truth=df_ground_truth,
                    predictions=predictions,
                    comparison_data=comparison_data,
                    overlap_threshold=overlap_threshold,
                    beta=beta,
                    entity_types=entity_types,
                    logger=logger,
                    generate_comparison_fn=generate_comparison_fn,  # type: ignore[arg-type]
                )

                for entity_type, entity_metrics in results.items():
                    all_results.append(
                        {
                            "tp_entidade": entity_type,
                            "qtd_ids": entity_metrics["qtd_ids"],
                            "qtd_entidades": entity_metrics["qtd_entidades"],
                            "fbeta": entity_metrics["fbeta"],
                            "precision": entity_metrics["precision"],
                            "recall": entity_metrics["recall"],
                            "tp": entity_metrics["tp"],
                            "fp": entity_metrics["fp"],
                            "fn": entity_metrics["fn"],
                            "overlap_threshold": overlap_threshold,
                            "beta": beta,
                        }
                    )

            except Exception as e:
                logger.exception("Erro na avaliação overlap=%.3f, beta=%.3f: %s", overlap_threshold, beta, e)  # noqa: TRY401
                continue

    results_df = pd.DataFrame(all_results)
    logger.debug("Avaliação múltipla concluída: %d registros", len(results_df))

    return results_df
