"""Módulo para carregamento de dados de avaliação NER.

Este módulo contém funções para carregar ground truth e textos de diferentes
formatos (DataFrame, arquivos, etc.) para avaliação de modelos NER.
"""

import logging
from pathlib import Path

import pandas as pd


def load_data(
    df_texts: pd.DataFrame,
    df_ground_truth: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega e valida dados de textos e ground truth.

    Args:
        df_texts: DataFrame com colunas ['id', 'text']
        df_ground_truth: DataFrame com colunas ['id', 'tp_entidade',
                         'start_entidade', 'end_entidade']
        logger: Logger para mensagens

    Returns:
        tuple: (df_texts, df_ground_truth) validados

    Raises:
        ValueError: Se colunas necessárias estão ausentes
    """
    # Validar df_texts
    required_text_cols = {"id", "text"}
    missing_text_cols = required_text_cols - set(df_texts.columns)
    if missing_text_cols:
        msg = f"Colunas ausentes em df_texts: {missing_text_cols}"
        logger.error(msg)
        raise ValueError(msg)

    # Validar df_ground_truth
    required_gt_cols = {"id", "tp_entidade", "start_entidade", "end_entidade"}
    missing_gt_cols = required_gt_cols - set(df_ground_truth.columns)
    if missing_gt_cols:
        msg = f"Colunas ausentes em df_ground_truth: {missing_gt_cols}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Dados carregados: %d textos, %d entidades ground truth", len(df_texts), len(df_ground_truth))

    return df_texts, df_ground_truth


def load_data_from_files(
    texts_path: str | Path,
    ground_truth_path: str | Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega dados de arquivos Parquet ou CSV.

    Args:
        texts_path: Caminho para arquivo de textos
        ground_truth_path: Caminho para arquivo de ground truth
        logger: Logger para mensagens

    Returns:
        tuple: (df_texts, df_ground_truth)

    Raises:
        FileNotFoundError: Se arquivo não existe
        ValueError: Se formato não é suportado
    """
    texts_path = Path(texts_path)
    gt_path = Path(ground_truth_path)

    # Verificar existência
    if not texts_path.exists():
        msg = f"Arquivo de textos não encontrado: {texts_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    if not gt_path.exists():
        msg = f"Arquivo de ground truth não encontrado: {gt_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Carregar textos
    text_suffix = texts_path.suffix.lower()
    if text_suffix == ".parquet":
        df_texts = pd.read_parquet(texts_path)
    elif text_suffix == ".csv":
        df_texts = pd.read_csv(texts_path)
    else:
        msg = f"Formato de arquivo não suportado: {text_suffix}. Use .parquet ou .csv"
        logger.error(msg)
        raise ValueError(msg)

    # Carregar ground truth
    gt_suffix = gt_path.suffix.lower()
    if gt_suffix == ".parquet":
        df_ground_truth = pd.read_parquet(gt_path)
    elif gt_suffix == ".csv":
        df_ground_truth = pd.read_csv(gt_path)
    else:
        msg = f"Formato de arquivo não suportado: {gt_suffix}. Use .parquet ou .csv"
        logger.error(msg)
        raise ValueError(msg)

    return load_data(df_texts, df_ground_truth, logger)


def set_predictions(
    predictions: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Valida e configura DataFrame de predições.

    Args:
        predictions: DataFrame com predições
        logger: Logger para mensagens

    Returns:
        DataFrame com predições validadas

    Raises:
        ValueError: Se colunas necessárias estão ausentes
    """
    required_cols = {"id", "tp_entidade", "start_entidade", "end_entidade"}
    missing_cols = required_cols - set(predictions.columns)

    if missing_cols:
        msg = f"Colunas ausentes nas predições: {missing_cols}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Predições configuradas: %d entidades", len(predictions))

    return predictions
