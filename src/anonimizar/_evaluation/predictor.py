"""Módulo para extração de predições de modelos NER.

Este módulo contém funções para extrair, carregar e salvar predições
de modelos de anonimização.
"""

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def extract_predictions(
    df_texts: pd.DataFrame,
    anonymizer,
    entity_mapping: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Extrai predições do modelo de anonimização.

    Processa cada texto do DataFrame e extrai todas as entidades detectadas
    pelo modelo, aplicando mapeamento de labels se fornecido.

    Args:
        df_texts: DataFrame com colunas ['id', 'text']
        anonymizer: Instância de SeiAnonimizar para extração
        entity_mapping: Dicionário para mapear labels (ex: {'PESSOA': 'CPF'})
        logger: Logger para mensagens

    Returns:
        DataFrame com colunas ['id', 'tp_entidade', 'text_entidade',
                               'start_entidade', 'end_entidade', 'detected_by']
    """
    if df_texts is None or len(df_texts) == 0:
        msg = "DataFrame de textos está vazio"
        logger.error(msg)
        raise ValueError(msg)

    if anonymizer is None:
        msg = "Objeto 'anonymizer' não fornecido"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Extraindo predições de %d textos", len(df_texts))

    predictions_data = []

    for _idx, row in tqdm(df_texts.iterrows(), total=len(df_texts), desc="Extraindo entidades"):
        try:
            text_val = row["text"]
            if not isinstance(text_val, str):
                logger.warning("Texto ID %s não é string. Convertendo via str().", row["id"])
                text_val = str(text_val)

            entities = anonymizer.extract_entities(text_or_path=text_val, return_type="label_detail")

            for entity in entities:
                predictions_data.append(
                    {
                        "id": row["id"],
                        "tp_entidade": entity["label"],
                        "text_entidade": entity["text"],
                        "start_entidade": entity["start_position"],
                        "end_entidade": entity["end_position"],
                        "detected_by": str(entity["detected_by"]),
                    }
                )

        except Exception as e:  # noqa: BLE001
            logger.warning("Erro ao processar texto ID %s: %s", row.get("id"), e)
            continue

    df_predictions = pd.DataFrame(predictions_data)

    if len(df_predictions) == 0:
        logger.warning("Nenhuma entidade foi extraída das predições")
        return df_predictions

    # Aplicar mapeamento de entidades
    df_predictions["tp_entidade"] = df_predictions["tp_entidade"].apply(lambda x: entity_mapping.get(x, x))

    logger.info("Extraídas %d entidades", len(df_predictions))

    return df_predictions


def load_predictions(predictions_path: str | Path, logger: logging.Logger) -> pd.DataFrame:
    """Carrega predições de arquivo Parquet ou CSV.

    Args:
        predictions_path: Caminho para arquivo de predições
        logger: Logger para mensagens

    Returns:
        DataFrame com predições

    Raises:
        FileNotFoundError: Se arquivo não existe
        ValueError: Se formato não é suportado
    """
    path = Path(predictions_path)

    if not path.exists():
        msg = f"Arquivo de predições não encontrado: {predictions_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(path)
        logger.info("Predições carregadas de %s (%d registros)", predictions_path, len(df))
    elif suffix == ".csv":
        df = pd.read_csv(path)
        logger.info("Predições carregadas de %s (%d registros)", predictions_path, len(df))
    else:
        msg = f"Formato de arquivo não suportado: {suffix}. Use .parquet ou .csv"
        logger.error(msg)
        raise ValueError(msg)

    return df


def save_predictions(
    df_predictions: pd.DataFrame,
    save_path: str | Path,
    logger: logging.Logger,
) -> None:
    """Salva predições em arquivo Parquet ou CSV.

    Args:
        df_predictions: DataFrame com predições
        save_path: Caminho para salvar arquivo
        logger: Logger para mensagens

    Raises:
        ValueError: Se formato não é suportado
    """
    path = Path(save_path)
    suffix = path.suffix.lower()

    # Criar diretório se não existir
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".parquet":
        df_predictions.to_parquet(path, index=False)
        logger.info("Predições salvas em %s (%d registros)", save_path, len(df_predictions))
    elif suffix == ".csv":
        df_predictions.to_csv(path, index=False)
        logger.info("Predições salvas em %s (%d registros)", save_path, len(df_predictions))
    else:
        msg = f"Formato de arquivo não suportado: {suffix}. Use .parquet ou .csv"
        logger.error(msg)
        raise ValueError(msg)
