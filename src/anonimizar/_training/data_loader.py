"""Carregamento e transformação de dados para treinamento NER.

Este módulo contém funções para transformar dados de diferentes formatos
(DataFrame Pandas, etc.) para o formato de treinamento spaCy.
"""

import logging
import random

import pandas as pd

from anonimizar._constants import (
    MIN_SPLIT_EXAMPLES,
    ROW_PER_ENTITY_COLUMNS,
    STANDARD_FORMAT_COLUMNS,
)


def transform_data_from_pandas(df: pd.DataFrame, logger: logging.Logger) -> list[tuple[str, dict[str, list]]]:
    """Converte DataFrame para o formato spaCy.

    Formatos aceitos:

    1. Colunas ``text`` e ``entities``
       - ``entities`` deve ser lista de tuplas.

    2. Formato "linha por entidade"
       - colunas ``texto``, ``start``, ``end`` e ``entidade``.

    Args:
        df: DataFrame a converter.
        logger: Logger para mensagens.

    Returns:
        list: Lista [(text, {"entities": [...]})].

    Raises:
        ValueError: Se o DataFrame não seguir nenhum dos formatos.
    """
    if all(col in df.columns for col in STANDARD_FORMAT_COLUMNS):
        logger.debug("Formato detectado: padrão (colunas 'text' e 'entities').")
        transformed = transform_standard_format(df, logger)
        logger.debug("Transformação concluída (padrão). Total de exemplos: %d", len(transformed))
        return transformed

    if all(col in df.columns for col in ROW_PER_ENTITY_COLUMNS):
        logger.debug("Formato detectado: linha por entidade (texto/start/end/entidade).")
        transformed = transform_row_per_entity_format(df, logger)
        logger.debug("Transformação concluída (linha por entidade). Total de exemplos: %d", len(transformed))
        return transformed

    msg = "DataFrame deve ter colunas 'text'/'entities' ou 'texto'/'start'/'end'/'entidade'"
    logger.exception(msg)
    raise ValueError(msg)


def transform_standard_format(df: pd.DataFrame, logger: logging.Logger) -> list[tuple[str, dict[str, list]]]:
    """Processa DataFrame com colunas ``text`` e ``entities``.

    Args:
        df: DataFrame com colunas 'text' e 'entities'.
        logger: Logger para mensagens.

    Returns:
        list: Lista [(text, {"entities": [...]})].
    """
    transformed = []
    invalid_rows = 0
    for _, row in df.iterrows():
        text = row.get("text", "")
        entities = row.get("entities", [])
        if not isinstance(entities, list):
            invalid_rows += 1
            logger.debug("Linha ignorada: entities inválido (não-lista). Valor=%r", entities)
            continue
        transformed.append((text, {"entities": entities}))
    if invalid_rows > 0:
        logger.debug("Total de linhas inválidas no formato padrão ignoradas: %d", invalid_rows)
    return transformed


def transform_row_per_entity_format(df: pd.DataFrame, logger: logging.Logger) -> list[tuple[str, dict[str, list]]]:
    """Processa DataFrame "uma linha por entidade".

    Args:
        df: DataFrame com colunas 'texto', 'start', 'end', 'entidade'.
        logger: Logger para mensagens.

    Returns:
        list: Lista [(text, {"entities": [...]})].
    """

    def combine_to_tuples(group: pd.DataFrame) -> list:
        entities = []
        for _, row in group.iterrows():
            start = row["start"]
            end = row["end"]
            label = row["entidade"]
            if pd.isna(start) or pd.isna(end) or pd.isna(label):
                logger.debug("Linha ignorada por dados faltantes: start=%r end=%r label=%r", start, end, label)
                continue
            try:
                start = int(start)
                end = int(end)
                label = str(label)
                entities.append((start, end, label))
            except ValueError as e:
                logger.debug("Falha ao converter tipos na linha: %s", e)
                continue

        return list(dict.fromkeys(entities))

    grouped = df.groupby("texto").apply(combine_to_tuples).reset_index()
    grouped.columns = ["text", "entities"]

    transformed = []
    for _, row in grouped.iterrows():
        text = row["text"]
        entities = row["entities"]
        transformed.append((text, {"entities": entities}))

    return transformed


def split_data(
    training_data: list[tuple[str, dict[str, list]]],
    train_ratio: float,
    logger: logging.Logger,
) -> tuple[list[tuple[str, dict[str, list]]], list[tuple[str, dict[str, list]]]]:
    """Divide dados em treino e validação.

    Args:
        training_data: Lista de dados de treinamento.
        train_ratio: Proporção destinada a treino (0-1).
        logger: Logger para mensagens.

    Returns:
        tuple: (train_data, val_data)

    Raises:
        ValueError: Se houver menos de 2 exemplos.
    """
    if len(training_data) < MIN_SPLIT_EXAMPLES:
        msg = "Dados insuficientes para divisão (mínimo: 2 exemplos)"
        logger.exception(msg)
        raise ValueError(msg)

    shuffled_data = training_data.copy()
    random.shuffle(shuffled_data)
    split_idx = max(1, int(len(shuffled_data) * train_ratio))
    train_data = shuffled_data[:split_idx]
    val_data = shuffled_data[split_idx:]

    logger.info(f"Dados divididos: {len(train_data)} treino, {len(val_data)} validação")

    return train_data, val_data


def val_data_to_evaluation(
    val_data: list[tuple[str, dict[str, list]]], logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Converte dados de validação para formato do SeiAnonimizarEvaluation.

    Args:
        val_data: Lista de dados de validação no formato spaCy.
        logger: Logger para mensagens.

    Returns:
        tuple: (df_texts, df_ground_truth)
            - df_texts: DataFrame com colunas [id, text]
            - df_ground_truth: DataFrame com colunas [id, tp_entidade, start_entidade, end_entidade]

    Raises:
        ValueError: Se val_data estiver vazio.
    """
    if not val_data:
        msg = "Não existem dados de validação."
        logger.exception(msg)
        raise ValueError(msg)

    texts, entities = [], []
    for idx, (text, ann) in enumerate(val_data, start=1):
        texts.append({"id": idx, "text": text})
        for s, e, label in ann["entities"]:
            entities.append({"id": idx, "tp_entidade": label, "start_entidade": s, "end_entidade": e})

    df_texts = pd.DataFrame(texts)
    df_ground_truth = pd.DataFrame(entities)

    logger.info("Conversão para avaliação concluída: textos=%d, entidades=%d", len(df_texts), len(df_ground_truth))
    return df_texts, df_ground_truth


def convert_input_data(
    data: list | dict | pd.DataFrame | str,
    logger: logging.Logger,
    transform_pandas_fn=None,
    load_jsonl_fn=None,
    errors: str = "raise",
) -> list[tuple[str, dict]]:
    """Converte dados de entrada para o formato spaCy interno.

    Aceita múltiplos formatos de entrada:
    - Lista de dicionários com 'text' e 'entities' ou 'labels'
    - Dicionário único com 'text' e 'entities' ou 'labels'
    - DataFrame Pandas
    - String (caminho para arquivo .jsonl)

    Args:
        data: Dados de entrada em qualquer formato aceito
        logger: Logger para mensagens
        transform_pandas_fn: Função para transformar DataFrame
        load_jsonl_fn: Função para carregar arquivo JSONL
        errors: Política de erro ('raise', 'coerce', 'ignore')

    Returns:
        list: Lista de tuplas (text, {"entities": [...]})

    Raises:
        TypeError: Se o formato de dados não é suportado
        ValueError: Se o arquivo não é .jsonl e errors='raise'
    """
    from pathlib import Path

    new_data = []

    if isinstance(data, list):
        for case in data:
            if "labels" in case and "entities" not in case:
                entities = []
                for item in case["labels"]:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        entities.append((int(item[0]), int(item[1]), str(item[2])))
                new_data.append((case["text"], {"entities": entities}))
            elif "entities" in case:
                new_data.append((case["text"], {"entities": case["entities"]}))
            else:
                logger.warning(f"Caso sem 'entities' ou 'labels': {list(case.keys())}")

    elif isinstance(data, dict):
        if "labels" in data and "entities" not in data:
            entities = []
            for item in data["labels"]:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    entities.append((int(item[0]), int(item[1]), str(item[2])))
            new_data.append((data["text"], {"entities": entities}))
        elif "entities" in data:
            new_data.append((data["text"], {"entities": data["entities"]}))
        else:
            logger.warning(f"Caso sem 'entities' ou 'labels': {list(data.keys())}")

    elif isinstance(data, pd.DataFrame):
        if transform_pandas_fn is None:
            msg = "transform_pandas_fn é necessário para converter DataFrame"
            raise ValueError(msg)
        transformed = transform_pandas_fn(data)
        new_data.extend(transformed)

    elif isinstance(data, str):
        data_path = Path(data)

        if data_path.suffix.lower() == ".jsonl":
            logger.info(f"Detectado arquivo JSONL: {data}")
            if load_jsonl_fn is None:
                msg = "load_jsonl_fn é necessário para carregar arquivo JSONL"
                raise ValueError(msg)
            loaded_data = load_jsonl_fn(data)
            new_data.extend(loaded_data)
        else:
            msg = f"Formato de arquivo não suportado: {data_path.suffix}. Use .jsonl"
            logger.exception(msg)
            if errors == "raise":
                raise ValueError(msg)
            return []
    else:
        msg = "Dados devem ser lista de dicionário, dicionario ou DataFrame Pandas, ou path jsonl."
        logger.exception(msg)
        raise TypeError(msg)

    return new_data


def apply_auto_clean(
    data: list[tuple[str, dict]],
    clean_entities_fn,
    strict_clean: bool,
    keep_empty_entities: bool,
    resolve_conflicts: str,
    logger: logging.Logger,
    errors: str = "coerce",
) -> list[tuple[str, dict]]:
    """Aplica limpeza automática aos dados de treinamento.

    Args:
        data: Lista de tuplas (text, {"entities": [...]})
        clean_entities_fn: Função para limpar entidades
        strict_clean: Se True, descarta exemplos com entidades removidas
        keep_empty_entities: Se True, mantém exemplos sem entidades
        resolve_conflicts: Como resolver conflitos ('raise', 'ignore', 'coerce')
        logger: Logger para mensagens
        errors: Política de erros ('raise', 'coerce', 'ignore')

    Returns:
        list: Lista de tuplas limpas
    """
    logger.debug(f"Aplicando limpeza automática {'estrita' if strict_clean else 'flexível'} em {len(data)} exemplos...")

    cleaned_data = []
    stats = {
        "original": len(data),
        "kept": 0,
        "discarded_strict": 0,
        "discarded_empty": 0,
        "discarded_conflicts": 0,
    }

    for text, annotations in data:
        entities = annotations.get("entities", [])
        original_count = len(entities)

        try:
            cleaned_entities = clean_entities_fn(
                text, entities, strict=strict_clean, resolve_conflicts=resolve_conflicts, errors=errors
            )

            if strict_clean and len(cleaned_entities) == 0 and original_count > 0:
                stats["discarded_strict"] += 1
                logger.debug(
                    f"Exemplo descartado (modo estrito): '{text[:50]}...' - "
                    f"{original_count} entidades originais não passaram na limpeza"
                )
            elif len(cleaned_entities) == 0 and not keep_empty_entities:
                stats["discarded_empty"] += 1
                logger.debug(f"Exemplo descartado (vazio): '{text[:50]}...'")
            else:
                cleaned_data.append((text, {"entities": cleaned_entities}))
                stats["kept"] += 1

                if len(cleaned_entities) != original_count:
                    logger.debug(f"Exemplo mantido com menos entidades: {original_count} -> {len(cleaned_entities)}")

        except ValueError as e:
            if "Conflitos de entidades" in str(e):
                stats["discarded_conflicts"] += 1
                logger.debug(f"Exemplo descartado por conflitos: '{text[:50]}...'")
            else:
                raise

    logger.debug(
        f"Limpeza concluída: {stats['kept']} mantidos, "
        f"{stats['discarded_strict']} descartados (modo estrito), "
        f"{stats['discarded_empty']} descartados (vazios), "
        f"{stats['discarded_conflicts']} descartados (conflitos)"
    )

    return cleaned_data
