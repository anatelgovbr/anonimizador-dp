"""Funções de cross-validation para treinamento NER.

Este módulo contém funções para criar folds, executar treinamento
cross-validation, e gerenciar holdout test sets.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from anonimizar._constants import (
    MAX_HOLDOUT_TEST_SIZE,
)


def make_folds_by_id(
    df_entidades: pd.DataFrame,
    n_splits: int,
    random_state: int,
    *,
    shuffle: bool,
    logger: logging.Logger,
) -> list[tuple]:
    """Cria folds baseados em IDs de documento para evitar vazamento de dados.

    Args:
        df_entidades: DataFrame com coluna 'id' (ou 'id_doc') identificando documentos
        n_splits: Número de folds para cross validation
        random_state: Seed para reprodutibilidade
        shuffle: Se deve embaralhar os dados antes da divisão
        logger: Logger para mensagens

    Returns:
        list: Lista de tuplas (ids_train, ids_val) para cada fold
    """
    id_col = "id_doc" if "id_doc" in df_entidades.columns else "id"
    ids = df_entidades[id_col].unique()

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = []

    for tr_idx, va_idx in kf.split(ids):
        ids_train = ids[tr_idx]
        ids_val = ids[va_idx]
        folds.append((ids_train, ids_val))

    logger.debug("Criados %d folds com KFold simples", len(folds))
    return folds


def make_stratified_folds_by_id(  # noqa: C901
    df_entidades: pd.DataFrame,
    features: list,
    n_splits: int,
    random_state: int,
    *,
    shuffle: bool,
    logger: logging.Logger,
) -> list:
    """Cria folds estratificados para balancear a distribuição de entidades entre os folds.

    Args:
        df_entidades: DataFrame com entidades anotadas
        features: Lista de tipos de entidades para estratificação
        n_splits: Número de folds
        random_state: Seed para reprodutibilidade
        shuffle: Se deve embaralhar os dados
        logger: Logger para mensagens

    Returns:
        list: Lista de tuplas ``(ids_train, ids_val)``, uma por fold. Cada
            ``ids_val`` contém os IDs destinados à validação daquele fold, e
            ``ids_train`` reúne os IDs dos demais folds.
    """
    id_col = "id_doc" if "id_doc" in df_entidades.columns else "id"
    entity_col = "tp_entidade" if "tp_entidade" in df_entidades.columns else "entidade"

    df_grouped = df_entidades.pivot_table(index=id_col, columns=entity_col, aggfunc="size", fill_value=0).reset_index()

    for col in features:
        if col in df_grouped.columns:
            df_grouped[col] = df_grouped[col].astype(bool)
        else:
            df_grouped[col] = False

    priority_list = (
        pd.DataFrame(df_grouped[features].sum(), columns=["contagem"])
        .sort_values("contagem")
        .reset_index()[entity_col]
        .tolist()
    )

    folds = [[] for _ in range(n_splits)]
    folds_counter = [0] * n_splits
    used_ids = []

    for ent in priority_list:
        available_docs = df_grouped[(df_grouped[ent]) & (~df_grouped[id_col].isin(used_ids))]

        if shuffle:
            available_docs = available_docs.sample(frac=1, random_state=random_state).reset_index()

        list_ids = available_docs[id_col].tolist()
        split_size, sobra = divmod(len(available_docs), n_splits)

        if split_size > 0:
            for i in range(n_splits):
                start_idx = i * split_size
                end_idx = start_idx + split_size
                ids_sample = list_ids[start_idx:end_idx]
                folds[i].extend(ids_sample)
                used_ids.extend(ids_sample)
                folds_counter[i] += split_size

            if sobra > 0:
                i = np.argmin(folds_counter)
                ids_sample = list_ids[n_splits * split_size :]
                folds[i].extend(ids_sample)
                used_ids.extend(ids_sample)
                folds_counter[i] += sobra

            else:
                for doc_id in list_ids:
                    i = np.argmin(folds_counter)
                    folds[i].append(doc_id)
                    used_ids.append(doc_id)
                    folds_counter[i] += 1

    remaining_ids = df_grouped[~df_grouped[id_col].isin(used_ids)][id_col].tolist()
    if remaining_ids:
        logger.debug("Distribuindo %d documentos restantes entre os folds", len(remaining_ids))
        for doc_id in remaining_ids:
            i = np.argmin(folds_counter)
            folds[i].append(doc_id)
            folds_counter[i] += 1

    logger.debug("Criados %d folds estratificados com contagens: %s", len(folds), folds_counter)
    return fold_for_train(folds, logger)


def fold_for_train(folds: list[list], logger: logging.Logger) -> list[tuple[list, list]]:
    """Transforma lista de folds em pares (train_ids, val_ids) para cada fold.

    Para cada fold i:
    - val = folds[i]
    - train = união de todos os outros folds

    Args:
        folds: Lista de listas, onde cada sublista contém IDs alocados a um fold
        logger: Logger para mensagens

    Returns:
        list: Lista de tuplas (train_ids, val_ids) para cada fold
    """
    n_folds = len(folds)
    fold_train = [[] for _ in range(n_folds)]
    fold_val = [[] for _ in range(n_folds)]

    for i in range(n_folds):
        for j, fold in enumerate(folds):
            if i == j:
                fold_val[i] = fold
            else:
                fold_train[i].extend(fold)

    result = list(zip(fold_train, fold_val, strict=True))
    logger.debug("Folds convertidos para treino/validação: %d pares gerados", len(result))
    return result


def separate_holdout_test(
    df_entidades: pd.DataFrame,
    df_textos: pd.DataFrame,
    holdout_test_size: float,
    features: list,
    random_state: int,
    output_path: Path,
    logger: logging.Logger,
    *,
    holdout_stratify: bool = True,
    make_stratified_folds_fn: Callable[..., list] | None = None,
    make_folds_fn: Callable[..., list] | None = None,
) -> tuple[list, list, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separa um conjunto de holdout test dos dados de cross-validation.

    Args:
        df_entidades: DataFrame com entidades anotadas
        df_textos: DataFrame com textos dos documentos
        holdout_test_size: Fração estritamente entre 0 e 0.5 dos dados para
            holdout test.
        features: Lista de tipos de entidades para estratificação
        random_state: Seed para reprodutibilidade
        output_path: Diretório para salvar informações do holdout
        logger: Logger para mensagens
        holdout_stratify: Se True, estratifica o holdout test
        make_stratified_folds_fn: Função para criar folds estratificados
        make_folds_fn: Função para criar folds simples

    Returns:
        tuple: (holdout_test_ids, cv_ids, df_holdout_texts, df_holdout_gt,
                df_entidades_cv, df_textos_cv)

    Raises:
        ValueError: Se holdout_test_size não estiver estritamente entre 0 e
            0.5, ou se nenhuma função de criação de folds for fornecida.
    """
    if not (0 < holdout_test_size < MAX_HOLDOUT_TEST_SIZE):
        msg = f"holdout_test_size deve estar entre 0 e 0.5, recebido {holdout_test_size}"
        logger.exception(msg)
        raise ValueError(msg)

    id_col = "id" if "id" in df_entidades.columns else "id_doc"
    text_col = "text" if "text" in df_textos.columns else "texto"
    entity_col = "tp_entidade" if "tp_entidade" in df_entidades.columns else "entidade"
    start_col = "start_entidade" if "start_entidade" in df_entidades.columns else "start"
    end_col = "end_entidade" if "end_entidade" in df_entidades.columns else "end"

    # Criar folds para holdout separation
    n_folds_holdout = int(1 / holdout_test_size)

    if holdout_stratify and features and make_stratified_folds_fn:
        logger.info("Separando holdout test estratificado (%.0f%%)", holdout_test_size * 100)
        folds_holdout = make_stratified_folds_fn(
            df_entidades=df_entidades, features=features, n_splits=n_folds_holdout, random_state=random_state
        )
    elif make_folds_fn:
        logger.info("Separando holdout test simples (%.0f%%)", holdout_test_size * 100)
        folds_holdout = make_folds_fn(df_entidades=df_entidades, n_splits=n_folds_holdout, random_state=random_state)
    else:
        msg = "É necessário fornecer make_stratified_folds_fn ou make_folds_fn"
        raise ValueError(msg)

    holdout_test_ids = list(folds_holdout[0][1])
    cv_ids = list(folds_holdout[0][0])

    # Preparar DataFrames de holdout
    df_holdout_texts = df_textos[df_textos[id_col].isin(holdout_test_ids)][[id_col, text_col]].rename(
        columns={id_col: "id", text_col: "text"}
    )
    df_holdout_gt = df_entidades[df_entidades[id_col].isin(holdout_test_ids)][
        [id_col, entity_col, start_col, end_col]
    ].rename(
        columns={
            id_col: "id",
            entity_col: "tp_entidade",
            start_col: "start_entidade",
            end_col: "end_entidade",
        }
    )

    logger.info("Holdout test separado: %d docs (test), %d docs para CV", len(holdout_test_ids), len(cv_ids))

    # Salvar informações do holdout
    holdout_info = {
        "holdout_test_ids": sorted([int(x) for x in holdout_test_ids]),
        "cv_ids": sorted([int(x) for x in cv_ids]),
        "n_test": len(holdout_test_ids),
        "n_cv": len(cv_ids),
        "test_size": float(holdout_test_size),
        "stratified": bool(holdout_stratify),
        "random_state": int(random_state),
    }

    with (output_path / "holdout_test_ids.json").open("w", encoding="utf-8") as f:
        json.dump(holdout_info, f, indent=2, ensure_ascii=False)

    pd.DataFrame({"id": sorted([int(x) for x in holdout_test_ids]), "split": "holdout_test"}).to_csv(
        output_path / "holdout_test_ids.csv", index=False
    )

    # Filtrar dados para CV (excluindo holdout)
    df_entidades_cv = df_entidades[df_entidades[id_col].isin(cv_ids)].copy()
    df_textos_cv = df_textos[df_textos[id_col].isin(cv_ids)].copy()

    logger.info("Cross-validation será executado apenas no CV set (excluindo holdout test)")

    return holdout_test_ids, cv_ids, df_holdout_texts, df_holdout_gt, df_entidades_cv, df_textos_cv


def aggregate_cv_results(
    results: list[tuple],
    output_path: Path,
    holdout_test_size: float | None,
    logger: logging.Logger,
) -> tuple[list, list, list | None]:
    """Agrega resultados de todos os folds do cross-validation.

    Args:
        results: Lista de tuplas (report, summary, holdout_summary) de cada fold
        output_path: Diretório para salvar resultados agregados
        holdout_test_size: Tamanho do holdout test (None se não houver holdout)
        logger: Logger para mensagens

    Returns:
        tuple: (all_reports, summary_metrics, holdout_results)
    """
    all_reports = [r[0] for r in results]
    summary_metrics = [r[1] for r in results if r[1] is not None]
    holdout_results = [r[2] for r in results if r[2] is not None]

    if summary_metrics:
        df_summary = pd.DataFrame(summary_metrics)
        avg_metrics = df_summary.mean(numeric_only=True)
        logger.info(
            "Métricas médias: Precision=%.3f, Recall=%.3f, Fbeta=%.3f",
            avg_metrics["precision"],
            avg_metrics["recall"],
            avg_metrics["fbeta"],
        )

        pd.concat(all_reports).to_parquet(output_path / "all_folds_detailed.parquet", index=False)
        df_summary.to_parquet(output_path / "fold_summaries.parquet", index=False)

    if holdout_results:
        df_holdout_summary = pd.DataFrame(holdout_results)
        df_holdout_summary.to_parquet(output_path / "holdout_test_summary.parquet", index=False)
        df_holdout_summary.to_csv(output_path / "holdout_test_summary.csv", index=False)

        holdout_stats = {
            "mean_precision": df_holdout_summary["precision"].mean(),
            "std_precision": df_holdout_summary["precision"].std(),
            "mean_recall": df_holdout_summary["recall"].mean(),
            "std_recall": df_holdout_summary["recall"].std(),
            "mean_fbeta": df_holdout_summary["fbeta"].mean(),
            "std_fbeta": df_holdout_summary["fbeta"].std(),
            "best_fold": str(df_holdout_summary.loc[df_holdout_summary["fbeta"].idxmax(), "fold"]),
            "best_fbeta": float(df_holdout_summary["fbeta"].max()),
        }

        with (output_path / "holdout_test_stats.json").open("w") as f:
            json.dump(holdout_stats, f, indent=2)

        logger.info(
            "Holdout Test - Média: Fbeta=%.3f±%.3f, Melhor fold: %s (Fbeta=%.3f)",
            holdout_stats["mean_fbeta"],
            holdout_stats["std_fbeta"],
            holdout_stats["best_fold"],
            holdout_stats["best_fbeta"],
        )

    holdout_results = None if holdout_test_size is None else holdout_results
    return all_reports, summary_metrics, holdout_results


def prepare_cv_data(
    df_entidades: pd.DataFrame,
    df_textos: pd.DataFrame,
    model_name: str,
    supported_labels: list,
    add_data_params: dict,
    logger: logging.Logger,
) -> tuple[list, pd.DataFrame]:
    """Prepara dados para cross-validation com pré-processamento via add_data.

    Aplica limpeza e validação em todos os dados antes de dividir em folds,
    garantindo consistência no pré-processamento.

    Args:
        df_entidades: DataFrame com entidades anotadas
        df_textos: DataFrame com textos dos documentos
        model_name: Nome do modelo spaCy
        supported_labels: Lista de labels suportados
        add_data_params: Parâmetros para o método add_data
        logger: Logger para mensagens

    Returns:
        tuple: (cleaned_training_data, id_to_index)
            - cleaned_training_data: Lista de tuplas (text, {"entities": [...]})
            - id_to_index: DataFrame mapeando id -> índice no cleaned_training_data
    """
    # Importação local para evitar import circular
    from .trainer_facade import Trainer

    id_col = "id" if "id" in df_entidades.columns else "id_doc"
    text_col = "text" if "text" in df_textos.columns else "texto"
    entity_col = "tp_entidade" if "tp_entidade" in df_entidades.columns else "entidade"
    start_col = "start_entidade" if "start_entidade" in df_entidades.columns else "start"
    end_col = "end_entidade" if "end_entidade" in df_entidades.columns else "end"

    logger.debug("Iniciando pré-processamento de todos os dados com add_data...")

    # Preparar DataFrame para add_data
    df_entidades_treino = df_entidades[[id_col, start_col, end_col, entity_col]].copy()
    df_entidades_treino = df_entidades_treino.merge(df_textos[[id_col, text_col]], on=id_col)
    df_entidades_treino.columns = ["id", "start", "end", "entidade", "texto"]

    # Criar trainer temporário para processar dados
    temp_trainer = Trainer(model_name=model_name, labels=supported_labels, logger=logger)
    temp_trainer.add_data(df_entidades_treino, **add_data_params)

    cleaned_training_data = temp_trainer.training_data.copy()
    logger.debug("Pré-processamento concluído: %d exemplos limpos", len(cleaned_training_data))

    # Criar mapeamento id -> índice
    id_to_index = df_entidades_treino.drop_duplicates("id").merge(
        df_entidades_treino.groupby("texto")["id"].max().reset_index().reset_index(names=["idx"])
    )[["idx", "id"]]

    return cleaned_training_data, id_to_index


def prepare_fold_data(
    fold_idx: int,
    ids_train: list,
    ids_val: list,
    id_to_index: pd.DataFrame,
    cleaned_training_data: list,
    df_entidades: pd.DataFrame,
    df_textos: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[list, pd.DataFrame, pd.DataFrame]:
    """Prepara dados de treino e validação para um fold específico.

    Args:
        fold_idx: Índice do fold
        ids_train: IDs dos documentos de treino
        ids_val: IDs dos documentos de validação
        id_to_index: DataFrame mapeando id -> índice no cleaned_training_data
        cleaned_training_data: Lista de tuplas (text, {"entities": [...]}) pré-processadas
        df_entidades: DataFrame com entidades anotadas
        df_textos: DataFrame com textos dos documentos
        logger: Logger para mensagens

    Returns:
        tuple: (fold_training_data, df_texts_val, df_gt_val)
    """
    id_col = "id" if "id" in df_entidades.columns else "id_doc"
    text_col = "text" if "text" in df_textos.columns else "texto"
    entity_col = "tp_entidade" if "tp_entidade" in df_entidades.columns else "entidade"
    start_col = "start_entidade" if "start_entidade" in df_entidades.columns else "start"
    end_col = "end_entidade" if "end_entidade" in df_entidades.columns else "end"

    # Preparar dados de treino do fold
    train_indices = id_to_index[id_to_index["id"].isin(ids_train)]["idx"].tolist()
    fold_training_data = [cleaned_training_data[i] for i in train_indices if i < len(cleaned_training_data)]

    # Preparar dados de validação do fold
    df_val_entities = df_entidades[df_entidades[id_col].isin(ids_val)].copy()
    df_val_texts = df_textos[df_textos[id_col].isin(ids_val)].copy()

    df_texts_val = df_val_texts[[id_col, text_col]].rename(columns={id_col: "id", text_col: "text"})
    df_gt_val = df_val_entities[[id_col, entity_col, start_col, end_col]].rename(
        columns={id_col: "id", entity_col: "tp_entidade", start_col: "start_entidade", end_col: "end_entidade"}
    )

    logger.debug(
        "Fold %d: %d exemplos de treino, %d de validação",
        fold_idx,
        len(fold_training_data),
        len(df_texts_val),
    )

    return fold_training_data, df_texts_val, df_gt_val


def save_fold_ids(
    fold_idx: int,
    ids_train: list,
    ids_val: list,
    fold_training_data: list,
    df_texts_val: pd.DataFrame,
    fold_dir: Path,
    logger: logging.Logger,
) -> None:
    """Salva informações dos IDs de documentos de um fold.

    Args:
        fold_idx: Índice do fold
        ids_train: IDs dos documentos de treino
        ids_val: IDs dos documentos de validação
        fold_training_data: Dados de treino do fold
        df_texts_val: DataFrame com textos de validação
        fold_dir: Diretório do fold
        logger: Logger para mensagens
    """
    fold_ids_info = {
        "fold": fold_idx,
        "train_ids": sorted([int(x) for x in ids_train]),
        "val_ids": sorted([int(x) for x in ids_val]),
        "n_train": len(ids_train),
        "n_val": len(ids_val),
        "train_samples": len(fold_training_data),
        "val_samples": len(df_texts_val),
    }

    with (fold_dir / "fold_ids.json").open("w", encoding="utf-8") as f:
        json.dump(fold_ids_info, f, indent=2, ensure_ascii=False)

    pd.DataFrame({"id": sorted([int(x) for x in ids_train]), "split": "train"}).to_csv(
        fold_dir / "train_ids.csv", index=False
    )

    pd.DataFrame({"id": sorted([int(x) for x in ids_val]), "split": "val"}).to_csv(
        fold_dir / "val_ids.csv", index=False
    )

    logger.debug("Fold %d: IDs salvos - %d treino, %d validação", fold_idx, len(ids_train), len(ids_val))
