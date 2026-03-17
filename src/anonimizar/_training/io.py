"""Utilitários de I/O para dados de treinamento NER.

Este módulo fornece funções para leitura e escrita de arquivos JSONL
compatíveis com o formato Doccano, usado para anotação de entidades.

Formatos JSONL suportados:
    - Formato 'labels': {"text": "...", "labels": [[start, end, "LABEL"], ...]}
    - Formato 'entities': {"text": "...", "entities": [{"start_offset": x, "end_offset": y, "label": "L"}]}
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


def load_doccano_jsonl(jsonl_path: str | Path) -> list[tuple[str, dict[str, list]]]:
    """Carrega dados de arquivo JSONL exportado do Doccano.

    Suporta dois formatos do Doccano:
    - {"text": "...", "labels": [[start, end, "LABEL"], ...]}
    - {"text": "...", "entities": [{"start_offset": x, "end_offset": y, "label": "L"}]}

    Args:
        jsonl_path: Caminho para arquivo .jsonl

    Returns:
        list: Dados no formato [(text, {"entities": [(start, end, label), ...]}), ...]

    Raises:
        FileNotFoundError: Se o arquivo não existir
    """
    logger = logging.getLogger(__name__)
    jsonl_path = Path(jsonl_path)

    if not jsonl_path.exists():
        msg = f"Arquivo não encontrado: {jsonl_path}"
        logger.exception(msg)
        raise FileNotFoundError(msg)

    training_data = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                text = data.get("text", "")

                if not text:
                    logger.warning(f"Linha {line_num}: texto vazio, ignorando")
                    continue

                entities = _extract_entities_from_record(data, line_num, logger)
                if entities is None:
                    continue

                training_data.append((text, {"entities": entities}))

            except json.JSONDecodeError as e:
                logger.exception(f"Linha {line_num}: erro JSON - {e!s}")
            except (KeyError, ValueError, TypeError) as e:
                logger.exception(f"Linha {line_num}: erro ao processar - {e!s}")

    logger.info(f"Carregados {len(training_data)} exemplos de {jsonl_path}")
    return training_data


def _extract_entities_from_record(
    data: dict[str, Any], line_num: int, logger: logging.Logger
) -> list[tuple[int, int, str]] | None:
    """Extrai entidades de um registro JSONL.

    Args:
        data: Dicionário com dados do registro JSONL
        line_num: Número da linha (para logging)
        logger: Logger para mensagens

    Returns:
        list: Lista de tuplas (start, end, label) ou None se inválido
    """
    if "labels" in data:
        labels = data["labels"]
        if not isinstance(labels, list):
            logger.warning(f"Linha {line_num}: 'labels' inválido")
            return None

        entities = []
        for item in labels:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                entities.append((int(item[0]), int(item[1]), str(item[2])))
        return entities

    if "entities" in data:
        entities_raw = data["entities"]
        entities = []

        for ent in entities_raw:
            if isinstance(ent, dict):
                if "start_offset" in ent:
                    entities.append((int(ent["start_offset"]), int(ent["end_offset"]), str(ent["label"])))
                elif "start" in ent:
                    entities.append((int(ent["start"]), int(ent["end"]), str(ent["label"])))
            elif isinstance(ent, (list, tuple)) and len(ent) >= 3:
                entities.append((int(ent[0]), int(ent[1]), str(ent[2])))
        return entities

    logger.warning(f"Linha {line_num}: sem 'labels' ou 'entities'")
    return None


def save_to_doccano_jsonl(
    output_path: str | Path, data: list[tuple[str, dict[str, list]]], format_type: str = "labels"
) -> None:
    """Salva dados em formato JSONL compatível com Doccano.

    Args:
        output_path: Caminho para salvar o arquivo .jsonl
        data: Dados no formato [(text, {"entities": [(start, end, label)]})]
        format_type: Tipo de formato:
            - 'labels': {"text": "...", "labels": [[start, end, "LABEL"]]}
            - 'entities': {"text": "...", "entities": [{"start_offset": x, ...}]}

    Raises:
        ValueError: Se format_type for inválido
    """
    logger = logging.getLogger(__name__)
    output_path = Path(output_path)

    if format_type not in ["labels", "entities"]:
        msg = f"format_type inválido: {format_type}. Use 'labels' ou 'entities'"
        logger.exception(msg)
        raise ValueError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for text, annotations in data:
            entities = annotations.get("entities", [])

            if format_type == "labels":
                record = {"text": text, "labels": list(entities)}
            else:
                formatted_entities = [
                    {"start_offset": start, "end_offset": end, "label": label} for start, end, label in entities
                ]
                record = {"text": text, "entities": formatted_entities}

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Salvos {len(data)} exemplos em {output_path}")


def load_jsonl_to_dataframes(jsonl_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Converte arquivo JSONL do Doccano para DataFrames.

    Formato JSONL esperado:
        {"text": "...", "labels": [[start, end, "LABEL"], ...]}
        ou
        {"text": "...", "entities": [{"start_offset": x, "end_offset": y, "label": "L"}]}

    Args:
        jsonl_path: Caminho para arquivo .jsonl

    Returns:
        tuple: (df_textos, df_entidades)
            - df_textos: DataFrame com colunas [id, text]
            - df_entidades: DataFrame com colunas [id, start, end, entidade]

    Raises:
        FileNotFoundError: Se o arquivo não existir
    """
    logger = logging.getLogger(__name__)
    jsonl_path = Path(jsonl_path)

    if not jsonl_path.exists():
        msg = f"Arquivo não encontrado: {jsonl_path}"
        logger.exception(msg)
        raise FileNotFoundError(msg)

    textos_data = []
    entidades_data = []

    with open(jsonl_path, encoding="utf-8") as f:
        for doc_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                text = data.get("text", "")

                if not text:
                    logger.warning(f"Linha {doc_id}: texto vazio, ignorando")
                    continue

                textos_data.append({"id": doc_id, "text": text})

                entities = _extract_entities_for_dataframe(data, doc_id)
                entidades_data.extend(entities)

            except json.JSONDecodeError as e:
                logger.exception(f"Linha {doc_id}: erro JSON - {e!s}")
            except (KeyError, ValueError, TypeError) as e:
                logger.exception(f"Linha {doc_id}: erro ao processar - {e!s}")

    df_textos = pd.DataFrame(textos_data)
    df_entidades = pd.DataFrame(entidades_data)

    logger.info(f"JSONL carregado: {len(df_textos)} documentos, {len(df_entidades)} entidades")

    return df_textos, df_entidades


def load_cv_input_data(
    df_entidades: pd.DataFrame | str,
    df_textos: pd.DataFrame | str | None,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega dados de entrada para cross-validation de múltiplos formatos.

    Aceita DataFrames ou caminhos para arquivos .jsonl. Converte automaticamente
    para o formato de DataFrames esperado.

    Args:
        df_entidades: DataFrame com entidades OU caminho para .jsonl
        df_textos: DataFrame com textos OU caminho para .jsonl OU None
        logger: Logger para mensagens

    Returns:
        tuple: (df_textos, df_entidades) como DataFrames

    Raises:
        ValueError: Se formato de arquivo não é suportado
        TypeError: Se após conversão não são DataFrames
    """
    from pathlib import Path

    if isinstance(df_entidades, str):
        entidades_path = Path(df_entidades)

        if entidades_path.suffix.lower() == ".jsonl":
            logger.info(f"Detectado arquivo JSONL: {df_entidades}")

            if df_textos is None:
                df_textos, df_entidades = load_jsonl_to_dataframes(df_entidades)

            elif isinstance(df_textos, str) and Path(df_textos).suffix.lower() == ".jsonl":
                logger.info(f"Detectado segundo arquivo JSONL: {df_textos}")
                df_textos_temp, df_entidades = load_jsonl_to_dataframes(df_entidades)
                df_textos, _ = load_jsonl_to_dataframes(df_textos)

            else:
                logger.info("Carregando entidades de JSONL, textos de DataFrame")
                df_textos_from_jsonl, df_entidades = load_jsonl_to_dataframes(df_entidades)
                if df_textos is None:
                    df_textos = df_textos_from_jsonl
        else:
            msg = f"Formato não suportado: {entidades_path.suffix}. Use .jsonl ou pd.DataFrame"
            logger.exception(msg)
            raise ValueError(msg)

    elif isinstance(df_textos, str):
        textos_path = Path(df_textos)

        if textos_path.suffix.lower() == ".jsonl":
            logger.info("Carregando textos de JSONL, entidades de DataFrame")
            df_textos, _ = load_jsonl_to_dataframes(df_textos)
        else:
            msg = f"Formato não suportado: {textos_path.suffix}. Use .jsonl ou pd.DataFrame"
            logger.exception(msg)
            raise ValueError(msg)

    if not isinstance(df_entidades, pd.DataFrame) or not isinstance(df_textos, pd.DataFrame):
        msg = "Após conversão, df_entidades e df_textos devem ser DataFrames"
        logger.exception(msg)
        raise TypeError(msg)

    return df_textos, df_entidades


def _extract_entities_for_dataframe(data: dict[str, Any], doc_id: int) -> list[dict[str, Any]]:
    """Extrai entidades de um registro JSONL para formato DataFrame.

    Args:
        data: Dicionário com dados do registro JSONL
        doc_id: ID do documento

    Returns:
        list: Lista de dicionários com entidades
    """
    entities = []

    if "labels" in data:
        labels = data["labels"]
        if isinstance(labels, list):
            for item in labels:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    entities.append(
                        {"id": doc_id, "start": int(item[0]), "end": int(item[1]), "entidade": str(item[2])}
                    )

    elif "entities" in data:
        entities_raw = data["entities"]
        for ent in entities_raw:
            if isinstance(ent, dict):
                if "start_offset" in ent:
                    entities.append(
                        {
                            "id": doc_id,
                            "start": int(ent["start_offset"]),
                            "end": int(ent["end_offset"]),
                            "entidade": str(ent["label"]),
                        }
                    )
                elif "start" in ent:
                    entities.append(
                        {
                            "id": doc_id,
                            "start": int(ent["start"]),
                            "end": int(ent["end"]),
                            "entidade": str(ent["label"]),
                        }
                    )
            elif isinstance(ent, (list, tuple)) and len(ent) >= 3:
                entities.append({"id": doc_id, "start": int(ent[0]), "end": int(ent[1]), "entidade": str(ent[2])})

    return entities
