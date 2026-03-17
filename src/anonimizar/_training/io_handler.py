"""Handler de I/O para dados de treinamento NER em formato JSONL.

Este módulo expõe funções canônicas para carregar e salvar dados
no formato JSONL compatível com Doccano, delegando para o módulo
interno `_training/io.py`.

Funções públicas:
    - ``load_from_doccano_jsonl``: carrega arquivo .jsonl → lista spaCy
    - ``save_to_doccano_jsonl``: salva lista spaCy → arquivo .jsonl
    - ``load_jsonl_to_dataframes``: carrega .jsonl → (df_textos, df_entidades)
"""

from pathlib import Path

import pandas as pd

from anonimizar._training import io as _io

__all__ = [
    "load_from_doccano_jsonl",
    "load_jsonl_to_dataframes",
    "save_to_doccano_jsonl",
]


def load_from_doccano_jsonl(
    jsonl_path: str | Path,
) -> list[tuple[str, dict[str, list]]]:
    """Carrega arquivo JSONL exportado do Doccano para formato spaCy.

    Suporta dois formatos do Doccano:

    - ``{"text": "...", "labels": [[start, end, "LABEL"], ...]}``
    - ``{"text": "...", "entities": [{"start_offset": x, "end_offset": y, "label": "L"}]}``

    Args:
        jsonl_path: Caminho para arquivo ``.jsonl``.

    Returns:
        Lista de tuplas ``(text, {"entities": [(start, end, label), ...]})``
        pronta para uso no treinamento spaCy.

    Raises:
        FileNotFoundError: Se o arquivo não existir.

    Examples:
        >>> data = load_from_doccano_jsonl("anotacoes.jsonl")
        >>> len(data)
        42
    """
    return _io.load_doccano_jsonl(jsonl_path)


def save_to_doccano_jsonl(
    output_path: str | Path,
    data: list[tuple[str, dict[str, list]]],
    *,
    format_type: str = "labels",
) -> None:
    """Salva dados de treinamento spaCy em arquivo JSONL compatível com Doccano.

    Args:
        output_path: Caminho de saída do arquivo ``.jsonl``.
        data: Lista de tuplas ``(text, {"entities": [(start, end, label)]})``.
        format_type: Formato de saída:
            - ``"labels"``: ``{"text": "...", "labels": [[start, end, "LABEL"]]}``
            - ``"entities"``: ``{"text": "...", "entities": [{"start_offset": x, ...}]}``

    Raises:
        ValueError: Se ``format_type`` for inválido.

    Examples:
        >>> data = [("CPF 123.456.789-09", {"entities": [(4, 18, "CPF")]})]
        >>> save_to_doccano_jsonl("saida.jsonl", data)
    """
    _io.save_to_doccano_jsonl(output_path, data, format_type)


def load_jsonl_to_dataframes(
    jsonl_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega arquivo JSONL do Doccano e retorna dois DataFrames.

    Args:
        jsonl_path: Caminho para arquivo ``.jsonl``.

    Returns:
        Tupla ``(df_textos, df_entidades)`` onde:

        - ``df_textos``: colunas ``[id, text]``
        - ``df_entidades``: colunas ``[id, start, end, entidade]``

    Raises:
        FileNotFoundError: Se o arquivo não existir.

    Examples:
        >>> df_textos, df_ents = load_jsonl_to_dataframes("anotacoes.jsonl")
        >>> df_textos.columns.tolist()
        ['id', 'text']
    """
    return _io.load_jsonl_to_dataframes(jsonl_path)
