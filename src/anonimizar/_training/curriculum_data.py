"""Preparação de dados em janelas de dificuldade para curriculum learning.

Este módulo reproduz no pacote a preparação de datasets por janelas de
dificuldade usada nos experimentos de curriculum learning da estória 942
(``notebooks/Estorias/matheus/sprint_82/estoria_942/treinos/``):

- ``w0``/``w1``/``w2``: parágrafo central com janela de contexto de
  ``0``/``1``/``2`` parágrafos ao redor, com offsets das entidades
  reposicionados para o texto montado.
- ``full``: documento inteiro.
- ``w00``: entidade isolada de ponta a ponta ``[(0, len(text), label)]``.

As funções aceitam DataFrames de textos (``id``, ``text``) e entidades
(``id``, ``start_entidade``/``end_entidade``/``tp_entidade`` ou ``id``,
``start``/``end``/``entidade``) e devolvem exemplos no formato interno do
pacote ``(texto, {"entities": [(start, end, label), ...]})``, consumíveis
diretamente pelo ``Trainer.train_curriculum()``.

Filtros obrigatórios aplicados (herdados da 942): documentos com
``TEM_ERRO == True`` removidos por completo, entidades com label contendo
``_remover`` descartadas, textos duplicados por ``id`` (mantém o último),
entidades duplicadas e spans inválidos (``start >= end``) removidos.
"""

import logging
from pathlib import Path

import pandas as pd

from anonimizar._common.logging import create_default_logger
from anonimizar._constants import DEFAULT_CURRICULUM_WINDOWS

__all__ = [
    "build_context_window_dataset",
    "build_curriculum_datasets",
    "build_full_text_dataset",
    "build_pure_entity_dataset",
    "load_curriculum_datasets",
    "save_curriculum_datasets",
]

Example = tuple[str, dict[str, list[tuple[int, int, str]]]]

_WINDOW_FULL = "full"
_JANELA_MAP = {"w0": 0, "w1": 1, "w2": 2, _WINDOW_FULL: _WINDOW_FULL, "w00": "w00"}
_MIN_COMPONENTES_ENTIDADE = 3
_TAM_EXEMPLO = 2


def _normalizar_textos(df_textos: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Normaliza colunas do DataFrame de textos para ``id`` e ``text``."""
    df_norm = df_textos.copy()
    if "id_documento" in df_norm.columns and "id" not in df_norm.columns:
        df_norm = df_norm.rename(columns={"id_documento": "id"})
    if "id" not in df_norm.columns or "text" not in df_norm.columns:
        msg = f"df_textos deve conter colunas 'id' e 'text'. Encontradas: {list(df_norm.columns)}"
        logger.error(msg)
        raise ValueError(msg)
    return df_norm[["id", "text"]].copy()


def _normalizar_entidades(df_entidades: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Normaliza colunas de entidades para ``id``/``start_entidade``/``end_entidade``/``tp_entidade``."""
    df_norm = df_entidades.copy()
    if "id_documento" in df_norm.columns and "id" not in df_norm.columns:
        df_norm = df_norm.rename(columns={"id_documento": "id"})
    if "id" not in df_norm.columns:
        msg = f"df_entidades deve conter coluna 'id'. Encontradas: {list(df_norm.columns)}"
        logger.error(msg)
        raise ValueError(msg)
    if "start_entidade" not in df_norm.columns:
        if "start" in df_norm.columns:
            df_norm = df_norm.rename(columns={"start": "start_entidade", "end": "end_entidade"})
        else:
            msg = (
                "df_entidades deve conter colunas 'start'/'end' (ou 'start_entidade'/'end_entidade'). "
                f"Encontradas: {list(df_norm.columns)}"
            )
            logger.error(msg)
            raise ValueError(msg)
    if "tp_entidade" not in df_norm.columns:
        if "entidade" in df_norm.columns:
            df_norm = df_norm.rename(columns={"entidade": "tp_entidade"})
        else:
            msg = f"df_entidades deve conter coluna 'tp_entidade' (ou 'entidade'). Encontradas: {list(df_norm.columns)}"
            logger.error(msg)
            raise ValueError(msg)
    return df_norm[["id", "start_entidade", "end_entidade", "tp_entidade"]].copy()


def _filtrar(
    df_textos: pd.DataFrame,
    df_entidades: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normaliza colunas e aplica os filtros obrigatórios da 942.

    A checagem de ``TEM_ERRO`` acontece antes da seleção de colunas para não
    perder a flag durante a normalização.
    """
    df_t = df_textos.copy()
    df_e = df_entidades.copy()
    if "id_documento" in df_t.columns and "id" not in df_t.columns:
        df_t = df_t.rename(columns={"id_documento": "id"})
    if "id_documento" in df_e.columns and "id" not in df_e.columns:
        df_e = df_e.rename(columns={"id_documento": "id"})

    if "TEM_ERRO" in df_e.columns:
        ids_remover = df_e.loc[df_e["TEM_ERRO"].astype(bool), "id"].unique()
        df_e = df_e[~df_e["id"].isin(ids_remover)].reset_index(drop=True)
        df_t = df_t[~df_t["id"].isin(ids_remover)].reset_index(drop=True)

    df_t = _normalizar_textos(df_t, logger)
    df_e = _normalizar_entidades(df_e, logger)

    df_e = df_e[~df_e["tp_entidade"].astype(str).str.contains("_remover", regex=False)].reset_index(drop=True)

    df_t = df_t.drop_duplicates(subset="id", keep="last").reset_index(drop=True)
    df_e = df_e.drop_duplicates(subset=["id", "start_entidade", "end_entidade", "tp_entidade"]).reset_index(drop=True)
    df_e = df_e[df_e["start_entidade"].astype(int) < df_e["end_entidade"].astype(int)].reset_index(drop=True)

    ids_com_entidades = df_e["id"].unique()
    df_t = df_t[df_t["id"].isin(ids_com_entidades)].reset_index(drop=True)

    logger.debug("Após filtros: %d textos e %d entidades", len(df_t), len(df_e))
    return df_t, df_e


def _split_paragrafos(texto: str, doc_id: object) -> list[dict[str, object]]:
    """Divide o texto em parágrafos não vazios com offsets absolutos no documento."""
    paragrafos = []
    start = 0
    par_idx = 0
    for paragrafo in texto.split("\n"):
        end = start + len(paragrafo)
        if paragrafo.strip():
            paragrafos.append({"id": doc_id, "par_idx": par_idx, "text": paragrafo, "start": start, "end": end})
            par_idx += 1
        start = end + 1
    return paragrafos


def _build_janelas_contexto(
    df_textos: pd.DataFrame,
    df_entidades: pd.DataFrame,
    window: int,
    _logger: logging.Logger,
) -> list[Example]:
    """Monta exemplos de parágrafo central com janela de contexto."""
    exemplos: list[Example] = []
    for _, row in df_textos.iterrows():
        doc_id = row["id"]
        paragrafos = _split_paragrafos(row["text"], doc_id)
        doc_ents = df_entidades[df_entidades["id"].eq(doc_id)]
        if not paragrafos or doc_ents.empty:
            continue
        for central in paragrafos:
            ctx = [p for p in paragrafos if central["par_idx"] - window <= p["par_idx"] <= central["par_idx"] + window]
            if not ctx:
                continue
            ctx.sort(key=lambda p: p["par_idx"])
            text_montado = "\n\n".join(p["text"] for p in ctx)
            entidades: list[tuple[int, int, str]] = []
            offset = 0
            for p in ctx:
                mask = doc_ents["start_entidade"].astype(int).ge(p["start"]) & doc_ents["end_entidade"].astype(int).le(
                    p["end"]
                )
                for _, ent in doc_ents[mask].iterrows():
                    start = int(ent["start_entidade"]) - p["start"] + offset
                    end = int(ent["end_entidade"]) - p["start"] + offset
                    entidades.append((start, end, str(ent["tp_entidade"])))
                offset += len(p["text"]) + 2
            if entidades:
                exemplos.append((text_montado, {"entities": entidades}))
    return exemplos


def _build_documentos_completos(
    df_textos: pd.DataFrame,
    df_entidades: pd.DataFrame,
    _logger: logging.Logger,
) -> list[Example]:
    """Monta exemplos com o documento inteiro e todas as entidades."""
    exemplos: list[Example] = []
    for _, row in df_textos.iterrows():
        doc_id = row["id"]
        doc_ents = df_entidades[df_entidades["id"].eq(doc_id)]
        if doc_ents.empty:
            continue
        doc_ents = doc_ents.sort_values("start_entidade")
        entidades = [
            (int(e["start_entidade"]), int(e["end_entidade"]), str(e["tp_entidade"])) for _, e in doc_ents.iterrows()
        ]
        exemplos.append((row["text"], {"entities": entidades}))
    return exemplos


def build_context_window_dataset(
    df_textos: pd.DataFrame,
    df_entidades: pd.DataFrame,
    *,
    window: int | str = 1,
    logger: logging.Logger | None = None,
) -> list[Example]:
    r"""Constrói exemplos com janela de contexto de parágrafos.

    Para cada parágrafo com entidades, monta o texto dos parágrafos em
    ``[par - window, par + window]`` juntados com ``"\n\n"`` e reposiciona os
    offsets das entidades para o texto montado.

    Args:
        df_textos: DataFrame com colunas ``id`` e ``text`` (``id_documento``
            também é aceito).
        df_entidades: DataFrame com ``id``, ``start_entidade``,
            ``end_entidade`` e ``tp_entidade`` (ou ``id``, ``start``, ``end`` e
            ``entidade``).
        window: Tamanho da janela de contexto (`0` = só o parágrafo, `1` = ±1
            parágrafo, etc.) ou ``"full"`` para o documento inteiro.
        logger: Logger para mensagens. Se None, cria logger padrão.

    Returns:
        Lista de exemplos ``(texto, {"entities": [(start, end, label), ...]})``.

    Raises:
        ValueError: Se ``window`` for inválido ou faltar coluna obrigatória.
        TypeError: Se ``window`` não for int nem str.
    """
    if logger is None:
        logger = create_default_logger(__name__)

    if isinstance(window, int):
        if window < 0:
            msg = f"window deve ser int >= 0 ou 'full', recebido: {window}"
            logger.exception(msg)
            raise ValueError(msg)
    elif isinstance(window, str):
        if window != _WINDOW_FULL:
            msg = f"window deve ser int >= 0 ou 'full', recebido: {window!r}"
            logger.exception(msg)
            raise ValueError(msg)
    else:
        msg = f"window deve ser int >= 0 ou 'full', recebido: {type(window).__name__}"
        logger.exception(msg)
        raise TypeError(msg)

    df_t, df_e = _filtrar(df_textos, df_entidades, logger)

    if isinstance(window, str):
        return _build_documentos_completos(df_t, df_e, logger)
    return _build_janelas_contexto(df_t, df_e, window, logger)


def build_full_text_dataset(
    df_textos: pd.DataFrame,
    df_entidades: pd.DataFrame,
    *,
    logger: logging.Logger | None = None,
) -> list[Example]:
    """Constrói exemplos com o documento inteiro (janela ``full``).

    Equivale a ``build_context_window_dataset(..., window="full")``.

    Args:
        df_textos: DataFrame com colunas ``id`` e ``text``.
        df_entidades: DataFrame com ``id``, ``start_entidade``,
            ``end_entidade`` e ``tp_entidade`` (ou formato alternativo).
        logger: Logger para mensagens.

    Returns:
        Lista de exemplos com o texto completo do documento.
    """
    return build_context_window_dataset(df_textos, df_entidades, window=_WINDOW_FULL, logger=logger)


def build_pure_entity_dataset(
    df_textos: pd.DataFrame,
    df_entidades: pd.DataFrame,
    *,
    oversample: dict[str, int] | None = None,
    logger: logging.Logger | None = None,
) -> list[Example]:
    """Constrói exemplos de entidade pura isolada (janela ``w00``).

    Cada entidade vira uma amostra com o texto da entidade e anotação de ponta
    a ponta ``[(0, len(texto), label)]``. Entidades cujo label aparece em
    ``oversample`` são repetidas pelo fator informado (ex.: ``{"ENDEREÇO": 3}``
    na 942). Não realiza pseudonimização/substituição sintética: usa o valor
    real da entidade (variante "puro" dos experimentos).

    Args:
        df_textos: DataFrame com colunas ``id`` e ``text``.
        df_entidades: DataFrame com ``id``, ``start_entidade``,
            ``end_entidade`` e ``tp_entidade`` (ou formato alternativo).
        oversample: Mapeamento label → fator de repetição.
        logger: Logger para mensagens.

    Returns:
        Lista de exemplos ``(texto_da_entidade, {"entities": [(0, len, label)]})``.
    """
    if logger is None:
        logger = create_default_logger(__name__)

    df_t, df_e = _filtrar(df_textos, df_entidades, logger)
    fator = oversample or {}
    texto_por_id = df_t.set_index("id")["text"]
    exemplos: list[Example] = []

    for _, ent in df_e.iterrows():
        texto_ent = texto_por_id.loc[ent["id"]][int(ent["start_entidade"]) : int(ent["end_entidade"])]
        if not texto_ent:
            continue
        rotulo = str(ent["tp_entidade"])
        exemplo = (texto_ent, {"entities": [(0, len(texto_ent), rotulo)]})
        repeticoes = int(fator.get(rotulo, 1))
        exemplos.extend([exemplo] * repeticoes)
    return exemplos


def build_curriculum_datasets(
    df_textos: pd.DataFrame,
    df_entidades: pd.DataFrame,
    *,
    windows: tuple[str, ...] = DEFAULT_CURRICULUM_WINDOWS,
    include_pure: bool = False,
    oversample: dict[str, int] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, dict[str, list[Example]]]:
    """Constrói o dicionário de datasets por janela para curriculum learning.

    Gera, para um conjunto único nomeado ``"default"``, um dataset por janela
    solicitada: ``w0``, ``w1``, ``w2``, ``full`` e ``w00`` (entidade pura,
    quando ``include_pure=True`` ou a janela é referenciada).

    Args:
        df_textos: DataFrame com colunas ``id`` e ``text``.
        df_entidades: DataFrame com ``id``, ``start_entidade``,
            ``end_entidade`` e ``tp_entidade`` (ou formato alternativo).
        windows: Janelas a gerar (padrão ``("w0", "w1", "w2", "full")``).
        include_pure: Se True, gera também a janela ``"w00"``.
        oversample: Oversampling do ``w00`` (label → fator).
        logger: Logger para mensagens.

    Returns:
        Dict ``{"default": {janela: exemplos}}`` no formato
        ``{conjunto: {janela: [(texto, {"entities": [...]})]}}``.

    Raises:
        ValueError: Se alguma janela de ``windows`` for desconhecida.
    """
    if logger is None:
        logger = create_default_logger(__name__)

    janelas = list(dict.fromkeys(windows))
    for janela in janelas:
        if janela not in _JANELA_MAP:
            msg = f"Janela desconhecida: {janela!r}. Válidas: {sorted(_JANELA_MAP)}"
            logger.exception(msg)
            raise ValueError(msg)

    if include_pure and "w00" not in janelas:
        janelas.append("w00")

    datasets: dict[str, list[Example]] = {}
    for janela in janelas:
        if janela == "w00":
            datasets[janela] = build_pure_entity_dataset(df_textos, df_entidades, oversample=oversample, logger=logger)
        else:
            datasets[janela] = build_context_window_dataset(
                df_textos, df_entidades, window=_JANELA_MAP[janela], logger=logger
            )
        logger.info("Janela %s gerada: %d exemplos", janela, len(datasets[janela]))
    return {"default": datasets}


def _df_para_exemplos(df: pd.DataFrame) -> list[Example]:
    """Converte DataFrame de amostras (colunas ``text`` e ``entities``) em exemplos."""
    exemplos: list[Example] = []
    if "text" not in df.columns or "entities" not in df.columns:
        msg = f"DataFrame de dataset deve conter colunas 'text' e 'entities'. Encontradas: {list(df.columns)}"
        raise ValueError(msg)
    for _, row in df.iterrows():
        entidades: list[tuple[int, int, str]] = []
        for ent in row["entities"] or []:
            if isinstance(ent, list | tuple) and len(ent) >= _MIN_COMPONENTES_ENTIDADE:
                entidades.append((int(ent[0]), int(ent[1]), str(ent[2])))
        if isinstance(row["text"], str) and row["text"] and entidades:
            exemplos.append((row["text"], {"entities": entidades}))
    return exemplos


def _converter_janela(dado: object) -> list[Example]:
    """Converte o conteúdo de uma janela (DataFrame ou lista) em exemplos."""
    if dado is None:
        return []
    if isinstance(dado, pd.DataFrame):
        return _df_para_exemplos(dado)
    if isinstance(dado, list):
        exemplos: list[Example] = []
        for item in dado:
            if isinstance(item, tuple) and len(item) == _TAM_EXEMPLO:
                exemplos.append(item)
            elif isinstance(item, dict) and "text" in item and "entities" in item:
                exemplos.append((item["text"], {"entities": item["entities"]}))
        return exemplos
    msg = f"Conteúdo de janela deve ser DataFrame ou lista, recebido: {type(dado).__name__}"
    raise TypeError(msg)


def _converter_estrutura(dados: object) -> dict[str, dict[str, list[Example]]]:
    """Converte estruturas carregadas (formato do pacote ou da 942) para o padrão interno."""
    if not isinstance(dados, dict):
        msg = f"Estrutura de datasets deve ser dict, recebido: {type(dados).__name__}"
        raise TypeError(msg)

    primeiro = next(iter(dados.values()), None)
    if isinstance(primeiro, dict):
        return {
            str(conjunto): {str(janela): _converter_janela(dado) for janela, dado in janelas.items()}
            for conjunto, janelas in dados.items()
        }
    return {"default": {str(janela): _converter_janela(dado) for janela, dado in dados.items()}}


def save_curriculum_datasets(datasets: dict, path: str | Path) -> None:
    """Persiste datasets de curriculum em arquivo joblib.

    Args:
        datasets: Estrutura ``{conjunto: {janela: exemplos}}`` (formato de
            ``build_curriculum_datasets`` ou joblib da estória 942).
        path: Caminho de destino (ex.: ``"./datasets_sujo_ouro.joblib"``).

    Raises:
        RuntimeError: Se ``joblib`` não estiver instalado.
    """
    try:
        import joblib
    except ImportError as exc:
        msg = "joblib não instalado (dependência de scikit-learn). Instale com 'pip install joblib'."
        raise RuntimeError(msg) from exc

    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(datasets, str(destino))


def load_curriculum_datasets(path: str | Path) -> dict[str, dict[str, list[Example]]]:
    """Carrega datasets de curriculum de arquivo joblib.

    Aceita tanto o formato do pacote (``{conjunto: {janela: exemplos}}``)
    quanto o formato da estória 942 (DataFrames com colunas ``text`` e
    ``entities``), convertendo para o formato interno quando necessário.

    Args:
        path: Caminho do arquivo joblib.

    Returns:
        Dict ``{conjunto: {janela: [(texto, {"entities": [...]})]}}``.

    Raises:
        RuntimeError: Se ``joblib`` não estiver instalado.
        ValueError: Se a estrutura carregada for inválida.
    """
    try:
        import joblib
    except ImportError as exc:
        msg = "joblib não instalado (dependência de scikit-learn). Instale com 'pip install joblib'."
        raise RuntimeError(msg) from exc

    dados = joblib.load(str(Path(path)))
    return _converter_estrutura(dados)
