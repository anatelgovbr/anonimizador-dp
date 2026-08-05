"""Gerenciamento de dados de treinamento NER.

Este módulo fornece a classe NERDataManager para centralizar o gerenciamento
    de dados de treinamento: adição e divisão treino/validação.
"""

import logging

import pandas as pd

from anonimizar._common.logging import create_default_logger
from anonimizar._constants import DEFAULT_RANDOM_STATE, DEFAULT_TRAIN_RATIO
from anonimizar._training import data_loader

__all__ = ["NERDataManager"]


class NERDataManager:
    """Gerencia dados de treinamento para modelos NER.

    Responsável por:
    - Adicionar dados de diversas fontes (list, dict, DataFrame, JSONL)
    - Converter formatos de entrada aceitos para o formato spaCy
    - Dividir dados em treino/validação
    - Expor training_data, train_data e val_data para a facade

    Examples:
        >>> manager = NERDataManager(supported_labels=["CPF", "EMAIL"])
        >>> manager.add_data([{"text": "CPF 123", "entities": [(4, 7, "CPF")]}])
        >>> manager.split_data(train_ratio=0.8)
        >>> len(manager.training_data)
        1
    """

    def __init__(
        self,
        *,
        supported_labels: list[str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Inicializa gerenciador de dados.

        Args:
            supported_labels: Labels armazenados para consumidores da instância.
                A adição de dados não os valida; se None ou vazio, armazena uma
                lista vazia.
            logger: Logger para mensagens. Se None, cria logger padrão.
        """
        self.training_data: list[tuple[str, dict]] = []
        self.train_data: list[tuple[str, dict]] = []
        self.val_data: list[tuple[str, dict]] = []
        self.supported_labels: list[str] = supported_labels or []
        self.logger: logging.Logger = logger or create_default_logger(__name__)

    # ------------------------------------------------------------------
    # Método público principal
    # ------------------------------------------------------------------

    def add_data(
        self,
        data: list | dict | pd.DataFrame | str | None,
        *,
        errors: str = "raise",
        auto_clean: bool = False,
        strict_clean: bool = True,
        keep_empty_entities: bool = True,
        resolve_conflicts: str = "raise",
        load_jsonl_fn: object = None,
    ) -> None:
        """Adiciona dados para treinamento.

        Aceita múltiplos formatos:
        - ``list[dict]``: chaves ``text``/``entities`` ou ``labels``
        - ``dict``: mesmo esquema, único registro
        - ``pd.DataFrame``: colunas ``text``/``entities`` ou ``texto``/``start``/``end``/``entidade``
        - ``str``: caminho para arquivo ``.jsonl`` (formato Doccano)

        Args:
            data: Dados a serem adicionados.
            errors: Política de erro (``'raise'``, ``'coerce'``, ``'ignore'``).
            auto_clean: Se True, aplica o callback interno de limpeza. Esse
                caminho depende de uma função de limpeza compatível com o
                protocolo de ``data_loader.apply_auto_clean``.
            strict_clean: Encaminhado ao callback de limpeza quando
                ``auto_clean=True``.
            keep_empty_entities: Se True, mantém exemplos sem entidades após limpeza.
            resolve_conflicts: Encaminhado ao callback de limpeza quando
                ``auto_clean=True``.
            load_jsonl_fn: Função para carregar JSONL. Se None e data for str, usa io.load.

        Raises:
            TypeError: Se o tipo de dado não for suportado.
            ValueError: Se os dados forem inválidos e errors='raise'.
            FileNotFoundError: Se arquivo JSONL não existir.
        """
        if data is None:
            msg = "Dados não podem ser None"
            self.logger.exception(msg)
            raise TypeError(msg)

        # Resolver função de carregamento JSONL
        if load_jsonl_fn is None:
            from anonimizar._training.io_handler import load_from_doccano_jsonl as _load_jsonl
        else:
            _load_jsonl = load_jsonl_fn  # type: ignore[assignment]

        new_data = data_loader.convert_input_data(
            data=data,
            logger=self.logger,
            transform_pandas_fn=lambda df: data_loader.transform_data_from_pandas(df, self.logger),
            load_jsonl_fn=_load_jsonl,
            errors=errors,
        )

        if not new_data:
            self.logger.warning("Nenhum dado válido encontrado para adicionar.")
            return

        # Limpar automaticamente se solicitado
        if auto_clean:
            from anonimizar._training import data_validator

            new_data = data_loader.apply_auto_clean(
                data=new_data,
                clean_entities_fn=lambda text, ents, *, strict, resolve_conflicts: (
                    data_validator.clean_entities(text, ents, strict=strict, resolve_conflicts=resolve_conflicts)
                ),
                strict_clean=strict_clean,
                keep_empty_entities=keep_empty_entities,
                resolve_conflicts=resolve_conflicts,
                logger=self.logger,
            )

        self.training_data.extend(new_data)
        self.logger.info(
            "Dados adicionados: %d exemplos (total: %d)",
            len(new_data),
            len(self.training_data),
        )

    # ------------------------------------------------------------------
    # Divisão treino/validação
    # ------------------------------------------------------------------

    def split_data(
        self,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        *,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
        """Divide training_data em treino e validação.

        Args:
            train_ratio: Proporção de dados para treino (0.0 a 1.0).
            random_state: Seed para reprodutibilidade (não usado diretamente
                pelo split atual, mantido para compatibilidade futura).

        Returns:
            Tupla (train_data, val_data) — também persiste em self.train_data
            e self.val_data.

        Raises:
            ValueError: Se não houver dados suficientes ou train_ratio inválido.
        """
        if not self.training_data:
            msg = "Nenhum dado disponível. Use add_data() primeiro."
            self.logger.exception(msg)
            raise ValueError(msg)

        if not 0.0 < train_ratio < 1.0:
            msg = f"train_ratio deve estar entre 0.0 e 1.0 (exclusivo), recebido: {train_ratio}"
            self.logger.exception(msg)
            raise ValueError(msg)

        self.logger.debug(
            "Dividindo %d exemplos com train_ratio=%.2f (seed=%d)",
            len(self.training_data),
            train_ratio,
            random_state,
        )

        self.train_data, self.val_data = data_loader.split_data(self.training_data, train_ratio, self.logger)

        self.logger.info(
            "Divisão concluída: %d treino, %d validação",
            len(self.train_data),
            len(self.val_data),
        )

        return self.train_data, self.val_data

    # ------------------------------------------------------------------
    # Propriedades de conveniência
    # ------------------------------------------------------------------

    @property
    def n_examples(self) -> int:
        """Retorna o número total de exemplos de treinamento.

        Returns:
            Número de exemplos em training_data.
        """
        return len(self.training_data)

    @property
    def is_empty(self) -> bool:
        """Indica se não há dados carregados.

        Returns:
            True se training_data estiver vazio.
        """
        return len(self.training_data) == 0
