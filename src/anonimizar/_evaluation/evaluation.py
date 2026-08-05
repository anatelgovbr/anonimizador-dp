r"""Módulo evaluation.

Este módulo oferece **ferramentas completas de avaliação** para modelos NER
especializados em anonimização de dados sensíveis gerados pelo *Anonimizar*.

Principais capacidades

1. **Extração de predições** diretamente de um objeto `Anonimizar`.
2. **Cálculo de métricas de desempenho** (F-beta, precisão, recall) por tipo de
   entidade e consolidadas.
3. **Geração de relatórios** sumarizados ou detalhados em `pandas.DataFrame`.
4. **Análise de casos** (TP, FP, FN, TN) e distribuição de *overlap*.
5. **Comparação de versões** de relatórios para identificar regressões ou ganhos.
6. **Exportação** dos resultados em Parquet/CSV/JSON.

Classes
- `Evaluation`: classe principal que centraliza todo o workflow de
  avaliação.

Exemplos:

    >>> from anonimizar import Anonimizar, Evaluation
    >>>
    >>> # Carregar anonimizador e textos
    >>>
    >>> anonymizer = Anonimizar("pt_core_news_lg")
    >>> anonymizer.add_apply_patterns(['CPF', 'EMAIL', 'TELEFONE'], use_model_labels=True)
    >>>
    >>> evaluator = Evaluation(
    ...     texts_path="texts.parquet",
    ...     ground_truth_path="gt.parquet",
    ...     overlap_threshold=0.8,
    ...     beta=2.0,
    ... )
    >>>
    >>> # Pipeline típico
    >>>
    >>> preds   = evaluator.extract_predictions(anonymizer, save_path="preds.parquet")
    >>> report = evaluator.evaluate_model()  # métricas
    >>> details = evaluator.get_detailed_report()  # DataFrame detalhado
    >>> print(evaluator.get_summary_report())

Convenções de nomenclatura

- **Colunas obrigatórias** nos DataFrames de *ground truth*:`id`, `tp_entidade`,
  `start_entidade`, `end_entidade`.
- Labels são **normalizadas** via `entity_mapping` para garantir consistência
  entre fontes heterogêneas.

Limites conhecidos

- O cálculo de *overlap* considera apenas a **proporção de interseção** sobre a
  extensão da entidade de referência (*ground truth*).
- Métricas macro não suportam **ponderação por frequência** de entidade.

Requisitos externos

- `pandas`, `numpy`, `scikit-learn`, `tqdm`.

Métodos principais:
    - _create_default_logger: Cria logger padrão quando não fornecido.
    - load_data: Carrega dados de teste e ground truth.
    - _normalize_entities: Normaliza entidades usando mapeamento configurado.
    - extract_predictions: Extrai predições do modelo e salva opcionalmente.
    - load_predictions: Carrega predições de arquivo.
    - save_predictions: Salva predições em arquivo.
    - set_predictions: Define predições diretamente via DataFrame.
    - calculate_overlap: Calcula sobreposição entre entidades verdadeiras e preditas.
    - generate_comparison_data: Gera dados de comparação entre ground truth e predições.
    - evaluate_model: Executa avaliação completa do modelo de anonimização.
    - get_detailed_report: Gera relatório detalhado com métricas por entidade.
    - evaluate_multiple_thresholds: Avalia modelo com múltiplos parâmetros.
    - export_results: Exporta resultados para diferentes formatos.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from anonimizar._common.logging import create_default_logger
from anonimizar._constants import (
    ALL_ENTITIES_KEY,
    DEFAULT_BETA,
    DEFAULT_ENTITY_MAPPING,
    DEFAULT_OVERLAP_THRESHOLD,
    MAX_ERROR_EXAMPLES,
    OVERLAP_BIN_LABELS,
    OVERLAP_BINS,
    REMOVE_FLAG_VALUE,
    REQUIRED_PREDICTION_COLUMNS,
)
from anonimizar._evaluation import comparison, metrics, predictor
from anonimizar._evaluation import data_loader as eval_data_loader
from anonimizar._evaluation import reporter as _reporter
from anonimizar._normalization import normalize_entity

if TYPE_CHECKING:
    from .._anonymization.anonymizer import Anonimizar


class Evaluation:
    """Classe para avaliação de modelos NER especializados em anonimização de dados sensíveis.

    Esta classe oferece funcionalidades completas para avaliar modelos de reconhecimento
    de entidades nomeadas (NER), com foco em identificação de dados pessoais e sensíveis
    para anonimização.

    Attributes:
        texts_path (Path | None): Caminho para arquivo de textos
        ground_truth_path (Path | None): Caminho para arquivo de ground truth
        overlap_threshold (float): Threshold mínimo de overlap para consideração de match
        beta (float): Valor beta para cálculo de F-beta score
        entity_mapping (dict[str, str]): Mapeamento de normalização de entidades
        logger (logging.Logger): Logger para registro de atividades
        df_texts (pd.DataFrame | None): DataFrame com textos para avaliação
        df_ground_truth (pd.DataFrame | None): DataFrame com ground truth
        df_predictions (pd.DataFrame | None): DataFrame com predições do modelo
        evaluation_results (dict): Resultados da última avaliação executada

    Example:
        >>> evaluator = Evaluation(
        ...     texts_path="texts.parquet",
        ...     ground_truth_path="gt.parquet",
        ...     overlap_threshold=0.8,
        ...     beta=2.0
        ... )
        >>> predictions = evaluator.extract_predictions(anonymizer)
        >>> results = evaluator.evaluate_model()
        >>> print(evaluator.get_summary_report())
    """

    def __init__(
        self,
        texts_path: str | None = None,
        ground_truth_path: str | None = None,
        *,
        entity_mapping: dict[str, str] | None = None,
        overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
        beta: float = DEFAULT_BETA,
        normalize_entities: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        """Inicializa o avaliador com configurações personalizadas.

        Args:
            texts_path (str | None, optional): Caminho para arquivo de textos.
                Defaults to None.
            ground_truth_path (str | None, optional): Caminho para ground truth.
                Defaults to None.
            entity_mapping (dict[str, str] | None, optional): Mapeamento de entidades.
                Defaults to None.
            overlap_threshold (float, optional): Threshold de overlap entre 0.0 e 1.0.
                Defaults to 0.8.
            beta (float, optional): Beta para F-beta score, deve ser > 0.
                Defaults to 2.0.
            normalize_entities (bool): Se True (padrão), normaliza entidades do ground
                truth e das predições removendo prefixos/sufixos textuais antes da
                comparação.
            logger (logging.Logger | None, optional): Logger personalizado.
                Defaults to None.

        Raises:
            ValueError: Se overlap_threshold não estiver entre 0.0 e 1.0.
            ValueError: Se beta for menor ou igual a 0.

        Examples:
            >>> evaluator = Evaluation(
            ...     texts_path="texts.parquet",
            ...     ground_truth_path="gt.parquet",
            ...     overlap_threshold=0.8,
            ...     beta=2.0
            ... )
        """
        self.logger = logger or create_default_logger(__name__)
        self.logger.info("Inicializando Evaluation")

        if not 0.0 <= overlap_threshold <= 1.0:
            msg = f"overlap_threshold deve estar entre 0.0 e 1.0, recebido: {overlap_threshold}"
            self.logger.error(msg)
            raise ValueError(msg)

        if beta <= 0:
            msg = f"beta deve ser maior que 0, recebido: {beta}"
            self.logger.error(msg)
            raise ValueError(msg)

        self.texts_path = Path(texts_path) if texts_path else None
        self.ground_truth_path = Path(ground_truth_path) if ground_truth_path else None
        self.overlap_threshold = overlap_threshold
        self.beta = beta
        self.normalize_entities = normalize_entities

        self.entity_mapping = entity_mapping or dict(DEFAULT_ENTITY_MAPPING)

        self.df_texts: pd.DataFrame | None = None
        self.df_ground_truth: pd.DataFrame | None = None
        self.df_predictions: pd.DataFrame | None = None
        self.evaluation_results: dict = {}

        if self.texts_path and self.ground_truth_path:
            self.logger.info("Carregando dados de arquivos: texts=%s gt=%s", self.texts_path, self.ground_truth_path)
            self.load_data_from_files()
        else:
            self.logger.warning("Inicialização sem arquivos: dados devem ser carregados via load_data() ou set_*()")

        self.logger.debug("Configuração: overlap_threshold=%.3f, beta=%.3f", overlap_threshold, beta)
        self.logger.debug("Mapeamento de entidades configurado (%d chaves)", len(self.entity_mapping))

    def load_data_from_files(self) -> None:
        """Carrega dados de teste e ground truth dos arquivos especificados."""
        # Check file formats before calling module
        if self.texts_path and self.texts_path.suffix not in (".parquet", ".csv"):
            msg = f"Formato não suportado: {self.texts_path.suffix}"
            self.logger.error(msg)
            raise ValueError(msg)

        if self.ground_truth_path and self.ground_truth_path.suffix not in (".parquet", ".csv"):
            msg = f"Formato não suportado: {self.ground_truth_path.suffix}"
            self.logger.error(msg)
            raise ValueError(msg)

        self.df_texts, self.df_ground_truth = eval_data_loader.load_data_from_files(
            texts_path=self.texts_path, ground_truth_path=self.ground_truth_path, logger=self.logger
        )

        # Apply entity-specific post-processing
        if "remove" in self.df_ground_truth.columns:
            initial_count = len(self.df_ground_truth)
            self.df_ground_truth = self.df_ground_truth[self.df_ground_truth["remove"] != REMOVE_FLAG_VALUE]
            removed_count = initial_count - len(self.df_ground_truth)
            if removed_count > 0:
                self.logger.info("Removidas %d entidades marcadas para remoção", removed_count)

        self._normalize_entities()
        self.logger.info("Dados carregados: %d textos, %d entidades", len(self.df_texts), len(self.df_ground_truth))

    def _normalize_entities(self) -> None:
        """Normaliza labels e spans do ground truth."""
        if self.df_ground_truth is None:
            return

        # Normalizar labels via entity_mapping
        original_count = len(self.df_ground_truth["tp_entidade"].unique())
        self.df_ground_truth["tp_entidade"] = self.df_ground_truth["tp_entidade"].apply(
            lambda x: self.entity_mapping.get(x, x)
        )
        final_count = len(self.df_ground_truth["tp_entidade"].unique())
        self.logger.debug("Entidades normalizadas: %d -> %d tipos únicos", original_count, final_count)

        # Normalizar spans (texto/offsets) se habilitado
        if self.normalize_entities and "text_entidade" in self.df_ground_truth.columns:
            self._normalize_df_spans(self.df_ground_truth)

    def _normalize_df_spans(self, df: pd.DataFrame) -> None:
        """Normaliza texto e offsets de entidades em um DataFrame.

        Aplica normalize_entity a cada linha do DataFrame, ajustando
        text_entidade, start_entidade e end_entidade in-place.

        Args:
            df: DataFrame com colunas text_entidade, start_entidade, end_entidade, tp_entidade.
        """
        if not self.normalize_entities:
            return

        updated = 0
        for idx, row in df.iterrows():
            text = row.get("text_entidade", "")
            start = row.get("start_entidade", 0)
            end = row.get("end_entidade", len(text))
            label = row.get("tp_entidade", "")

            cleaned, new_start, new_end, _rule = normalize_entity(text, start, end, label)
            if new_start != start or new_end != end:
                df.loc[idx, ["text_entidade", "start_entidade", "end_entidade"]] = (
                    cleaned,
                    new_start,
                    new_end,
                )
                updated += 1

        if updated:
            self.logger.debug("Spans normalizados: %d entidades atualizadas", updated)

    def load_data(self, df_texts: pd.DataFrame, df_ground_truth: pd.DataFrame) -> None:
        """Carrega dados diretamente de DataFrames.

        Args:
            df_texts (pd.DataFrame): DataFrame com textos
            df_ground_truth (pd.DataFrame): DataFrame com ground truth
        """
        self.df_texts, self.df_ground_truth = eval_data_loader.load_data(
            df_texts=df_texts, df_ground_truth=df_ground_truth, logger=self.logger
        )

        # Apply entity-specific post-processing
        if "remove" in self.df_ground_truth.columns:
            initial_count = len(self.df_ground_truth)
            self.df_ground_truth = self.df_ground_truth[self.df_ground_truth["remove"] != REMOVE_FLAG_VALUE]
            removed_count = initial_count - len(self.df_ground_truth)
            if removed_count > 0:
                self.logger.info("Removidas %d entidades marcadas para remoção", removed_count)

        self._normalize_entities()
        self.logger.debug(
            "Dados carregados (em memória): %d textos, %d entidades", len(self.df_texts), len(self.df_ground_truth)
        )

    def extract_predictions(
        self, anonymizer: "Anonimizar", save_path: str | None = None, *, force_recompute: bool = False
    ) -> pd.DataFrame:
        """Extrai predições do modelo e opcionalmente salva em arquivo.

        Args:
            anonymizer (Anonimizar): Instância configurada para extrair entidades.
            save_path (str | None, optional): Caminho para salvar predições. Defaults to None.
            force_recompute (bool, optional): Se True, recomputa predições. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame com predições normalizadas.

        Raises:
            ValueError: Se os textos de avaliação não tiverem sido carregados.
        """
        if self.df_predictions is not None and not force_recompute:
            self.logger.info("Usando predições já extraídas. Use force_recompute=True para recomputar.")
            return self.df_predictions

        if self.df_texts is None:
            msg = "Dados de texto não carregados. Execute load_data() primeiro."
            self.logger.error(msg)
            raise ValueError(msg)

        self.df_predictions = predictor.extract_predictions(
            df_texts=self.df_texts, anonymizer=anonymizer, entity_mapping=self.entity_mapping, logger=self.logger
        )
        self._normalize_df_spans(self.df_predictions)

        if save_path:
            self.save_predictions(save_path)

        return self.df_predictions

    def load_predictions(self, predictions_path: str | Path) -> pd.DataFrame:
        """Carrega predições de arquivo.

        Args:
            predictions_path (str | Path): Caminho para arquivo de predições

        Returns:
            pd.DataFrame: DataFrame com predições carregadas
        """
        predictions_path = Path(predictions_path)

        # Handle JSON format separately as module doesn't support it
        if predictions_path.suffix == ".json":
            if not predictions_path.exists():
                msg = f"Arquivo de predições não encontrado: {predictions_path}"
                self.logger.error(msg)
                raise FileNotFoundError(msg)

            self.logger.debug("Carregando predições de: %s", predictions_path)
            self.df_predictions = pd.read_json(predictions_path)
        elif predictions_path.suffix in (".parquet", ".csv"):
            self.df_predictions = predictor.load_predictions(predictions_path=predictions_path, logger=self.logger)
        else:
            msg = f"Formato não suportado: {predictions_path.suffix}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Apply entity mapping after loading
        required_cols = REQUIRED_PREDICTION_COLUMNS
        missing_cols = set(required_cols) - set(self.df_predictions.columns)
        if missing_cols:
            msg = f"Colunas ausentes nas predições: {missing_cols}"
            self.logger.error(msg)
            raise ValueError(msg)

        self.df_predictions["tp_entidade"] = self.df_predictions["tp_entidade"].apply(
            lambda x: self.entity_mapping.get(x, x)
        )
        self._normalize_df_spans(self.df_predictions)

        return self.df_predictions

    def save_predictions(self, output_path: str | Path, format: str = "parquet") -> None:  # noqa: A002
        """Salva predições em arquivo.

        Args:
            output_path (str | Path): Caminho para salvar. Para ``csv`` e
                ``parquet``, a extensão é alterada para o formato escolhido.
                Para ``json``, o caminho é usado como informado.
            format (str, optional): Um de ``"parquet"``, ``"csv"`` ou ``"json"``.
                JSON é exportado como lista de registros e converte ``detected_by``
                para string. Defaults to ``"parquet"``.

        Raises:
            ValueError: Se não houver predições ou o formato não for suportado.
        """
        if self.df_predictions is None:
            msg = "Nenhuma predição para salvar. Execute extract_predictions() primeiro."
            self.logger.error(msg)
            raise ValueError(msg)

        output_path = Path(output_path)

        # Handle JSON format separately as module doesn't support it
        if format == "json":
            self.logger.info("Salvando predições em %s (formato=json)", output_path)
            df_to_save = self.df_predictions.copy()
            if "detected_by" in df_to_save.columns:
                df_to_save["detected_by"] = df_to_save["detected_by"].astype("str")
            df_to_save.to_json(output_path, orient="records", indent=2, force_ascii=False)
            self.logger.info("Predições salvas em: %s", output_path)
            return

        # Handle format parameter by adjusting output_path extension if needed
        if format == "csv" and output_path.suffix != ".csv":
            output_path = output_path.with_suffix(".csv")
        elif format == "parquet" and output_path.suffix != ".parquet":
            output_path = output_path.with_suffix(".parquet")
        elif format not in ("parquet", "csv"):
            msg = f"Formato não suportado: {format}"
            self.logger.error(msg)
            raise ValueError(msg)

        predictor.save_predictions(df_predictions=self.df_predictions, save_path=output_path, logger=self.logger)

    def set_predictions(self, predictions: pd.DataFrame) -> None:
        """Define predições diretamente via DataFrame.

        Args:
            predictions (pd.DataFrame): DataFrame com predições
        """
        self.df_predictions = eval_data_loader.set_predictions(predictions=predictions, logger=self.logger)

        # Apply label mapping and span normalization
        self.df_predictions["tp_entidade"] = self.df_predictions["tp_entidade"].apply(
            lambda x: self.entity_mapping.get(x, x)
        )
        self._normalize_df_spans(self.df_predictions)

        self.logger.info("Predições definidas: %d entidades", len(self.df_predictions))

    def calculate_overlap(self, start_true: int, end_true: int, start_pred: int, end_pred: int) -> float:
        """Calcula a proporção de sobreposição entre entidade verdadeira e predita.

        Delega para `_evaluation.comparison.calculate_overlap()` (IoU) após
        validações de NaN.

        Args:
            start_true (int): Posição inicial da entidade verdadeira
            end_true (int): Posição final da entidade verdadeira
            start_pred (int): Posição inicial da entidade predita
            end_pred (int): Posição final da entidade predita

        Returns:
            float: Taxa de overlap entre 0.0 e 1.0

        Examples:
            >>> evaluator = Evaluation()
            >>> evaluator.calculate_overlap(0, 10, 5, 15)
            0.333
        """
        if pd.isna(start_pred) or pd.isna(end_pred):
            return 0.0

        if pd.isna(start_true) or pd.isna(end_true):
            return 0.0

        start_true = int(start_true)
        end_true = int(end_true)
        start_pred = int(start_pred)
        end_pred = int(end_pred)

        return comparison.calculate_overlap(start_true, end_true, start_pred, end_pred)

    def generate_comparison_data(
        self, predictions: pd.DataFrame, overlap_threshold: float | None = None
    ) -> pd.DataFrame:
        """Gera dados de comparação entre ground truth e predições.

        Args:
            predictions (pd.DataFrame): DataFrame com predições
            overlap_threshold (float | None, optional): Threshold de overlap. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame com comparação
        """
        if overlap_threshold is None:
            overlap_threshold = self.overlap_threshold

        return comparison.generate_comparison_data(
            df_ground_truth=self.df_ground_truth,
            predictions=predictions,
            entity_mapping=self.entity_mapping,
            overlap_threshold=overlap_threshold,
            logger=self.logger,
        )

    def evaluate_model(
        self,
        anonymizer: "Anonimizar | None" = None,
        predictions: pd.DataFrame | None = None,
        overlap_threshold: float | None = None,
        beta: float | None = None,
        entity_types: list[str] | None = None,
    ) -> dict:
        """Executa avaliação completa do modelo de anonimização.

        Args:
            anonymizer (Anonimizar | None): Modelo para extração; obrigatório
                quando não houver ``predictions`` nem predições previamente extraídas.
            predictions (pd.DataFrame | None, optional): DataFrame com predições. Defaults to None.
            overlap_threshold (float | None, optional): Threshold de overlap. Defaults to None.
            beta (float | None, optional): Beta para F-score. Defaults to None.
            entity_types (list[str] | None, optional): Tipos de entidade para avaliar. Defaults to None.

        Returns:
            dict: Métricas por entidade e para ``ALL_ENTITIES_KEY``. Cada entrada
                contém, no mínimo, ``fbeta``, ``precision``, ``recall``, ``tp``,
                ``fp`` e ``fn``.

        Raises:
            ValueError: Se o ground truth não estiver carregado ou não houver
                anonimizador/predições disponíveis.
            RuntimeError: Se a geração do relatório falhar.
        """
        if self.df_ground_truth is None:
            msg = "Dados de ground truth não carregados. Execute load_data() primeiro."
            self.logger.error(msg)
            raise ValueError(msg)

        if overlap_threshold is None:
            overlap_threshold = self.overlap_threshold

        if beta is None:
            beta = self.beta

        if entity_types is None:
            entity_types = self.df_ground_truth["tp_entidade"].unique().tolist()

        if predictions is not None:
            self.set_predictions(predictions)
        elif self.df_predictions is None:
            if anonymizer is None:
                msg = "Forneça 'anonymizer' para extração ou 'predictions' já extraídas."
                self.logger.error(msg)
                raise ValueError(msg)
            self.extract_predictions(anonymizer)

        self.logger.debug(
            "Iniciando avaliação com %d ground truth, overlap=%.3f, beta=%.3f",
            len(self.df_ground_truth),
            overlap_threshold,
            beta,
        )
        self.logger.debug("Tipos de entidade: %s", entity_types)

        try:
            results = self._generate_evaluation_report(self.df_predictions, overlap_threshold, beta, entity_types)
        except Exception as e:
            msg = f"Erro na geração do relatório: {e}"
            self.logger.exception(msg)
            raise RuntimeError(msg) from e

        self.evaluation_results = results

        self.logger.info("Avaliação concluída com sucesso")
        return results

    def _generate_evaluation_report(
        self, predictions: pd.DataFrame, overlap_threshold: float, beta: float, entity_types: list[str]
    ) -> dict:
        """Gera relatório de avaliação com métricas detalhadas.

        Args:
            predictions (pd.DataFrame): DataFrame com predições
            overlap_threshold (float): Threshold de overlap
            beta (float): Beta para F-score
            entity_types (list[str]): Tipos de entidade para avaliar

        Returns:
            dict: Dicionário com métricas por tipo de entidade
        """
        # Generate comparison data first
        comparison_data = self.generate_comparison_data(predictions, overlap_threshold)

        return metrics.generate_evaluation_report(
            df_ground_truth=self.df_ground_truth,
            predictions=predictions,
            comparison_data=comparison_data,
            overlap_threshold=overlap_threshold,
            beta=beta,
            entity_types=entity_types,
            logger=self.logger,
            generate_comparison_fn=self.generate_comparison_data,
        )

    def get_summary_report(self) -> str:
        """Retorna relatório resumido das últimas métricas calculadas."""
        return metrics.format_summary_report(self.evaluation_results)

    def get_detailed_report(self) -> pd.DataFrame:
        """Retorna relatório detalhado em formato DataFrame."""
        return metrics.get_detailed_report(self.evaluation_results)

    def evaluate_multiple_thresholds(
        self,
        anonymizer: None = None,
        predictions: pd.DataFrame | None = None,
        overlap_thresholds: list[float] | None = None,
        beta_values: list[float] | None = None,
    ) -> pd.DataFrame:
        """Avalia modelo com múltiplos thresholds e valores de beta.

        Delega para `_evaluation.metrics.evaluate_multiple_thresholds()`.

        Args:
            anonymizer: Modelo para extração (opcional se predictions fornecidas)
            predictions (pd.DataFrame | None, optional): DataFrame com predições. Defaults to None.
            overlap_thresholds (list[float], optional): Lista de thresholds. Defaults to [0.7, 0.8, 0.9].
            beta_values (list[float], optional): Lista de valores beta. Defaults to [1, 2].

        Returns:
            pd.DataFrame: DataFrame com resultados para cada combinação de threshold e beta.
        """
        if self.df_ground_truth is None:
            msg = "Dados não carregados. Execute load_data() primeiro."
            self.logger.error(msg)
            raise ValueError(msg)

        if predictions is not None:
            self.set_predictions(predictions)
        elif self.df_predictions is None:
            if anonymizer is None:
                msg = "Forneça 'anonymizer' para extração ou 'predictions' já extraídas."
                self.logger.error(msg)
                raise ValueError(msg)
            self.extract_predictions(anonymizer)

        thresholds_to_use = list(overlap_thresholds) if overlap_thresholds is not None else [0.7, 0.8, 0.9]
        betas_to_use = list(beta_values) if beta_values is not None else [1, 2]

        result_df = metrics.evaluate_multiple_thresholds(
            predictions=self.df_predictions,
            df_ground_truth=self.df_ground_truth,
            generate_comparison_fn=self.generate_comparison_data,
            overlap_thresholds=overlap_thresholds,
            beta_values=beta_values,
            logger=self.logger,
        )

        entity_types = self.df_ground_truth["tp_entidade"].unique().tolist()
        self.evaluation_results = self._generate_evaluation_report(
            self.df_predictions, thresholds_to_use[0], betas_to_use[0], entity_types
        )

        return result_df

    def export_results(self, output_path: str, format: str = "parquet") -> None:  # noqa: A002
        """Exporta resultados da avaliação para arquivo.

        Delega para `_evaluation.reporter.export_dataframe()`.

        Args:
            output_path (str): Caminho para salvar o arquivo
            format (str, optional): Formato do arquivo. Defaults to 'parquet'.
        """
        if not self.evaluation_results:
            msg = "Nenhum resultado para exportar. Execute evaluate_model() primeiro."
            self.logger.error(msg)
            raise ValueError(msg)

        detailed_df = self.get_detailed_report()

        _reporter.export_dataframe(
            df=detailed_df,
            output_path=output_path,
            format=format,
            logger=self.logger,
        )

        self.logger.debug("Resultados exportados para: %s", output_path)

    def compare_reports(self, current_report: pd.DataFrame, previous_report: pd.DataFrame) -> pd.DataFrame:
        """Compara dois relatórios de métricas e identifica melhorias/pioras.

        Delega para `_evaluation.comparison.compare_reports()`.

        Args:
            current_report (pd.DataFrame): Relatório atual com métricas
            previous_report (pd.DataFrame): Relatório anterior para comparação

        Returns:
            pd.DataFrame: DataFrame com comparação entre relatórios

        Raises:
            ValueError: Se os DataFrames não possuem colunas necessárias

        Example:
            >>> current = evaluator.get_detailed_report()
            >>> previous = pd.read_parquet("metrics_v1.parquet")
            >>> comparison = evaluator.compare_reports(current, previous)
            >>> print(comparison[comparison['fbeta_melhorou'] == True])
        """
        return comparison.compare_reports(
            current_report=current_report,
            previous_report=previous_report,
            logger=self.logger,
        )

    def get_classification_cases(
        self, entity_type: str | None = None, case_type: str = "all", overlap_threshold: float | None = None
    ) -> pd.DataFrame:
        """Extrai casos individuais de TP, FP, TN, FN para análise detalhada.

        Delega para `_evaluation.comparison.get_classification_cases()`.

        Args:
            entity_type (str | None, optional): Tipo específico de entidade.
                Se None, considera todas. Defaults to None.
            case_type (str, optional): Tipo de caso ('tp', 'fp', 'tn', 'fn', 'all').
                Defaults to 'all'.
            overlap_threshold (float | None, optional): Threshold de overlap.
                Se None, usa o padrão da classe. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame com casos individuais e suas classificações

        Raises:
            ValueError: Se predições não foram extraídas ou case_type inválido

        Example:
            >>> # Todos os falsos positivos
            >>> fp_cases = evaluator.get_classification_cases(case_type='fp')
            >>>
            >>> # Falsos negativos apenas para CPF
            >>> fn_cpf = evaluator.get_classification_cases(entity_type='CPF', case_type='fn')
            >>>
            >>> # Todos os casos para EMAIL
            >>> email_cases = evaluator.get_classification_cases(entity_type='EMAIL')
        """
        if self.df_predictions is None or self.df_ground_truth is None:
            msg = "Predições e ground truth devem estar carregados. Execute extract_predictions() primeiro."
            self.logger.error(msg)
            raise ValueError(msg)

        if overlap_threshold is None:
            overlap_threshold = self.overlap_threshold

        comparison_data = self.generate_comparison_data(self.df_predictions, overlap_threshold)

        return comparison.get_classification_cases(
            comparison_data=comparison_data,
            entity_type=entity_type,
            case_type=case_type,
            all_entities_key=ALL_ENTITIES_KEY,
            logger=self.logger,
        )

    def get_error_analysis(self, entity_type: str | None = None) -> dict:
        """Gera análise detalhada de erros por tipo de entidade.

        Delega para `_evaluation.comparison.get_error_analysis()`.

        Args:
            entity_type (str | None, optional): Tipo específico de entidade.
                Se None, analisa todas. Defaults to None.

        Returns:
            dict: Dicionário com estatísticas e exemplos de erros

        Example:
            >>> # Análise de erros para todas as entidades
            >>> error_analysis = evaluator.get_error_analysis()
            >>> print(error_analysis['summary'])
            >>>
            >>> # Análise específica para CPF
            >>> cpf_errors = evaluator.get_error_analysis('CPF')
            >>> print(cpf_errors['fp_examples'][:5])  # Primeiros 5 FP
        """
        if self.df_predictions is None or self.df_ground_truth is None:
            msg = "Dados não carregados. Execute extract_predictions() primeiro."
            self.logger.error(msg)
            raise ValueError(msg)

        comparison_data = self.generate_comparison_data(self.df_predictions, self.overlap_threshold)

        return comparison.get_error_analysis(
            comparison_data=comparison_data,
            entity_type=entity_type,
            max_examples=MAX_ERROR_EXAMPLES,
            overlap_bins=list(OVERLAP_BINS),
            overlap_bin_labels=list(OVERLAP_BIN_LABELS),
            all_entities_key=ALL_ENTITIES_KEY,
            logger=self.logger,
        )

    def save_classification_cases(
        self,
        output_path: str,
        entity_type: str | None = None,
        case_type: str = "all",
        format: str = "parquet",  # noqa: A002
    ) -> None:
        """Salva casos de classificação em arquivo para análise externa.

        Delega para `_evaluation.reporter.save_classification_cases()`.

        Args:
            output_path (str): Caminho para salvar o arquivo
            entity_type (str | None, optional): Tipo de entidade. Defaults to None.
            case_type (str, optional): Tipo de caso. Defaults to 'all'.
            format (str, optional): Formato do arquivo. Defaults to 'parquet'.

        Example:
            >>> # Salvar todos os falsos positivos
            >>> evaluator.save_classification_cases(
            ...     "fp_cases.parquet",
            ...     case_type='fp'
            ... )
            >>>
            >>> # Salvar falsos negativos de EMAIL
            >>> evaluator.save_classification_cases(
            ...     "email_fn.csv",
            ...     entity_type='EMAIL',
            ...     case_type='fn',
            ...     format='csv'
            ... )
        """
        if self.df_predictions is None or self.df_ground_truth is None:
            msg = "Predições e ground truth devem estar carregados. Execute extract_predictions() primeiro."
            self.logger.error(msg)
            raise ValueError(msg)

        comparison_data = self.generate_comparison_data(self.df_predictions, self.overlap_threshold)

        _reporter.save_classification_cases(
            comparison_data=comparison_data,
            output_path=output_path,
            entity_type=entity_type,
            case_type=case_type,
            format=format,
            logger=self.logger,
        )
