"""Gerenciamento de cross-validation para modelos NER.

Este módulo fornece a classe NERCrossValidator para centralizar
a lógica de cross-validation, separando-a da facade Trainer.
"""

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from anonimizar._common.logging import create_default_logger
from anonimizar._constants import (
    ALL_ENTITIES_KEY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BETA,
    DEFAULT_DROP,
    DEFAULT_N_ITER,
    DEFAULT_N_JOBS,
    DEFAULT_N_SPLITS,
    DEFAULT_OVERLAP_THRESHOLD,
    DEFAULT_RANDOM_STATE,
)
from anonimizar._training import cross_validation as _cv
from anonimizar._training import io as _training_io

__all__ = ["NERCrossValidator"]


class NERCrossValidator:
    """Gerencia cross-validation de modelos NER.

    Encapsula toda a lógica de cross-validation extraída da facade
    Trainer, incluindo criação de folds, treinamento
    por fold, avaliação e agregação de resultados.

    Example:
        O uso requer DataFrames de entidades e textos, uma factory de trainer e
        um diretório de saída. O trecho abaixo é pseudocódigo; defina esses
        valores conforme a aplicação.

        ```python
        cv = NERCrossValidator()
        results = cv.run(
            df_entidades=df_entidades,
            df_textos=df_textos,
            trainer_factory=trainer_factory,
            output_dir="resultados_cv",
            n_splits=5,
        )
        ```
    """

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Inicializa o cross-validator.

        Args:
            logger: Logger para mensagens. Se None, usa logger padrão do módulo.
        """
        self.logger: logging.Logger = logger or create_default_logger(__name__)

    # ------------------------------------------------------------------
    # Métodos públicos de criação de folds
    # ------------------------------------------------------------------

    def make_folds_by_id(
        self,
        df_entidades: pd.DataFrame,
        n_splits: int = DEFAULT_N_SPLITS,
        random_state: int = DEFAULT_RANDOM_STATE,
        *,
        shuffle: bool = True,
    ) -> list[tuple[list, list]]:
        """Cria folds simples (não estratificados) baseados em IDs de documento.

        Args:
            df_entidades: DataFrame com coluna 'id' ou 'id_doc' identificando documentos.
            n_splits: Número de folds.
            random_state: Seed para reprodutibilidade.
            shuffle: Se deve embaralhar antes de dividir.

        Returns:
            Lista de tuplas (ids_train, ids_val) para cada fold.
        """
        return _cv.make_folds_by_id(
            df_entidades=df_entidades,
            n_splits=n_splits,
            random_state=random_state,
            shuffle=shuffle,
            logger=self.logger,
        )

    def make_stratified_folds_by_id(
        self,
        df_entidades: pd.DataFrame,
        features: list[str],
        n_splits: int = DEFAULT_N_SPLITS,
        random_state: int = DEFAULT_RANDOM_STATE,
        *,
        shuffle: bool = True,
    ) -> list[tuple[list, list]]:
        """Cria folds estratificados balanceando distribuição de entidades.

        Args:
            df_entidades: DataFrame com entidades anotadas.
            features: Lista de tipos de entidades para estratificação.
            n_splits: Número de folds.
            random_state: Seed para reprodutibilidade.
            shuffle: Se deve embaralhar antes de dividir.

        Returns:
            Lista de tuplas (ids_train, ids_val) para cada fold.
        """
        return _cv.make_stratified_folds_by_id(
            df_entidades=df_entidades,
            features=features,
            n_splits=n_splits,
            random_state=random_state,
            shuffle=shuffle,
            logger=self.logger,
        )

    # ------------------------------------------------------------------
    # Método principal: execução do cross-validation
    # ------------------------------------------------------------------

    def run(  # noqa: PLR0915, C901
        self,
        df_entidades: pd.DataFrame | str,
        df_textos: pd.DataFrame | str | None,
        trainer_factory: Any,  # noqa: ANN401
        *,
        n_splits: int = DEFAULT_N_SPLITS,
        features: list[str] | None = None,
        output_dir: str | Path | None = None,
        n_jobs: int = DEFAULT_N_JOBS,
        random_state: int = DEFAULT_RANDOM_STATE,
        train_params: dict[str, Any] | None = None,
        eval_params: dict[str, Any] | None = None,
        add_data_params: dict[str, Any] | None = None,
        supported_labels: list[str] | None = None,
        model_name: str | None = None,
        output_path_base: Path | None = None,
        stratified: bool = True,
        replace: bool = False,
        holdout_test_size: float | None = None,
        holdout_stratify: bool = True,
    ) -> tuple[pd.DataFrame, list, list, Any]:
        """Executa cross-validation completo do modelo NER.

        Args:
            df_entidades: DataFrame com entidades anotadas OU caminho para arquivo .jsonl.
            df_textos: DataFrame com textos OU caminho para arquivo .jsonl.
                Se None e df_entidades for str, assume que é o mesmo arquivo JSONL.
            trainer_factory: Callable que cria instância de Trainer,
                recebendo (model_name, output_dir, labels, logger) como kwargs.
            n_splits: Número de folds.
            features: Tipos de entidades para estratificação.
            output_dir: Diretório para salvar resultados.
            n_jobs: Número de processos paralelos (1 = sequencial).
            random_state: Seed para reprodutibilidade.
            train_params: Parâmetros para treinamento (n_iter, drop, batch_size).
            eval_params: Parâmetros para avaliação (overlap_threshold, beta).
            add_data_params: Parâmetros para add_data (errors, auto_clean, etc.).
            supported_labels: Labels suportados pelo modelo.
            model_name: Nome do modelo spaCy base.
            output_path_base: Path base para saída (usado se output_dir for None).
            stratified: Se deve usar estratificação por tipo de entidade.
            replace: Se True, substitui diretório existente.
            holdout_test_size: Fração dos dados para separar como test fixo.
            holdout_stratify: Se True, estratifica o holdout test.

        Returns:
            Tupla (all_reports, summary_metrics, fold_results, holdout_results).

        Raises:
            ValueError: Se dados inválidos ou parâmetros incorretos.
            FileExistsError: Se diretório de saída já existir e replace=False.
        """
        import shutil

        train_params = train_params or {}
        eval_params = eval_params or {}
        add_data_params = add_data_params or {}
        features = features or supported_labels or []

        # Carregar dados de entrada (DataFrame ou arquivo JSONL)
        df_textos, df_entidades = _training_io.load_cv_input_data(df_entidades, df_textos, self.logger)

        output_path = Path(output_dir) if output_dir else output_path_base
        if output_path is None:
            msg = "output_dir ou output_path_base deve ser fornecido"
            raise ValueError(msg)

        if output_path.exists() and replace:
            shutil.rmtree(output_path)

        if output_path.exists() and not replace:
            msg = "ja existe esse diretorio"
            self.logger.exception(msg)
            raise FileExistsError(msg)

        output_path.mkdir(parents=True, exist_ok=True)

        # Separar holdout test se solicitado
        df_holdout_texts = None
        df_holdout_gt = None

        if holdout_test_size is not None:
            _holdout_test_ids, _cv_ids, df_holdout_texts, df_holdout_gt, df_entidades, df_textos = (
                _cv.separate_holdout_test(
                    df_entidades=df_entidades,
                    df_textos=df_textos,
                    holdout_test_size=holdout_test_size,
                    features=features,
                    random_state=random_state,
                    output_path=output_path,
                    logger=self.logger,
                    holdout_stratify=holdout_stratify,
                    make_stratified_folds_fn=self.make_stratified_folds_by_id,
                    make_folds_fn=self.make_folds_by_id,
                )
            )

        cleaned_training_data, id_to_index = _cv.prepare_cv_data(
            df_entidades=df_entidades,
            df_textos=df_textos,
            model_name=model_name or "",
            supported_labels=supported_labels or [],
            add_data_params=add_data_params,
            logger=self.logger,
        )

        # Criar folds
        if stratified:
            if not features:
                msg = "É necessária informar quais as colunas para a estratificação."
                self.logger.exception(msg)
                raise ValueError(msg)
            self.logger.debug("Criando folds estratificados...")
            folds = self.make_stratified_folds_by_id(
                df_entidades=df_entidades,
                features=features,
                n_splits=n_splits,
                random_state=random_state,
            )
        else:
            self.logger.debug("Criando folds simples...")
            folds = self.make_folds_by_id(
                df_entidades=df_entidades,
                n_splits=n_splits,
                random_state=random_state,
            )

        self.logger.info("Iniciando cross validation com %d folds", n_splits)

        def run_single_fold(
            fold_idx: int,
            ids_train: list,
            ids_val: list,
        ) -> tuple[pd.DataFrame, dict | None, dict | None]:
            """Executa um único fold do cross validation.

            Args:
                fold_idx: Índice do fold (1-based).
                ids_train: IDs de treino do fold.
                ids_val: IDs de validação do fold.

            Returns:
                Tupla (report, summary, holdout_summary).
            """
            fold_training_data, df_texts_val, df_gt_val = _cv.prepare_fold_data(
                fold_idx=fold_idx,
                ids_train=ids_train,
                ids_val=ids_val,
                id_to_index=id_to_index,
                cleaned_training_data=cleaned_training_data,
                df_entidades=df_entidades,
                df_textos=df_textos,
                logger=self.logger,
            )

            fold_dir = output_path / f"fold_{fold_idx}"
            fold_dir.mkdir(exist_ok=True)

            _cv.save_fold_ids(
                fold_idx=fold_idx,
                ids_train=ids_train,
                ids_val=ids_val,
                fold_training_data=fold_training_data,
                df_texts_val=df_texts_val,
                fold_dir=fold_dir,
                logger=self.logger,
            )

            fold_trainer = trainer_factory(
                model_name=model_name,
                output_dir=str(fold_dir / "model"),
                labels=supported_labels,
                logger=self.logger,
            )

            fold_trainer.training_data = fold_training_data

            fold_trainer.train(
                n_iter=train_params.get("n_iter", DEFAULT_N_ITER),
                drop=train_params.get("drop", DEFAULT_DROP),
                batch_size=train_params.get("batch_size", DEFAULT_BATCH_SIZE),
                validation_split=0.0,
            )

            fold_trainer.save_model()

            try:
                from .._anonymization.anonymizer import Anonimizar
                from .._evaluation.evaluation import Evaluation

                model_path = str(fold_dir / "model")
                anonymizer = Anonimizar(model_path, labels=supported_labels)
                evaluator = Evaluation(
                    overlap_threshold=eval_params.get("overlap_threshold", DEFAULT_OVERLAP_THRESHOLD),
                    beta=eval_params.get("beta", DEFAULT_BETA),
                )

                evaluator.load_data(df_texts_val, df_gt_val)
                evaluator.extract_predictions(anonymizer)
                evaluator.evaluate_model()

                report = evaluator.get_detailed_report()
                report["fold"] = str(fold_idx)
                report.to_parquet(fold_dir / "metrics_detailed.parquet", index=False)

                summary = None
                overall_metrics = report[report["tp_entidade"] == ALL_ENTITIES_KEY]
                if len(overall_metrics) == 1:
                    summary = overall_metrics.iloc[0].to_dict()
                    summary["fold"] = str(fold_idx)

                holdout_report = None
                holdout_summary = None

                if df_holdout_texts is not None and df_holdout_gt is not None:
                    self.logger.debug("Fold %d: Avaliando no holdout test...", fold_idx)

                    evaluator_holdout = Evaluation(
                        overlap_threshold=eval_params.get("overlap_threshold", DEFAULT_OVERLAP_THRESHOLD),
                        beta=eval_params.get("beta", DEFAULT_BETA),
                    )

                    evaluator_holdout.load_data(df_holdout_texts, df_holdout_gt)
                    evaluator_holdout.extract_predictions(anonymizer)
                    evaluator_holdout.evaluate_model()

                    holdout_report = evaluator_holdout.get_detailed_report()
                    holdout_report["fold"] = "holdout_" + str(fold_idx)
                    holdout_report.to_parquet(fold_dir / "holdout_test_detailed.parquet", index=False)

                    overall_holdout = holdout_report[holdout_report["tp_entidade"] == ALL_ENTITIES_KEY]
                    if len(overall_holdout) == 1:
                        holdout_summary = overall_holdout.iloc[0].to_dict()
                        holdout_summary["fold"] = "holdout_" + str(fold_idx)
                        self.logger.info(
                            "Fold %d no Holdout Test: Precision=%.3f, Recall=%.3f, Fbeta=%.3f",
                            fold_idx,
                            holdout_summary.get("precision", 0),
                            holdout_summary.get("recall", 0),
                            holdout_summary.get("fbeta", 0),
                        )

                self.logger.debug("Fold %d concluído", fold_idx)
                reports_concat = [r for r in [report, holdout_report] if r is not None]
                combined = pd.concat(reports_concat, ignore_index=True) if reports_concat else pd.DataFrame()

            except Exception:
                self.logger.exception("Erro no fold %d", fold_idx)
                return pd.DataFrame(), None, None
            else:
                return (combined, summary, holdout_summary)

        start_time = time.time()

        if n_jobs == 1:
            results = []
            for fold_idx, (ids_train, ids_val) in enumerate(folds, start=1):
                result = run_single_fold(fold_idx, ids_train, ids_val)
                results.append(result)
        else:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_fold)(fold_idx, ids_train, ids_val)
                for fold_idx, (ids_train, ids_val) in enumerate(folds, start=1)
            )

        total_time = time.time() - start_time
        self.logger.info("Cross validation finalizado em %.1fs", total_time)

        all_reports, summary_metrics, holdout_results = _cv.aggregate_cv_results(
            results=results,
            output_path=output_path,
            holdout_test_size=holdout_test_size,
            logger=self.logger,
        )

        return all_reports, summary_metrics, results, holdout_results
