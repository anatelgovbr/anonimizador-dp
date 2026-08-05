"""Testes para _training/cross_validation.py.

Cobre branches nao exercitados:
- make_stratified_folds_by_id: sobra > 0 (branch linha 116) e split_size == 0
- separate_holdout_test: funcao completa (linhas 204-274)
- aggregate_cv_results: branch holdout_results (linhas 309-328)
"""

import json
import logging

import pandas as pd
import pytest

from anonimizar._training.cross_validation import (
    aggregate_cv_results,
    fold_for_train,
    make_folds_by_id,
    make_stratified_folds_by_id,
    separate_holdout_test,
)


@pytest.fixture
def logger() -> logging.Logger:
    """Logger silencioso para testes."""
    log = logging.getLogger("test_cross_validation")
    log.setLevel(logging.DEBUG)
    return log


@pytest.fixture
def df_entidades_simples() -> pd.DataFrame:
    """DataFrame simples com 10 documentos, cada um com uma entidade CPF."""
    return pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "tp_entidade": ["CPF"] * 10,
            "start_entidade": [4] * 10,
            "end_entidade": [18] * 10,
        }
    )


@pytest.fixture
def df_textos_simples() -> pd.DataFrame:
    """DataFrame simples com 10 textos."""
    return pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "text": [f"CPF 529.982.247-{i:02d}" for i in range(10)],
        }
    )


# =============================================================================
# make_folds_by_id
# =============================================================================


class TestMakeFoldsByID:
    """Testa make_folds_by_id."""

    def test_cria_folds_corretos(self, df_entidades_simples, logger) -> None:
        """Cria n_splits folds com ids de treino e validacao."""
        folds = make_folds_by_id(df_entidades_simples, n_splits=5, random_state=42, shuffle=True, logger=logger)
        assert len(folds) == 5
        for ids_train, ids_val in folds:
            assert len(ids_train) > 0
            assert len(ids_val) > 0

    def test_usa_coluna_id_doc(self, logger) -> None:
        """Usa coluna 'id_doc' quando disponivel."""
        _df = pd.DataFrame(
            {
                "id_doc": list(range(1, 7)),
                "tp_entidade": ["CPF"] * 6,
            }
        )
        folds = make_folds_by_id(_df, n_splits=3, random_state=None, shuffle=False, logger=logger)
        assert len(folds) == 3


# =============================================================================
# make_stratified_folds_by_id
# =============================================================================


class TestMakeStratifiedFoldsByID:
    """Testa make_stratified_folds_by_id."""

    def test_folds_basicos(self, df_entidades_simples, logger) -> None:
        """Cria folds estratificados para um label."""
        folds = make_stratified_folds_by_id(
            df_entidades_simples,
            features=["CPF"],
            n_splits=5,
            random_state=42,
            shuffle=True,
            logger=logger,
        )
        assert len(folds) == 5

    def test_sobra_distribuida(self, logger) -> None:
        """Linha 116-120: sobra > 0 distribui documentos extras entre folds."""
        # 11 documentos, 3 splits -> split_size=3, sobra=2 para CPF
        _df = pd.DataFrame(
            {
                "id": list(range(1, 12)),
                "tp_entidade": ["CPF"] * 11,
            }
        )
        folds = make_stratified_folds_by_id(
            _df, features=["CPF"], n_splits=3, random_state=0, shuffle=False, logger=logger
        )
        assert len(folds) == 3
        all_val_ids = [val for _, val in folds]
        # Cada documento deve aparecer exatamente uma vez na validacao
        flat_val = [i for val in all_val_ids for i in val]
        assert len(flat_val) == len(set(flat_val))  # sem duplicatas

    def test_split_size_zero_distribui_individualmente(self, logger) -> None:
        """Linha 123-127: split_size==0 (menos docs que splits) distribui um a um."""
        # 2 documentos com CPF, 5 splits -> split_size=0
        _df = pd.DataFrame(
            {
                "id": [1, 2],
                "tp_entidade": ["CPF", "CPF"],
            }
        )
        folds = make_stratified_folds_by_id(
            _df, features=["CPF"], n_splits=5, random_state=0, shuffle=False, logger=logger
        )
        assert len(folds) == 5


# =============================================================================
# fold_for_train
# =============================================================================


class TestFoldForTrain:
    """Testa fold_for_train."""

    def test_gera_pares_treino_val(self, logger) -> None:
        """Cada fold tem train = todos menos val."""
        folds = [[1, 2], [3, 4], [5, 6]]
        result = fold_for_train(folds, logger)
        assert len(result) == 3
        train_0, val_0 = result[0]
        assert 1 not in train_0
        assert 2 not in train_0
        assert val_0 == [1, 2]


# =============================================================================
# separate_holdout_test
# =============================================================================


class TestSeparateHoldoutTest:
    """Testa separate_holdout_test (linhas 204-274)."""

    def test_holdout_test_size_invalido(self, df_entidades_simples, df_textos_simples, tmp_path, logger) -> None:
        """Linha 204-207: holdout_test_size fora do intervalo lanca ValueError."""
        with pytest.raises(ValueError, match=r"holdout_test_size deve estar"):
            separate_holdout_test(
                df_entidades=df_entidades_simples,
                df_textos=df_textos_simples,
                holdout_test_size=0.6,
                features=["CPF"],
                random_state=42,
                output_path=tmp_path,
                logger=logger,
            )

    def test_sem_fn_lanca_erro(self, df_entidades_simples, df_textos_simples, tmp_path, logger) -> None:
        """Linha 226-228: sem make_folds_fn nem make_stratified_folds_fn lanca ValueError."""
        with pytest.raises(ValueError, match=r"make_stratified_folds_fn ou make_folds_fn"):
            separate_holdout_test(
                df_entidades=df_entidades_simples,
                df_textos=df_textos_simples,
                holdout_test_size=0.2,
                features=["CPF"],
                random_state=42,
                output_path=tmp_path,
                logger=logger,
                holdout_stratify=False,
                make_folds_fn=None,
            )

    def test_holdout_simples(self, df_entidades_simples, df_textos_simples, tmp_path, logger) -> None:
        """Linhas 223-274: holdout_stratify=False com make_folds_fn."""

        def mock_folds_fn(df_entidades, n_splits, random_state):  # noqa: ARG001
            ids = df_entidades["id"].unique().tolist()
            mid = len(ids) // n_splits
            return [(ids[mid:], ids[:mid])]

        holdout_ids, cv_ids, _df_ht, _df_gt, _df_ent_cv, _df_txt_cv = separate_holdout_test(
            df_entidades=df_entidades_simples,
            df_textos=df_textos_simples,
            holdout_test_size=0.2,
            features=["CPF"],
            random_state=42,
            output_path=tmp_path,
            logger=logger,
            holdout_stratify=False,
            make_folds_fn=mock_folds_fn,
        )
        assert len(holdout_ids) > 0
        assert len(cv_ids) > 0
        assert (tmp_path / "holdout_test_ids.json").exists()
        assert (tmp_path / "holdout_test_ids.csv").exists()

    def test_holdout_estratificado(self, df_entidades_simples, df_textos_simples, tmp_path, logger) -> None:
        """Linhas 218-222: holdout_stratify=True com make_stratified_folds_fn."""

        def mock_strat_fn(df_entidades, features, n_splits, random_state):  # noqa: ARG001
            ids = df_entidades["id"].unique().tolist()
            mid = len(ids) // n_splits
            return [(ids[mid:], ids[:mid])]

        holdout_ids, cv_ids, _df_ht, _df_gt, _df_ent_cv, _df_txt_cv = separate_holdout_test(
            df_entidades=df_entidades_simples,
            df_textos=df_textos_simples,
            holdout_test_size=0.2,
            features=["CPF"],
            random_state=42,
            output_path=tmp_path,
            logger=logger,
            holdout_stratify=True,
            make_stratified_folds_fn=mock_strat_fn,
        )
        assert len(holdout_ids) > 0
        assert len(cv_ids) > 0


# =============================================================================
# aggregate_cv_results
# =============================================================================


class TestAggregateCVResults:
    """Testa aggregate_cv_results."""

    def _make_report(self) -> pd.DataFrame:
        """Cria um report de fold simples."""
        return pd.DataFrame(
            {
                "entidade": ["CPF"],
                "precision": [0.9],
                "recall": [0.8],
                "fbeta": [0.85],
            }
        )

    def test_sem_summary_sem_holdout(self, tmp_path, logger) -> None:
        """Resultados sem summary nem holdout nao geram arquivos."""
        results = [(self._make_report(), None, None)]
        all_reports, summary, holdout = aggregate_cv_results(results, tmp_path, holdout_test_size=None, logger=logger)
        assert len(all_reports) == 1
        assert summary == []
        assert holdout is None

    def test_com_summary_gera_parquet(self, tmp_path, logger) -> None:
        """Com summary_metrics, gera arquivos parquet."""
        summary_m = {"fold": 0, "precision": 0.9, "recall": 0.85, "fbeta": 0.87}
        results = [(self._make_report(), summary_m, None)]
        aggregate_cv_results(results, tmp_path, holdout_test_size=None, logger=logger)
        assert (tmp_path / "all_folds_detailed.parquet").exists()
        assert (tmp_path / "fold_summaries.parquet").exists()

    def test_com_holdout_gera_arquivos(self, tmp_path, logger) -> None:
        """Linhas 309-328: com holdout_results gera parquet, csv e json."""
        summary_m = {"fold": 0, "precision": 0.9, "recall": 0.85, "fbeta": 0.87}
        holdout_m = {"fold": 0, "precision": 0.88, "recall": 0.82, "fbeta": 0.85}
        results = [(self._make_report(), summary_m, holdout_m)]
        aggregate_cv_results(results, tmp_path, holdout_test_size=0.2, logger=logger)
        assert (tmp_path / "holdout_test_summary.parquet").exists()
        assert (tmp_path / "holdout_test_summary.csv").exists()
        assert (tmp_path / "holdout_test_stats.json").exists()
        with (tmp_path / "holdout_test_stats.json").open(encoding="utf-8") as f:
            stats = json.load(f)
        assert "mean_fbeta" in stats
