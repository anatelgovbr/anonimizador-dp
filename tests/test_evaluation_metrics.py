"""Testes unitários para módulo _evaluation/metrics.py.

Este módulo testa as funções de cálculo de métricas de avaliação NER:
calculate_overlap, calculate_entity_metrics, generate_evaluation_report,
format_summary_report, get_detailed_report e evaluate_multiple_thresholds.
"""

import logging

import pandas as pd
import pytest

from anonimizar._constants import ALL_ENTITIES_KEY, DEFAULT_BETA_VALUES, DEFAULT_OVERLAP_THRESHOLDS
from anonimizar._evaluation.metrics import (
    calculate_entity_metrics,
    evaluate_multiple_thresholds,
    format_summary_report,
    generate_evaluation_report,
    get_detailed_report,
)

__all__ = [
    "TestCalculateEntityMetrics",
    "TestEvaluateMultipleThresholds",
    "TestFormatSummaryReport",
    "TestGenerateEvaluationReport",
    "TestGetDetailedReport",
]

# Constantes para testes
_BETA_ONE = 1.0
_BETA_TWO = 2.0
_THRESHOLD_08 = 0.8
_THRESHOLD_05 = 0.5
_THRESHOLD_09 = 0.9
_TP_COUNT = 2
_FP_COUNT = 1
_FN_COUNT = 1
_TOTAL_ROWS_2X2 = 4  # 2 thresholds x 2 betas x 1 entity = 4 rows (+ ALL_ENTITIES)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_gt() -> pd.DataFrame:
    """Ground truth com duas entidades CPF em IDs distintos."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "tp_entidade": ["CPF", "CPF"],
            "start_entidade": [0, 5],
            "end_entidade": [11, 16],
        }
    )


@pytest.fixture
def sample_predictions() -> pd.DataFrame:
    """Predições correspondentes ao ground truth."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "tp_entidade": ["CPF", "CPF"],
            "start_entidade": [0, 5],
            "end_entidade": [11, 16],
        }
    )


@pytest.fixture
def sample_comparison() -> pd.DataFrame:
    """Comparação pré-calculada com dois TPs."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "tp_entidade": ["CPF", "CPF"],
            "y_true": [1, 1],
            "y_pred": [1, 1],
        }
    )


@pytest.fixture
def evaluation_results() -> dict:
    """Resultado de avaliação com uma entidade e totais."""
    return {
        "CPF": {
            "qtd_ids": 2,
            "qtd_entidades": 2,
            "fbeta": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "tp": 2,
            "fp": 0,
            "fn": 0,
        },
        "TODAS_ENTIDADES": {
            "qtd_ids": 2,
            "qtd_entidades": 2,
            "fbeta": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "tp": 2,
            "fp": 0,
            "fn": 0,
        },
    }


# ---------------------------------------------------------------------------
# TestCalculateEntityMetrics
# ---------------------------------------------------------------------------


class TestCalculateEntityMetrics:
    """Testes para função calculate_entity_metrics."""

    def test_perfect_predictions(self) -> None:
        """Predições perfeitas retornam métricas 1.0."""
        result = calculate_entity_metrics([1, 1, 0], [1, 1, 0], beta=_BETA_ONE)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["fbeta"] == pytest.approx(1.0)
        assert result["tp"] == _TP_COUNT
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_all_false_positives(self) -> None:
        """Todos FP: precision=0, recall=0."""
        result = calculate_entity_metrics([0, 0, 0], [1, 1, 1], beta=_BETA_ONE)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)
        assert result["fp"] == 3

    def test_all_false_negatives(self) -> None:
        """Todos FN: precision=0, recall=0."""
        result = calculate_entity_metrics([1, 1, 1], [0, 0, 0], beta=_BETA_ONE)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)
        assert result["fn"] == 3

    def test_mixed_results(self) -> None:
        """Resultados misturados contagem TP/FP/FN."""
        # y_true=[1,1,0,1], y_pred=[1,0,1,1]
        # TP=2, FP=1, FN=1
        result = calculate_entity_metrics([1, 1, 0, 1], [1, 0, 1, 1], beta=_BETA_ONE)
        assert result["tp"] == _TP_COUNT
        assert result["fp"] == 1
        assert result["fn"] == 1

    def test_beta_two_weights_recall(self) -> None:
        """Beta=2 pondera recall mais que precision, resultando em fbeta diferente."""
        # Precision=1.0, Recall=0.5 (1 TP, 0 FP, 1 FN)
        # Com beta=1: F1 = 2*(1*0.5)/(1+0.5) ≈ 0.667
        # Com beta=2: F2 = 5*(1*0.5)/(4*1+0.5) = 2.5/4.5 ≈ 0.556
        result_b1 = calculate_entity_metrics([1, 1], [1, 0], beta=_BETA_ONE)
        result_b2 = calculate_entity_metrics([1, 1], [1, 0], beta=_BETA_TWO)
        assert result_b1["fbeta"] != pytest.approx(result_b2["fbeta"])

    def test_all_zeros(self) -> None:
        """Listas totalmente zero: nenhuma entidade predita ou verdadeira."""
        result = calculate_entity_metrics([0, 0], [0, 0], beta=_BETA_ONE)
        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_return_keys(self) -> None:
        """Dicionário retornado contém todas as chaves esperadas."""
        result = calculate_entity_metrics([1], [1], beta=_BETA_ONE)
        expected_keys = {"fbeta", "precision", "recall", "tp", "fp", "fn"}
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# TestFormatSummaryReport
# ---------------------------------------------------------------------------


class TestFormatSummaryReport:
    """Testes para função format_summary_report."""

    def test_empty_results_returns_message(self) -> None:
        """Dicionário vazio retorna mensagem padrão."""
        result = format_summary_report({})
        assert "Nenhuma avaliação" in result

    def test_non_empty_contains_header(self, evaluation_results: dict) -> None:
        """Resultados não vazios contêm cabeçalho do relatório."""
        result = format_summary_report(evaluation_results)
        assert "RELATÓRIO DE AVALIAÇÃO" in result
        assert "=" * 30 in result

    def test_non_empty_contains_entity_type(self, evaluation_results: dict) -> None:
        """Relatório contém o tipo de entidade."""
        result = format_summary_report(evaluation_results)
        assert "CPF" in result

    def test_non_empty_contains_metrics(self, evaluation_results: dict) -> None:
        """Relatório contém as métricas F-beta, Precision, Recall."""
        result = format_summary_report(evaluation_results)
        assert "F-beta" in result
        assert "Precision" in result
        assert "Recall" in result

    def test_returns_string(self, evaluation_results: dict) -> None:
        """Função sempre retorna string."""
        result = format_summary_report(evaluation_results)
        assert isinstance(result, str)

    def test_contains_count_info(self, evaluation_results: dict) -> None:
        """Relatório contém informações de quantidade."""
        result = format_summary_report(evaluation_results)
        assert "Quantidade de IDs" in result
        assert "Quantidade de entidades" in result


# ---------------------------------------------------------------------------
# TestGetDetailedReport
# ---------------------------------------------------------------------------


class TestGetDetailedReport:
    """Testes para função get_detailed_report."""

    def test_empty_results_returns_empty_df(self) -> None:
        """Dicionário vazio retorna DataFrame vazio."""
        result = get_detailed_report({})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_non_empty_returns_dataframe(self, evaluation_results: dict) -> None:
        """Resultados não vazios retornam DataFrame com linhas."""
        result = get_detailed_report(evaluation_results)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(evaluation_results)

    def test_has_tp_entidade_column(self, evaluation_results: dict) -> None:
        """DataFrame resultante tem coluna 'tp_entidade'."""
        result = get_detailed_report(evaluation_results)
        assert "tp_entidade" in result.columns

    def test_entity_values_present(self, evaluation_results: dict) -> None:
        """Tipos de entidade estão no DataFrame."""
        result = get_detailed_report(evaluation_results)
        assert "CPF" in result["tp_entidade"].to_numpy()

    def test_metrics_columns_present(self, evaluation_results: dict) -> None:
        """Colunas de métricas estão presentes."""
        result = get_detailed_report(evaluation_results)
        for col in ("fbeta", "precision", "recall", "tp", "fp", "fn"):
            assert col in result.columns

    def test_single_entity(self) -> None:
        """Uma única entidade resulta em DataFrame com 1 linha."""
        results = {
            "EMAIL": {
                "qtd_ids": 1,
                "qtd_entidades": 3,
                "fbeta": 0.8,
                "precision": 0.9,
                "recall": 0.7,
                "tp": 3,
                "fp": 0,
                "fn": 1,
            }
        }
        result = get_detailed_report(results)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestGenerateEvaluationReport
# ---------------------------------------------------------------------------


class TestGenerateEvaluationReport:
    """Testes para função generate_evaluation_report."""

    @pytest.fixture
    def mock_logger(self) -> logging.Logger:
        """Logger simples para testes."""
        return logging.getLogger("test_metrics")

    def _make_comparison_fn(self, comparison_df: pd.DataFrame):
        """Cria função de comparação que retorna DataFrame fixo."""

        def gen_fn(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:  # noqa: ARG001
            return comparison_df

        return gen_fn

    def test_basic_report_structure(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        sample_comparison: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Relatório básico retorna dict com chave de entidade e TODAS_ENTIDADES."""
        gen_fn = self._make_comparison_fn(sample_comparison)

        result = generate_evaluation_report(
            df_ground_truth=sample_gt,
            predictions=sample_predictions,
            comparison_data=sample_comparison,
            overlap_threshold=_THRESHOLD_08,
            beta=_BETA_ONE,
            entity_types=["CPF"],
            logger=mock_logger,
            generate_comparison_fn=gen_fn,
        )

        assert isinstance(result, dict)
        assert "CPF" in result
        assert ALL_ENTITIES_KEY in result

    def test_entity_metrics_keys(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        sample_comparison: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Métricas de cada entidade contêm todas as chaves esperadas."""
        gen_fn = self._make_comparison_fn(sample_comparison)

        result = generate_evaluation_report(
            df_ground_truth=sample_gt,
            predictions=sample_predictions,
            comparison_data=sample_comparison,
            overlap_threshold=_THRESHOLD_08,
            beta=_BETA_ONE,
            entity_types=["CPF"],
            logger=mock_logger,
            generate_comparison_fn=gen_fn,
        )

        cpf_metrics = result["CPF"]
        for key in ("qtd_ids", "qtd_entidades", "fbeta", "precision", "recall", "tp", "fp", "fn"):
            assert key in cpf_metrics

    def test_entity_not_in_ground_truth_skipped(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        sample_comparison: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Entidade sem ground truth é ignorada (warning)."""
        gen_fn = self._make_comparison_fn(sample_comparison)

        result = generate_evaluation_report(
            df_ground_truth=sample_gt,
            predictions=sample_predictions,
            comparison_data=sample_comparison,
            overlap_threshold=_THRESHOLD_08,
            beta=_BETA_ONE,
            entity_types=["CPF", "EMAIL"],  # EMAIL não está no gt
            logger=mock_logger,
            generate_comparison_fn=gen_fn,
        )

        # EMAIL não tem gt, deve ter sido ignorado
        assert "EMAIL" not in result

    def test_empty_comparison_entity_gets_zero_metrics(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Entidade sem comparação recebe métricas zeradas."""
        # Comparação vazia para CPF
        empty_comparison = pd.DataFrame(columns=["id", "tp_entidade", "y_true", "y_pred"])

        gen_fn = self._make_comparison_fn(empty_comparison)

        result = generate_evaluation_report(
            df_ground_truth=sample_gt,
            predictions=sample_predictions,
            comparison_data=empty_comparison,
            overlap_threshold=_THRESHOLD_08,
            beta=_BETA_ONE,
            entity_types=["CPF"],
            logger=mock_logger,
            generate_comparison_fn=gen_fn,
        )

        assert result["CPF"]["fbeta"] == pytest.approx(0.0)
        assert result["CPF"]["precision"] == pytest.approx(0.0)
        assert result["CPF"]["recall"] == pytest.approx(0.0)
        assert result["CPF"]["fn"] == len(sample_gt)

    def test_perfect_predictions_give_fbeta_one(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        sample_comparison: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Predições perfeitas resultam em fbeta=1.0."""
        gen_fn = self._make_comparison_fn(sample_comparison)

        result = generate_evaluation_report(
            df_ground_truth=sample_gt,
            predictions=sample_predictions,
            comparison_data=sample_comparison,
            overlap_threshold=_THRESHOLD_08,
            beta=_BETA_ONE,
            entity_types=["CPF"],
            logger=mock_logger,
            generate_comparison_fn=gen_fn,
        )

        assert result["CPF"]["fbeta"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestEvaluateMultipleThresholds
# ---------------------------------------------------------------------------


class TestEvaluateMultipleThresholds:
    """Testes para função evaluate_multiple_thresholds."""

    @pytest.fixture
    def mock_logger(self) -> logging.Logger:
        """Logger simples para testes."""
        return logging.getLogger("test_metrics_multi")

    def _make_comparison_fn(self, comparison_df: pd.DataFrame):
        """Cria função de comparação que retorna DataFrame fixo."""

        def gen_fn(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:  # noqa: ARG001
            return comparison_df

        return gen_fn

    def test_default_thresholds_used_when_none(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        sample_comparison: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Sem thresholds fornecidos, usa DEFAULT_OVERLAP_THRESHOLDS."""
        gen_fn = self._make_comparison_fn(sample_comparison)

        result = evaluate_multiple_thresholds(
            predictions=sample_predictions,
            df_ground_truth=sample_gt,
            generate_comparison_fn=gen_fn,
            logger=mock_logger,
        )

        assert isinstance(result, pd.DataFrame)
        expected_rows = len(DEFAULT_OVERLAP_THRESHOLDS) * len(DEFAULT_BETA_VALUES) * 2  # 2 entities (CPF + ALL)
        assert len(result) == expected_rows

    def test_custom_thresholds_and_betas(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        sample_comparison: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Com thresholds e betas customizados, colunas corretas no resultado."""
        gen_fn = self._make_comparison_fn(sample_comparison)

        result = evaluate_multiple_thresholds(
            predictions=sample_predictions,
            df_ground_truth=sample_gt,
            generate_comparison_fn=gen_fn,
            overlap_thresholds=[_THRESHOLD_05, _THRESHOLD_08],
            beta_values=[_BETA_ONE],
            logger=mock_logger,
        )

        assert "overlap_threshold" in result.columns
        assert "beta" in result.columns
        assert "tp_entidade" in result.columns

    def test_result_contains_expected_columns(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        sample_comparison: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """DataFrame resultado tem todas as colunas esperadas."""
        gen_fn = self._make_comparison_fn(sample_comparison)

        result = evaluate_multiple_thresholds(
            predictions=sample_predictions,
            df_ground_truth=sample_gt,
            generate_comparison_fn=gen_fn,
            overlap_thresholds=[_THRESHOLD_08],
            beta_values=[_BETA_ONE],
            logger=mock_logger,
        )

        expected_cols = {
            "tp_entidade",
            "qtd_ids",
            "qtd_entidades",
            "fbeta",
            "precision",
            "recall",
            "tp",
            "fp",
            "fn",
            "overlap_threshold",
            "beta",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_single_threshold_single_beta(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        sample_comparison: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Um threshold e um beta: 2 linhas (1 entidade + ALL_ENTITIES)."""
        gen_fn = self._make_comparison_fn(sample_comparison)

        result = evaluate_multiple_thresholds(
            predictions=sample_predictions,
            df_ground_truth=sample_gt,
            generate_comparison_fn=gen_fn,
            overlap_thresholds=[_THRESHOLD_08],
            beta_values=[_BETA_ONE],
            logger=mock_logger,
        )

        # 1 threshold x 1 beta x 2 entity_types (CPF + ALL_ENTITIES)
        assert len(result) == 2

    def test_none_logger_does_not_raise(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        sample_comparison: pd.DataFrame,
    ) -> None:
        """Logger None é substituído por logger padrão sem levantar erro."""
        gen_fn = self._make_comparison_fn(sample_comparison)

        result = evaluate_multiple_thresholds(
            predictions=sample_predictions,
            df_ground_truth=sample_gt,
            generate_comparison_fn=gen_fn,
            overlap_thresholds=[_THRESHOLD_08],
            beta_values=[_BETA_ONE],
            logger=None,
        )

        assert isinstance(result, pd.DataFrame)

    def test_empty_results_when_comparison_fn_raises(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Erros na função de comparação são silenciados e continuam as iterações."""

        def failing_fn(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:  # noqa: ARG001
            msg = "Erro simulado"
            raise RuntimeError(msg)

        result = evaluate_multiple_thresholds(
            predictions=sample_predictions,
            df_ground_truth=sample_gt,
            generate_comparison_fn=failing_fn,
            overlap_thresholds=[_THRESHOLD_08, _THRESHOLD_09],
            beta_values=[_BETA_ONE],
            logger=mock_logger,
        )

        # Todos os erros foram silenciados, resultado vazio
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
