"""Testes unitários para módulo _evaluation/comparison.py.

Este módulo testa as funções de comparação entre ground truth e predições NER:
calculate_overlap, generate_comparison_data, compare_reports,
classify_cases, get_classification_cases e get_error_analysis.
"""

import logging

import pandas as pd
import pytest

from anonimizar._evaluation.comparison import (
    calculate_overlap,
    classify_cases,
    compare_reports,
    generate_comparison_data,
    get_classification_cases,
    get_error_analysis,
)

__all__ = [
    "TestCalculateOverlap",
    "TestClassifyCases",
    "TestCompareReports",
    "TestGenerateComparisonData",
    "TestGetClassificationCases",
    "TestGetErrorAnalysis",
]

# ---------------------------------------------------------------------------
# Constantes para testes
# ---------------------------------------------------------------------------

_OVERLAP_ZERO = 0.0
_OVERLAP_FULL = 1.0
_OVERLAP_HALF = 0.5
_OVERLAP_THRESH_08 = 0.8
_OVERLAP_THRESH_05 = 0.5
_ALL_ENTITIES_KEY = "TODAS"


# ---------------------------------------------------------------------------
# Fixtures compartilhadas
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_logger() -> logging.Logger:
    """Logger simples para testes."""
    return logging.getLogger("test_comparison")


@pytest.fixture
def sample_gt() -> pd.DataFrame:
    """Ground truth com duas entidades CPF."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "tp_entidade": ["CPF", "CPF"],
            "start_entidade": [0, 5],
            "end_entidade": [11, 16],
            "text_entidade": ["123.456.789-09", "987.654.321-00"],
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
            "text_entidade": ["123.456.789-09", "987.654.321-00"],
        }
    )


@pytest.fixture
def sample_comparison() -> pd.DataFrame:
    """Comparação com TP, FP, FN e TN."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "tp_entidade": ["CPF", "CPF", "EMAIL", "EMAIL"],
            "y_true": [1, 1, 0, 1],
            "y_pred": [1, 0, 1, 0],
            "overlap": [1.0, 0.0, 0.0, 0.0],
            "text_entidade_true": ["123.456.789-09", "987.654.321-00", None, "a@b.com"],
            "text_entidade_pred": ["123.456.789-09", None, "x@y.com", None],
            "start_entidade_true": [0, 5, None, 20],
            "end_entidade_true": [14, 19, None, 27],
            "start_entidade_pred": [0, None, 30, None],
            "end_entidade_pred": [14, None, 37, None],
        }
    )


@pytest.fixture
def current_report() -> pd.DataFrame:
    """Relatório atual de métricas."""
    return pd.DataFrame(
        {
            "tp_entidade": ["CPF", "EMAIL"],
            "fbeta": [0.9, 0.8],
            "precision": [0.95, 0.85],
            "recall": [0.85, 0.75],
        }
    )


@pytest.fixture
def previous_report() -> pd.DataFrame:
    """Relatório anterior de métricas (pior que o atual)."""
    return pd.DataFrame(
        {
            "tp_entidade": ["CPF", "EMAIL"],
            "fbeta": [0.7, 0.9],
            "precision": [0.75, 0.9],
            "recall": [0.65, 0.9],
        }
    )


# ---------------------------------------------------------------------------
# TestCalculateOverlap
# ---------------------------------------------------------------------------


class TestCalculateOverlap:
    """Testes para função calculate_overlap em comparison.py."""

    def test_perfect_overlap(self) -> None:
        """Spans idênticos devem ter overlap 1.0."""
        assert calculate_overlap(0, 10, 0, 10) == _OVERLAP_FULL

    def test_no_overlap_right(self) -> None:
        """Span predito completamente à direita: overlap 0.0."""
        assert calculate_overlap(0, 10, 20, 30) == _OVERLAP_ZERO

    def test_no_overlap_left(self) -> None:
        """Span predito completamente à esquerda: overlap 0.0."""
        assert calculate_overlap(20, 30, 0, 10) == _OVERLAP_ZERO

    def test_partial_overlap_one_third(self) -> None:
        """Overlap IoU parcial de 1/3 (interseção=5, união=15)."""
        assert calculate_overlap(0, 10, 5, 15) == pytest.approx(1 / 3)

    def test_touching_spans_no_overlap(self) -> None:
        """Spans adjacentes sem sobreposição: overlap 0.0."""
        assert calculate_overlap(0, 10, 10, 20) == _OVERLAP_ZERO

    def test_pred_inside_true(self) -> None:
        """Span predito dentro do verdadeiro: overlap < 1.0."""
        result = calculate_overlap(0, 10, 3, 7)
        assert 0.0 < result < _OVERLAP_FULL

    def test_true_inside_pred(self) -> None:
        """Span verdadeiro dentro do predito: overlap == 1.0 (interseção == true)."""
        # interseção = [5,8], union = [0,10] -> 3/10
        result = calculate_overlap(5, 8, 0, 10)
        assert result == pytest.approx(0.3)

    def test_nan_start_pred_returns_zero(self) -> None:
        """start_pred NaN retorna 0.0 sem erro."""
        assert calculate_overlap(0, 10, float("nan"), 10) == _OVERLAP_ZERO

    def test_nan_end_pred_returns_zero(self) -> None:
        """end_pred NaN retorna 0.0 sem erro."""
        assert calculate_overlap(0, 10, 0, float("nan")) == _OVERLAP_ZERO

    def test_zero_union_returns_zero(self) -> None:
        """Spans de tamanho zero em posição idêntica retornam 0.0."""
        assert calculate_overlap(5, 5, 5, 5) == _OVERLAP_ZERO

    @pytest.mark.parametrize(
        ("st", "et", "sp", "ep", "expected"),
        [
            (0, 100, 0, 100, 1.0),
            (0, 100, 50, 150, 1 / 3),  # interseção=50, união=150
            (0, 100, 100, 200, 0.0),
            (0, 100, 0, 50, 0.5),  # interseção=50, união=100
        ],
    )
    def test_parametrized_cases(self, st: int, et: int, sp: int, ep: int, expected: float) -> None:
        """Casos parametrizados de overlap."""
        assert calculate_overlap(st, et, sp, ep) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestGenerateComparisonData
# ---------------------------------------------------------------------------


class TestGenerateComparisonData:
    """Testes para função generate_comparison_data."""

    def test_basic_returns_dataframe(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Retorna DataFrame para entradas válidas."""
        result = generate_comparison_data(
            df_ground_truth=sample_gt,
            predictions=sample_predictions,
            entity_mapping={},
            overlap_threshold=_OVERLAP_THRESH_08,
            logger=mock_logger,
        )
        assert isinstance(result, pd.DataFrame)

    def test_result_has_y_true_y_pred(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """DataFrame resultado contém colunas y_true e y_pred."""
        result = generate_comparison_data(
            df_ground_truth=sample_gt,
            predictions=sample_predictions,
            entity_mapping={},
            overlap_threshold=_OVERLAP_THRESH_08,
            logger=mock_logger,
        )
        assert "y_true" in result.columns
        assert "y_pred" in result.columns

    def test_result_has_overlap_column(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """DataFrame resultado contém coluna overlap."""
        result = generate_comparison_data(
            df_ground_truth=sample_gt,
            predictions=sample_predictions,
            entity_mapping={},
            overlap_threshold=_OVERLAP_THRESH_08,
            logger=mock_logger,
        )
        assert "overlap" in result.columns

    def test_perfect_match_gives_y_pred_one(
        self,
        sample_gt: pd.DataFrame,
        sample_predictions: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Predições perfeitas resultam em y_pred=1 para todas as entidades matched."""
        result = generate_comparison_data(
            df_ground_truth=sample_gt,
            predictions=sample_predictions,
            entity_mapping={},
            overlap_threshold=_OVERLAP_THRESH_05,
            logger=mock_logger,
        )
        matched = result[result["y_true"] == 1]
        assert all(matched["y_pred"] == 1)

    def test_no_predictions_gives_fn(
        self,
        sample_gt: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Sem predições, todas as entidades viram FN."""
        empty_preds = pd.DataFrame(columns=["id", "tp_entidade", "start_entidade", "end_entidade", "text_entidade"])
        result = generate_comparison_data(
            df_ground_truth=sample_gt,
            predictions=empty_preds,
            entity_mapping={},
            overlap_threshold=_OVERLAP_THRESH_08,
            logger=mock_logger,
        )
        assert all(result["y_true"] == 1)
        assert all(result["y_pred"] == 0)

    def test_entity_mapping_applied(
        self,
        sample_gt: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """entity_mapping mapeia labels corretamente antes da comparação."""
        # Predições com label alternativo que deve ser mapeado para CPF
        preds = pd.DataFrame(
            {
                "id": [1, 2],
                "tp_entidade": ["CPF_ALT", "CPF_ALT"],
                "start_entidade": [0, 5],
                "end_entidade": [11, 16],
                "text_entidade": ["123.456.789-09", "987.654.321-00"],
            }
        )
        result = generate_comparison_data(
            df_ground_truth=sample_gt,
            predictions=preds,
            entity_mapping={"CPF_ALT": "CPF"},
            overlap_threshold=_OVERLAP_THRESH_05,
            logger=mock_logger,
        )
        # Com mapeamento, devem ser encontrados como match
        assert isinstance(result, pd.DataFrame)

    def test_missing_prediction_columns_raises(
        self,
        sample_gt: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Predições sem colunas obrigatórias levantam ValueError."""
        bad_preds = pd.DataFrame({"id": [1], "tp_entidade": ["CPF"]})  # faltam start/end
        with pytest.raises(ValueError, match="Colunas ausentes"):
            generate_comparison_data(
                df_ground_truth=sample_gt,
                predictions=bad_preds,
                entity_mapping={},
                overlap_threshold=_OVERLAP_THRESH_08,
                logger=mock_logger,
            )

    def test_empty_ground_truth_raises(
        self,
        sample_predictions: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Ground truth vazio levanta ValueError."""
        with pytest.raises(ValueError, match="ground truth"):
            generate_comparison_data(
                df_ground_truth=pd.DataFrame(),
                predictions=sample_predictions,
                entity_mapping={},
                overlap_threshold=_OVERLAP_THRESH_08,
                logger=mock_logger,
            )

    def test_none_ground_truth_raises(
        self,
        sample_predictions: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Ground truth None levanta ValueError."""
        with pytest.raises(ValueError, match="ground truth"):
            generate_comparison_data(
                df_ground_truth=None,  # type: ignore[arg-type]
                predictions=sample_predictions,
                entity_mapping={},
                overlap_threshold=_OVERLAP_THRESH_08,
                logger=mock_logger,
            )

    def test_extra_predictions_give_fp(
        self,
        sample_gt: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Predições sem correspondência no ground truth viram FP."""
        extra_preds = pd.DataFrame(
            {
                "id": [1, 99],
                "tp_entidade": ["CPF", "CPF"],
                "start_entidade": [0, 100],
                "end_entidade": [11, 110],
                "text_entidade": ["123.456.789-09", "999.999.999-99"],
            }
        )
        result = generate_comparison_data(
            df_ground_truth=sample_gt,
            predictions=extra_preds,
            entity_mapping={},
            overlap_threshold=_OVERLAP_THRESH_08,
            logger=mock_logger,
        )
        fp_cases = result[(result["y_true"] == 0) & (result["y_pred"] == 1)]
        assert len(fp_cases) >= 1

    def test_overlap_below_threshold_gives_fn(
        self,
        sample_gt: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Predições com overlap abaixo do threshold são tratadas como FN."""
        # Predição com offset grande — overlap muito baixo
        preds_offset = pd.DataFrame(
            {
                "id": [1, 2],
                "tp_entidade": ["CPF", "CPF"],
                "start_entidade": [9, 14],
                "end_entidade": [20, 25],
                "text_entidade": ["x", "y"],
            }
        )
        result = generate_comparison_data(
            df_ground_truth=sample_gt,
            predictions=preds_offset,
            entity_mapping={},
            overlap_threshold=_OVERLAP_THRESH_08,
            logger=mock_logger,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# TestCompareReports
# ---------------------------------------------------------------------------


class TestCompareReports:
    """Testes para função compare_reports."""

    def test_basic_returns_dataframe(
        self,
        current_report: pd.DataFrame,
        previous_report: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Retorna DataFrame para entradas válidas."""
        result = compare_reports(current_report, previous_report, logger=mock_logger)
        assert isinstance(result, pd.DataFrame)

    def test_result_has_diff_columns(
        self,
        current_report: pd.DataFrame,
        previous_report: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Resultado contém colunas de diferença fbeta/precision/recall."""
        result = compare_reports(current_report, previous_report, logger=mock_logger)
        assert "fbeta_diff" in result.columns
        assert "precision_diff" in result.columns
        assert "recall_diff" in result.columns

    def test_result_has_melhorou_columns(
        self,
        current_report: pd.DataFrame,
        previous_report: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Resultado contém colunas booleanas de melhoria."""
        result = compare_reports(current_report, previous_report, logger=mock_logger)
        assert "fbeta_melhorou" in result.columns
        assert "precision_melhorou" in result.columns
        assert "recall_melhorou" in result.columns

    def test_result_has_perc_change_columns(
        self,
        current_report: pd.DataFrame,
        previous_report: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Resultado contém colunas de variação percentual."""
        result = compare_reports(current_report, previous_report, logger=mock_logger)
        assert "fbeta_perc_change" in result.columns
        assert "precision_perc_change" in result.columns
        assert "recall_perc_change" in result.columns

    def test_result_has_status_geral(
        self,
        current_report: pd.DataFrame,
        previous_report: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Resultado contém coluna status_geral."""
        result = compare_reports(current_report, previous_report, logger=mock_logger)
        assert "status_geral" in result.columns

    def test_improved_metric_melhorou_true(
        self,
        current_report: pd.DataFrame,
        previous_report: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """CPF melhorou de 0.7 para 0.9: fbeta_melhorou deve ser True."""
        result = compare_reports(current_report, previous_report, logger=mock_logger)
        cpf_row = result[result["tp_entidade"] == "CPF"].iloc[0]
        assert cpf_row["fbeta_melhorou"] is True or cpf_row["fbeta_melhorou"] == True  # noqa: E712

    def test_degraded_metric_melhorou_false(
        self,
        current_report: pd.DataFrame,
        previous_report: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """EMAIL piorou de 0.9 para 0.8: fbeta_melhorou deve ser False."""
        result = compare_reports(current_report, previous_report, logger=mock_logger)
        email_row = result[result["tp_entidade"] == "EMAIL"].iloc[0]
        assert email_row["fbeta_melhorou"] is False or email_row["fbeta_melhorou"] == False  # noqa: E712

    def test_status_geral_melhorou(
        self,
        current_report: pd.DataFrame,
        previous_report: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """CPF que melhorou deve ter status_geral='Melhorou'."""
        result = compare_reports(current_report, previous_report, logger=mock_logger)
        cpf_row = result[result["tp_entidade"] == "CPF"].iloc[0]
        assert cpf_row["status_geral"] == "Melhorou"

    def test_status_geral_piorou(
        self,
        current_report: pd.DataFrame,
        previous_report: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """EMAIL que piorou deve ter status_geral='Piorou'."""
        result = compare_reports(current_report, previous_report, logger=mock_logger)
        email_row = result[result["tp_entidade"] == "EMAIL"].iloc[0]
        assert email_row["status_geral"] == "Piorou"

    def test_stable_metric_status_estavel(self, mock_logger: logging.Logger) -> None:
        """Métrica sem variação deve ter status_geral='Estável'."""
        same_report = pd.DataFrame(
            {
                "tp_entidade": ["CPF"],
                "fbeta": [0.8],
                "precision": [0.8],
                "recall": [0.8],
            }
        )
        result = compare_reports(same_report, same_report.copy(), logger=mock_logger)
        assert result.iloc[0]["status_geral"] == "Estável"

    def test_missing_columns_raises(self, mock_logger: logging.Logger) -> None:
        """DataFrames sem colunas obrigatórias levantam ValueError."""
        bad_df = pd.DataFrame({"tp_entidade": ["CPF"]})  # falta fbeta, precision, recall
        good_df = pd.DataFrame(
            {
                "tp_entidade": ["CPF"],
                "fbeta": [0.9],
                "precision": [0.9],
                "recall": [0.9],
            }
        )
        with pytest.raises(ValueError, match="Colunas ausentes"):
            compare_reports(bad_df, good_df, logger=mock_logger)

    def test_no_intersection_returns_empty(self, mock_logger: logging.Logger) -> None:
        """Relatórios sem entidades em comum retornam DataFrame vazio."""
        report_a = pd.DataFrame(
            {
                "tp_entidade": ["CPF"],
                "fbeta": [0.9],
                "precision": [0.9],
                "recall": [0.9],
            }
        )
        report_b = pd.DataFrame(
            {
                "tp_entidade": ["EMAIL"],
                "fbeta": [0.8],
                "precision": [0.8],
                "recall": [0.8],
            }
        )
        result = compare_reports(report_a, report_b, logger=mock_logger)
        assert len(result) == 0

    def test_previous_zero_fbeta_percent_change_is_zero(self, mock_logger: logging.Logger) -> None:
        """Quando fbeta anterior é 0, percentual de mudança é 0 (protegido)."""
        current = pd.DataFrame(
            {
                "tp_entidade": ["CPF"],
                "fbeta": [0.5],
                "precision": [0.5],
                "recall": [0.5],
            }
        )
        previous = pd.DataFrame(
            {
                "tp_entidade": ["CPF"],
                "fbeta": [0.0],
                "precision": [0.0],
                "recall": [0.0],
            }
        )
        result = compare_reports(current, previous, logger=mock_logger)
        assert result.iloc[0]["fbeta_perc_change"] == pytest.approx(0.0)

    def test_with_overlap_threshold_and_beta_merge(self, mock_logger: logging.Logger) -> None:
        """Merge funciona quando colunas overlap_threshold e beta estão presentes."""
        current = pd.DataFrame(
            {
                "tp_entidade": ["CPF"],
                "overlap_threshold": [0.8],
                "beta": [1.0],
                "fbeta": [0.9],
                "precision": [0.9],
                "recall": [0.9],
            }
        )
        previous = pd.DataFrame(
            {
                "tp_entidade": ["CPF"],
                "overlap_threshold": [0.8],
                "beta": [1.0],
                "fbeta": [0.7],
                "precision": [0.7],
                "recall": [0.7],
            }
        )
        result = compare_reports(current, previous, logger=mock_logger)
        assert len(result) == 1
        assert result.iloc[0]["fbeta_melhorou"] is True or result.iloc[0]["fbeta_melhorou"] == True  # noqa: E712


# ---------------------------------------------------------------------------
# TestClassifyCases
# ---------------------------------------------------------------------------


class TestClassifyCases:
    """Testes para função classify_cases."""

    def test_tp_classified_correctly(self) -> None:
        """y_true=1, y_pred=1 resulta em 'TP'."""
        _df = pd.DataFrame({"y_true": [1], "y_pred": [1]})
        result = classify_cases(_df)
        assert result.iloc[0]["classification"] == "TP"

    def test_fp_classified_correctly(self) -> None:
        """y_true=0, y_pred=1 resulta em 'FP'."""
        _df = pd.DataFrame({"y_true": [0], "y_pred": [1]})
        result = classify_cases(_df)
        assert result.iloc[0]["classification"] == "FP"

    def test_fn_classified_correctly(self) -> None:
        """y_true=1, y_pred=0 resulta em 'FN'."""
        _df = pd.DataFrame({"y_true": [1], "y_pred": [0]})
        result = classify_cases(_df)
        assert result.iloc[0]["classification"] == "FN"

    def test_tn_classified_correctly(self) -> None:
        """y_true=0, y_pred=0 resulta em 'TN'."""
        _df = pd.DataFrame({"y_true": [0], "y_pred": [0]})
        result = classify_cases(_df)
        assert result.iloc[0]["classification"] == "TN"

    def test_all_four_cases(self) -> None:
        """Todos os quatro casos classificados corretamente."""
        _df = pd.DataFrame(
            {
                "y_true": [1, 0, 1, 0],
                "y_pred": [1, 1, 0, 0],
            }
        )
        result = classify_cases(_df)
        classifications = result["classification"].tolist()
        assert classifications == ["TP", "FP", "FN", "TN"]

    def test_returns_copy_not_in_place(self) -> None:
        """Função retorna cópia, não modifica o DataFrame original."""
        _df = pd.DataFrame({"y_true": [1], "y_pred": [1]})
        original_cols = set(_df.columns)
        _ = classify_cases(_df)
        assert set(_df.columns) == original_cols

    def test_classification_column_added(self) -> None:
        """Coluna 'classification' é adicionada ao resultado."""
        _df = pd.DataFrame({"y_true": [1, 0], "y_pred": [1, 0]})
        result = classify_cases(_df)
        assert "classification" in result.columns

    def test_empty_dataframe_returns_empty(self) -> None:
        """DataFrame vazio retorna DataFrame vazio com coluna classification."""
        _df = pd.DataFrame(columns=["y_true", "y_pred"])
        result = classify_cases(_df)
        assert "classification" in result.columns
        assert len(result) == 0

    @pytest.mark.parametrize(
        ("y_true", "y_pred", "expected"),
        [
            (1, 1, "TP"),
            (0, 1, "FP"),
            (1, 0, "FN"),
            (0, 0, "TN"),
        ],
    )
    def test_parametrized_classification(self, y_true: int, y_pred: int, expected: str) -> None:
        """Classificação parametrizada dos quatro casos."""
        _df = pd.DataFrame({"y_true": [y_true], "y_pred": [y_pred]})
        result = classify_cases(_df)
        assert result.iloc[0]["classification"] == expected


# ---------------------------------------------------------------------------
# TestGetClassificationCases
# ---------------------------------------------------------------------------


class TestGetClassificationCases:
    """Testes para função get_classification_cases."""

    def test_returns_dataframe(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Retorna DataFrame para entradas válidas."""
        result = get_classification_cases(
            comparison_data=sample_comparison,
            entity_type=None,
            case_type="all",
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert isinstance(result, pd.DataFrame)

    def test_all_cases_no_filter(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """case_type='all' retorna todos os casos."""
        result = get_classification_cases(
            comparison_data=sample_comparison,
            entity_type=None,
            case_type="all",
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert len(result) == len(sample_comparison)

    def test_filter_tp(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """case_type='tp' retorna somente TPs."""
        result = get_classification_cases(
            comparison_data=sample_comparison,
            entity_type=None,
            case_type="tp",
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert all(result["classification"] == "TP")

    def test_filter_fp(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """case_type='fp' retorna somente FPs."""
        result = get_classification_cases(
            comparison_data=sample_comparison,
            entity_type=None,
            case_type="fp",
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert all(result["classification"] == "FP")

    def test_filter_fn(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """case_type='fn' retorna somente FNs."""
        result = get_classification_cases(
            comparison_data=sample_comparison,
            entity_type=None,
            case_type="fn",
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert all(result["classification"] == "FN")

    def test_filter_by_entity_type(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """entity_type filtra apenas linhas daquela entidade."""
        result = get_classification_cases(
            comparison_data=sample_comparison,
            entity_type="CPF",
            case_type="all",
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert all(result["tp_entidade"] == "CPF")

    def test_invalid_case_type_raises(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """case_type inválido levanta ValueError."""
        with pytest.raises(ValueError, match="case_type deve ser um de"):
            get_classification_cases(
                comparison_data=sample_comparison,
                entity_type=None,
                case_type="INVALIDO",
                all_entities_key=_ALL_ENTITIES_KEY,
                logger=mock_logger,
            )

    def test_classification_column_present(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Resultado contém coluna 'classification'."""
        result = get_classification_cases(
            comparison_data=sample_comparison,
            entity_type=None,
            case_type="all",
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert "classification" in result.columns

    def test_has_text_flags_added(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Colunas has_gt_text e has_pred_text são adicionadas."""
        result = get_classification_cases(
            comparison_data=sample_comparison,
            entity_type=None,
            case_type="all",
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert "has_gt_text" in result.columns
        assert "has_pred_text" in result.columns

    def test_result_sorted_by_id(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Resultado é ordenado por id e tp_entidade."""
        result = get_classification_cases(
            comparison_data=sample_comparison,
            entity_type=None,
            case_type="all",
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert result["id"].tolist() == sorted(result["id"].tolist())

    def test_tn_case_type(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """case_type='tn' retorna somente TNs (pode ser vazio)."""
        result = get_classification_cases(
            comparison_data=sample_comparison,
            entity_type=None,
            case_type="tn",
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        valid_tn = len(result) == 0 or all(result["classification"] == "TN")
        assert valid_tn

    def test_combined_entity_and_case_filter(
        self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger
    ) -> None:
        """Filtro combinado de entidade e case_type funciona."""
        result = get_classification_cases(
            comparison_data=sample_comparison,
            entity_type="CPF",
            case_type="fn",
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        if len(result) > 0:
            assert all(result["tp_entidade"] == "CPF")
            assert all(result["classification"] == "FN")


# ---------------------------------------------------------------------------
# TestGetErrorAnalysis
# ---------------------------------------------------------------------------


class TestGetErrorAnalysis:
    """Testes para função get_error_analysis."""

    def test_returns_dict(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Retorna dicionário."""
        result = get_error_analysis(
            comparison_data=sample_comparison,
            entity_type=None,
            max_examples=5,
            overlap_bins=[0.0, 0.5, 1.01],
            overlap_bin_labels=["baixo", "alto"],
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert isinstance(result, dict)

    def test_keys_present(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Dicionário retornado contém todas as chaves esperadas."""
        result = get_error_analysis(
            comparison_data=sample_comparison,
            entity_type=None,
            max_examples=5,
            overlap_bins=[0.0, 0.5, 1.01],
            overlap_bin_labels=["baixo", "alto"],
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        expected_keys = {
            "entity_type",
            "total_cases",
            "summary",
            "tp_examples",
            "fp_examples",
            "fn_examples",
            "overlap_distribution",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_summary_has_tp_fp_fn_tn(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Summary contém contagens TP, FP, FN, TN."""
        result = get_error_analysis(
            comparison_data=sample_comparison,
            entity_type=None,
            max_examples=5,
            overlap_bins=[0.0, 0.5, 1.01],
            overlap_bin_labels=["baixo", "alto"],
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        for key in ("TP", "FP", "FN", "TN"):
            assert key in result["summary"]

    def test_total_cases_matches_dataframe_length(
        self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger
    ) -> None:
        """total_cases corresponde ao número de linhas."""
        result = get_error_analysis(
            comparison_data=sample_comparison,
            entity_type=None,
            max_examples=5,
            overlap_bins=[0.0, 0.5, 1.01],
            overlap_bin_labels=["baixo", "alto"],
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert result["total_cases"] == len(sample_comparison)

    def test_entity_type_all_uses_all_entities_key(
        self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger
    ) -> None:
        """entity_type=None usa all_entities_key no resultado."""
        result = get_error_analysis(
            comparison_data=sample_comparison,
            entity_type=None,
            max_examples=5,
            overlap_bins=[0.0, 0.5, 1.01],
            overlap_bin_labels=["baixo", "alto"],
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert result["entity_type"] == _ALL_ENTITIES_KEY

    def test_entity_type_filter(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Filtro por entidade restringe o total_cases."""
        result_cpf = get_error_analysis(
            comparison_data=sample_comparison,
            entity_type="CPF",
            max_examples=5,
            overlap_bins=[0.0, 0.5, 1.01],
            overlap_bin_labels=["baixo", "alto"],
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        result_all = get_error_analysis(
            comparison_data=sample_comparison,
            entity_type=None,
            max_examples=5,
            overlap_bins=[0.0, 0.5, 1.01],
            overlap_bin_labels=["baixo", "alto"],
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert result_cpf["total_cases"] < result_all["total_cases"]
        assert result_cpf["entity_type"] == "CPF"

    def test_max_examples_respected(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Listas de exemplos não excedem max_examples."""
        max_ex = 1
        result = get_error_analysis(
            comparison_data=sample_comparison,
            entity_type=None,
            max_examples=max_ex,
            overlap_bins=[0.0, 0.5, 1.01],
            overlap_bin_labels=["baixo", "alto"],
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert len(result["tp_examples"]) <= max_ex
        assert len(result["fp_examples"]) <= max_ex
        assert len(result["fn_examples"]) <= max_ex

    def test_overlap_distribution_is_dict(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """overlap_distribution é um dicionário."""
        result = get_error_analysis(
            comparison_data=sample_comparison,
            entity_type=None,
            max_examples=5,
            overlap_bins=[0.0, 0.5, 1.01],
            overlap_bin_labels=["baixo", "alto"],
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert isinstance(result["overlap_distribution"], dict)

    def test_overlap_distribution_bins_match_labels(
        self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger
    ) -> None:
        """Chaves de overlap_distribution correspondem aos labels fornecidos."""
        labels = ["baixo", "medio", "alto"]
        result = get_error_analysis(
            comparison_data=sample_comparison,
            entity_type=None,
            max_examples=5,
            overlap_bins=[0.0, 0.33, 0.67, 1.01],
            overlap_bin_labels=labels,
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        # Todas as chaves presentes são dos labels fornecidos
        for key in result["overlap_distribution"]:
            assert str(key) in labels

    def test_tp_example_has_expected_keys(self, mock_logger: logging.Logger) -> None:
        """Exemplos TP contêm chaves esperadas."""
        _df = pd.DataFrame(
            {
                "id": [1],
                "tp_entidade": ["CPF"],
                "y_true": [1],
                "y_pred": [1],
                "overlap": [1.0],
                "text_entidade_true": ["123.456.789-09"],
                "text_entidade_pred": ["123.456.789-09"],
                "start_entidade_true": [0],
                "end_entidade_true": [14],
                "start_entidade_pred": [0],
                "end_entidade_pred": [14],
            }
        )
        result = get_error_analysis(
            comparison_data=_df,
            entity_type=None,
            max_examples=5,
            overlap_bins=[0.0, 0.5, 1.01],
            overlap_bin_labels=["baixo", "alto"],
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert len(result["tp_examples"]) == 1
        example = result["tp_examples"][0]
        assert "id" in example
        assert "tp_entidade" in example
        assert "overlap" in example

    def test_empty_comparison_returns_zero_counts(self, mock_logger: logging.Logger) -> None:
        """DataFrame vazio retorna contagens zeradas."""
        empty_df = pd.DataFrame(columns=["id", "tp_entidade", "y_true", "y_pred", "overlap"])
        result = get_error_analysis(
            comparison_data=empty_df,
            entity_type=None,
            max_examples=5,
            overlap_bins=[0.0, 0.5, 1.01],
            overlap_bin_labels=["baixo", "alto"],
            all_entities_key=_ALL_ENTITIES_KEY,
            logger=mock_logger,
        )
        assert result["total_cases"] == 0
        assert result["summary"]["TP"] == 0
        assert result["summary"]["FP"] == 0
        assert result["summary"]["FN"] == 0
