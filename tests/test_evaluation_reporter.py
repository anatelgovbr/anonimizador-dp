"""Testes unitários para módulo _evaluation/reporter.py.

Este módulo testa as funções de geração de relatórios e exportação de dados:
classify_case, get_classification_cases, get_error_analysis,
save_classification_cases e export_dataframe.
"""

import logging
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from anonimizar._evaluation.reporter import (
    classify_case,
    export_dataframe,
    get_classification_cases,
    get_error_analysis,
    save_classification_cases,
)

__all__ = [
    "TestClassifyCase",
    "TestExportDataframe",
    "TestGetClassificationCases",
    "TestGetErrorAnalysis",
    "TestSaveClassificationCases",
]

# Constantes para testes
_TP_LABEL = "TP"
_FP_LABEL = "FP"
_FN_LABEL = "FN"
_TN_LABEL = "TN"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_comparison() -> pd.DataFrame:
    """DataFrame de comparação com TP, FP, FN e TN."""
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
def mock_logger() -> logging.Logger:
    """Logger simples para testes."""
    return logging.getLogger("test_reporter")


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """DataFrame simples para testes de exportação."""
    return pd.DataFrame(
        {
            "tp_entidade": ["CPF", "EMAIL"],
            "fbeta": [0.9, 0.8],
            "precision": [0.95, 0.85],
            "recall": [0.85, 0.75],
        }
    )


# ---------------------------------------------------------------------------
# TestClassifyCase
# ---------------------------------------------------------------------------


class TestClassifyCase:
    """Testes para função classify_case."""

    def test_true_positive(self) -> None:
        """y_true=1, y_pred=1 resulta em 'TP'."""
        assert classify_case(1, 1) == _TP_LABEL

    def test_false_positive(self) -> None:
        """y_true=0, y_pred=1 resulta em 'FP'."""
        assert classify_case(0, 1) == _FP_LABEL

    def test_false_negative(self) -> None:
        """y_true=1, y_pred=0 resulta em 'FN'."""
        assert classify_case(1, 0) == _FN_LABEL

    def test_true_negative(self) -> None:
        """y_true=0, y_pred=0 resulta em 'TN'."""
        assert classify_case(0, 0) == _TN_LABEL

    @pytest.mark.parametrize(
        ("y_true", "y_pred", "expected"),
        [
            (1, 1, "TP"),
            (1, 0, "FN"),
            (0, 1, "FP"),
            (0, 0, "TN"),
        ],
    )
    def test_all_cases_parametrized(self, y_true: int, y_pred: int, expected: str) -> None:
        """Todos os quatro casos de classificação."""
        assert classify_case(y_true, y_pred) == expected


# ---------------------------------------------------------------------------
# TestGetClassificationCases
# ---------------------------------------------------------------------------


class TestGetClassificationCases:
    """Testes para função get_classification_cases."""

    def test_returns_all_cases_by_default(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Sem filtros retorna todos os casos classificados."""
        result = get_classification_cases(sample_comparison, logger=mock_logger)
        assert len(result) == len(sample_comparison)
        assert "classification" in result.columns

    def test_classification_column_values(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Coluna classification contém somente valores válidos."""
        result = get_classification_cases(sample_comparison, logger=mock_logger)
        valid_labels = {"TP", "FP", "FN", "TN"}
        assert set(result["classification"].unique()).issubset(valid_labels)

    def test_filter_by_entity_type(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Filtro por entidade retorna apenas linhas dessa entidade."""
        result = get_classification_cases(sample_comparison, entity_type="CPF", logger=mock_logger)
        assert all(result["tp_entidade"] == "CPF")

    def test_filter_by_case_type_tp(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Filtro por 'tp' retorna apenas TPs."""
        result = get_classification_cases(sample_comparison, case_type="tp", logger=mock_logger)
        assert all(result["classification"] == "TP")

    def test_filter_by_case_type_fp(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Filtro por 'fp' retorna apenas FPs."""
        result = get_classification_cases(sample_comparison, case_type="fp", logger=mock_logger)
        assert all(result["classification"] == "FP")

    def test_filter_by_case_type_fn(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Filtro por 'fn' retorna apenas FNs."""
        result = get_classification_cases(sample_comparison, case_type="fn", logger=mock_logger)
        assert all(result["classification"] == "FN")

    def test_invalid_case_type_raises(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """case_type inválido levanta ValueError."""
        with pytest.raises(ValueError, match="case_type deve ser um de"):
            get_classification_cases(sample_comparison, case_type="invalid", logger=mock_logger)

    def test_has_text_flags_added(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Colunas has_gt_text e has_pred_text são adicionadas."""
        result = get_classification_cases(sample_comparison, logger=mock_logger)
        assert "has_gt_text" in result.columns
        assert "has_pred_text" in result.columns

    def test_result_sorted_by_id_and_entity(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Resultado é ordenado por id e tp_entidade."""
        result = get_classification_cases(sample_comparison, logger=mock_logger)
        sorted_ids = result["id"].tolist()
        assert sorted_ids == sorted(sorted_ids)

    def test_none_logger_works(self, sample_comparison: pd.DataFrame) -> None:
        """Logger None não causa erro."""
        result = get_classification_cases(sample_comparison, logger=None)
        assert "classification" in result.columns

    def test_empty_dataframe(self, mock_logger: logging.Logger) -> None:
        """DataFrame vazio retorna vazio com coluna classification."""
        empty_df = pd.DataFrame(columns=["id", "tp_entidade", "y_true", "y_pred", "overlap"])
        result = get_classification_cases(empty_df, logger=mock_logger)
        assert len(result) == 0
        assert "classification" in result.columns


# ---------------------------------------------------------------------------
# TestGetErrorAnalysis
# ---------------------------------------------------------------------------


class TestGetErrorAnalysis:
    """Testes para função get_error_analysis."""

    def test_returns_dict(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Retorna dicionário com análise de erros."""
        result = get_error_analysis(sample_comparison, logger=mock_logger)
        assert isinstance(result, dict)

    def test_summary_keys_present(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Dicionário contém as chaves summary com TP/FP/FN/TN."""
        result = get_error_analysis(sample_comparison, logger=mock_logger)
        assert "summary" in result
        assert "TP" in result["summary"]
        assert "FP" in result["summary"]
        assert "FN" in result["summary"]
        assert "TN" in result["summary"]

    def test_examples_keys_present(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Dicionário contém chaves de exemplos."""
        result = get_error_analysis(sample_comparison, logger=mock_logger)
        assert "tp_examples" in result
        assert "fp_examples" in result
        assert "fn_examples" in result

    def test_entity_type_set_correctly(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """entity_type no resultado corresponde ao filtro aplicado."""
        result = get_error_analysis(sample_comparison, entity_type="CPF", logger=mock_logger)
        assert result["entity_type"] == "CPF"

    def test_total_cases_count(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """total_cases corresponde ao número de linhas na comparação."""
        result = get_error_analysis(sample_comparison, logger=mock_logger)
        assert result["total_cases"] == len(sample_comparison)

    def test_overlap_distribution_present(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Distribuição de overlap é gerada."""
        result = get_error_analysis(sample_comparison, logger=mock_logger)
        assert "overlap_distribution" in result
        assert isinstance(result["overlap_distribution"], dict)

    def test_filter_by_entity_type(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Filtro por entidade restringe análise."""
        result_cpf = get_error_analysis(sample_comparison, entity_type="CPF", logger=mock_logger)
        result_all = get_error_analysis(sample_comparison, logger=mock_logger)
        assert result_cpf["total_cases"] < result_all["total_cases"]

    def test_none_logger_works(self, sample_comparison: pd.DataFrame) -> None:
        """Logger None não causa erro."""
        result = get_error_analysis(sample_comparison, logger=None)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestSaveClassificationCases
# ---------------------------------------------------------------------------


class TestSaveClassificationCases:
    """Testes para função save_classification_cases."""

    def test_save_parquet(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Salva arquivo parquet corretamente."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.parquet"
            save_classification_cases(sample_comparison, output_path=path, format="parquet", logger=mock_logger)
            assert path.exists()
            loaded = pd.read_parquet(path)
            assert len(loaded) == len(sample_comparison)

    def test_save_csv(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Salva arquivo CSV corretamente."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.csv"
            save_classification_cases(sample_comparison, output_path=path, format="csv", logger=mock_logger)
            assert path.exists()
            loaded = pd.read_csv(path)
            assert len(loaded) == len(sample_comparison)

    def test_save_json(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Salva arquivo JSON corretamente."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.json"
            save_classification_cases(sample_comparison, output_path=path, format="json", logger=mock_logger)
            assert path.exists()

    def test_unsupported_format_raises(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Formato não suportado levanta ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.xml"
            with pytest.raises(ValueError, match="Formato não suportado"):
                save_classification_cases(sample_comparison, output_path=path, format="xml", logger=mock_logger)

    def test_filter_by_entity_type(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Filtro por entidade salva apenas linhas filtradas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cpf.parquet"
            save_classification_cases(sample_comparison, output_path=path, entity_type="CPF", logger=mock_logger)
            loaded = pd.read_parquet(path)
            assert all(loaded["tp_entidade"] == "CPF")

    def test_creates_parent_directory(self, sample_comparison: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Cria diretório pai se não existir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "nested" / "dir" / "cases.parquet"
            save_classification_cases(sample_comparison, output_path=nested, logger=mock_logger)
            assert nested.exists()


# ---------------------------------------------------------------------------
# TestExportDataframe
# ---------------------------------------------------------------------------


class TestExportDataframe:
    """Testes para função export_dataframe."""

    def test_export_parquet(self, simple_df: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Exporta DataFrame como parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.parquet"
            export_dataframe(simple_df, output_path=path, format="parquet", logger=mock_logger)
            assert path.exists()
            loaded = pd.read_parquet(path)
            assert list(loaded.columns) == list(simple_df.columns)

    def test_export_csv(self, simple_df: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Exporta DataFrame como CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.csv"
            export_dataframe(simple_df, output_path=path, format="csv", logger=mock_logger)
            assert path.exists()
            loaded = pd.read_csv(path)
            assert len(loaded) == len(simple_df)

    def test_export_json(self, simple_df: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Exporta DataFrame como JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            export_dataframe(simple_df, output_path=path, format="json", logger=mock_logger)
            assert path.exists()

    def test_export_unsupported_format_raises(self, simple_df: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Formato não suportado levanta ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.xml"
            with pytest.raises(ValueError, match="Formato não suportado"):
                export_dataframe(simple_df, output_path=path, format="xml", logger=mock_logger)

    def test_none_logger_works(self, simple_df: pd.DataFrame) -> None:
        """Logger None não causa erro."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.parquet"
            export_dataframe(simple_df, output_path=path, logger=None)
            assert path.exists()

    def test_default_format_is_parquet(self, simple_df: pd.DataFrame, mock_logger: logging.Logger) -> None:
        """Formato padrão é parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.parquet"
            export_dataframe(simple_df, output_path=path, logger=mock_logger)
            assert path.exists()
            loaded = pd.read_parquet(path)
            assert len(loaded) == len(simple_df)
