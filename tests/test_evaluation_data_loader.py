"""Testes unitários para módulo _evaluation/data_loader.py.

Este módulo testa as funções de carregamento e validação de dados
para avaliação de modelos NER:
load_data, load_data_from_files e set_predictions.
"""

import logging
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from anonimizar._evaluation.data_loader import (
    load_data,
    load_data_from_files,
    set_predictions,
)

__all__ = [
    "TestLoadData",
    "TestLoadDataFromFiles",
    "TestSetPredictions",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_logger() -> logging.Logger:
    """Logger simples para testes."""
    return logging.getLogger("test_data_loader")


@pytest.fixture
def valid_texts() -> pd.DataFrame:
    """DataFrame de textos válido."""
    return pd.DataFrame({"id": [1, 2, 3], "text": ["Texto 1", "Texto 2", "Texto 3"]})


@pytest.fixture
def valid_ground_truth() -> pd.DataFrame:
    """DataFrame de ground truth válido."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "tp_entidade": ["CPF", "EMAIL"],
            "start_entidade": [0, 5],
            "end_entidade": [11, 20],
        }
    )


@pytest.fixture
def valid_predictions() -> pd.DataFrame:
    """DataFrame de predições válido."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "tp_entidade": ["CPF", "EMAIL"],
            "start_entidade": [0, 5],
            "end_entidade": [11, 20],
        }
    )


# ---------------------------------------------------------------------------
# TestLoadData
# ---------------------------------------------------------------------------


class TestLoadData:
    """Testes para função load_data."""

    def test_valid_inputs_return_tuple(
        self,
        valid_texts: pd.DataFrame,
        valid_ground_truth: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Entradas válidas retornam tupla de dois DataFrames."""
        result = load_data(valid_texts, valid_ground_truth, logger=mock_logger)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_texts_unchanged(
        self,
        valid_texts: pd.DataFrame,
        valid_ground_truth: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """DataFrame de textos retornado tem mesmas linhas."""
        df_texts, _ = load_data(valid_texts, valid_ground_truth, logger=mock_logger)
        assert len(df_texts) == len(valid_texts)

    def test_returns_gt_unchanged(
        self,
        valid_texts: pd.DataFrame,
        valid_ground_truth: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """DataFrame de ground truth retornado tem mesmas linhas."""
        _, df_gt = load_data(valid_texts, valid_ground_truth, logger=mock_logger)
        assert len(df_gt) == len(valid_ground_truth)

    def test_missing_text_column_raises(
        self,
        valid_ground_truth: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """DataFrame de textos sem coluna 'text' levanta ValueError."""
        bad_texts = pd.DataFrame({"id": [1], "conteudo": ["texto"]})
        with pytest.raises(ValueError, match="Colunas ausentes em df_texts"):
            load_data(bad_texts, valid_ground_truth, logger=mock_logger)

    def test_missing_id_column_in_texts_raises(
        self,
        valid_ground_truth: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """DataFrame de textos sem coluna 'id' levanta ValueError."""
        bad_texts = pd.DataFrame({"text": ["texto1"]})
        with pytest.raises(ValueError, match="Colunas ausentes em df_texts"):
            load_data(bad_texts, valid_ground_truth, logger=mock_logger)

    def test_missing_tp_entidade_in_gt_raises(
        self,
        valid_texts: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Ground truth sem coluna 'tp_entidade' levanta ValueError."""
        bad_gt = pd.DataFrame({"id": [1], "start_entidade": [0], "end_entidade": [10]})
        with pytest.raises(ValueError, match="Colunas ausentes em df_ground_truth"):
            load_data(valid_texts, bad_gt, logger=mock_logger)

    def test_missing_start_end_in_gt_raises(
        self,
        valid_texts: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Ground truth sem colunas de posição levanta ValueError."""
        bad_gt = pd.DataFrame({"id": [1], "tp_entidade": ["CPF"]})
        with pytest.raises(ValueError, match="Colunas ausentes em df_ground_truth"):
            load_data(valid_texts, bad_gt, logger=mock_logger)

    def test_empty_dataframes_pass_validation(
        self,
        mock_logger: logging.Logger,
    ) -> None:
        """DataFrames vazios com colunas corretas passam na validação."""
        empty_texts = pd.DataFrame(columns=["id", "text"])
        empty_gt = pd.DataFrame(columns=["id", "tp_entidade", "start_entidade", "end_entidade"])
        df_texts, df_gt = load_data(empty_texts, empty_gt, logger=mock_logger)
        assert len(df_texts) == 0
        assert len(df_gt) == 0

    def test_extra_columns_are_kept(
        self,
        mock_logger: logging.Logger,
    ) -> None:
        """Colunas extras além das obrigatórias são mantidas no resultado."""
        texts_with_extra = pd.DataFrame({"id": [1], "text": ["texto"], "source": ["web"]})
        gt_with_extra = pd.DataFrame(
            {
                "id": [1],
                "tp_entidade": ["CPF"],
                "start_entidade": [0],
                "end_entidade": [11],
                "text_entidade": ["123.456.789-09"],
            }
        )
        df_texts, df_gt = load_data(texts_with_extra, gt_with_extra, logger=mock_logger)
        assert "source" in df_texts.columns
        assert "text_entidade" in df_gt.columns


# ---------------------------------------------------------------------------
# TestLoadDataFromFiles
# ---------------------------------------------------------------------------


class TestLoadDataFromFiles:
    """Testes para função load_data_from_files."""

    def test_load_csv_files(
        self,
        valid_texts: pd.DataFrame,
        valid_ground_truth: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Carrega arquivos CSV válidos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            texts_path = Path(tmpdir) / "texts.csv"
            gt_path = Path(tmpdir) / "gt.csv"
            valid_texts.to_csv(texts_path, index=False)
            valid_ground_truth.to_csv(gt_path, index=False)

            df_texts, df_gt = load_data_from_files(
                texts_path=texts_path,
                ground_truth_path=gt_path,
                logger=mock_logger,
            )
            assert len(df_texts) == len(valid_texts)
            assert len(df_gt) == len(valid_ground_truth)

    def test_load_parquet_files(
        self,
        valid_texts: pd.DataFrame,
        valid_ground_truth: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Carrega arquivos Parquet válidos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            texts_path = Path(tmpdir) / "texts.parquet"
            gt_path = Path(tmpdir) / "gt.parquet"
            valid_texts.to_parquet(texts_path, index=False)
            valid_ground_truth.to_parquet(gt_path, index=False)

            df_texts, df_gt = load_data_from_files(
                texts_path=texts_path,
                ground_truth_path=gt_path,
                logger=mock_logger,
            )
            assert len(df_texts) == len(valid_texts)
            assert len(df_gt) == len(valid_ground_truth)

    def test_texts_file_not_found_raises(
        self,
        valid_ground_truth: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Arquivo de textos inexistente levanta FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = Path(tmpdir) / "gt.parquet"
            valid_ground_truth.to_parquet(gt_path, index=False)

            with pytest.raises(FileNotFoundError):
                load_data_from_files(
                    texts_path=Path(tmpdir) / "nonexistent.parquet",
                    ground_truth_path=gt_path,
                    logger=mock_logger,
                )

    def test_gt_file_not_found_raises(
        self,
        valid_texts: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Arquivo de ground truth inexistente levanta FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            texts_path = Path(tmpdir) / "texts.parquet"
            valid_texts.to_parquet(texts_path, index=False)

            with pytest.raises(FileNotFoundError):
                load_data_from_files(
                    texts_path=texts_path,
                    ground_truth_path=Path(tmpdir) / "nonexistent.parquet",
                    logger=mock_logger,
                )

    def test_unsupported_texts_format_raises(
        self,
        valid_ground_truth: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Formato de arquivo de textos não suportado levanta ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Criar arquivo com extensão não suportada
            texts_path = Path(tmpdir) / "texts.json"
            texts_path.write_text("{}")
            gt_path = Path(tmpdir) / "gt.parquet"
            valid_ground_truth.to_parquet(gt_path, index=False)

            with pytest.raises(ValueError, match="Formato de arquivo não suportado"):
                load_data_from_files(
                    texts_path=texts_path,
                    ground_truth_path=gt_path,
                    logger=mock_logger,
                )

    def test_unsupported_gt_format_raises(
        self,
        valid_texts: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Formato de arquivo de ground truth não suportado levanta ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            texts_path = Path(tmpdir) / "texts.parquet"
            valid_texts.to_parquet(texts_path, index=False)
            gt_path = Path(tmpdir) / "gt.xlsx"
            gt_path.write_text("dummy")

            with pytest.raises(ValueError, match="Formato de arquivo não suportado"):
                load_data_from_files(
                    texts_path=texts_path,
                    ground_truth_path=gt_path,
                    logger=mock_logger,
                )

    def test_string_paths_accepted(
        self,
        valid_texts: pd.DataFrame,
        valid_ground_truth: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Caminhos como strings também são aceitos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            texts_path = Path(tmpdir) / "texts.csv"
            gt_path = Path(tmpdir) / "gt.csv"
            valid_texts.to_csv(texts_path, index=False)
            valid_ground_truth.to_csv(gt_path, index=False)

            df_texts, _df_gt = load_data_from_files(
                texts_path=str(texts_path),
                ground_truth_path=str(gt_path),
                logger=mock_logger,
            )
            assert len(df_texts) == len(valid_texts)

    def test_validates_columns_after_loading(
        self,
        mock_logger: logging.Logger,
    ) -> None:
        """Valida colunas dos arquivos carregados — erro se obrigatórias ausentes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Arquivo de textos sem coluna 'text'
            bad_texts = pd.DataFrame({"id": [1], "conteudo": ["x"]})
            valid_gt = pd.DataFrame({"id": [1], "tp_entidade": ["CPF"], "start_entidade": [0], "end_entidade": [11]})
            texts_path = Path(tmpdir) / "texts.csv"
            gt_path = Path(tmpdir) / "gt.csv"
            bad_texts.to_csv(texts_path, index=False)
            valid_gt.to_csv(gt_path, index=False)

            with pytest.raises(ValueError, match="Colunas ausentes"):
                load_data_from_files(
                    texts_path=texts_path,
                    ground_truth_path=gt_path,
                    logger=mock_logger,
                )


# ---------------------------------------------------------------------------
# TestSetPredictions
# ---------------------------------------------------------------------------


class TestSetPredictions:
    """Testes para função set_predictions."""

    def test_valid_predictions_returned(
        self,
        valid_predictions: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Predições válidas são retornadas sem modificação."""
        result = set_predictions(valid_predictions, logger=mock_logger)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(valid_predictions)

    def test_missing_id_column_raises(
        self,
        mock_logger: logging.Logger,
    ) -> None:
        """Predições sem coluna 'id' levanta ValueError."""
        bad_preds = pd.DataFrame(
            {
                "tp_entidade": ["CPF"],
                "start_entidade": [0],
                "end_entidade": [11],
            }
        )
        with pytest.raises(ValueError, match="Colunas ausentes"):
            set_predictions(bad_preds, logger=mock_logger)

    def test_missing_tp_entidade_raises(
        self,
        mock_logger: logging.Logger,
    ) -> None:
        """Predições sem coluna 'tp_entidade' levanta ValueError."""
        bad_preds = pd.DataFrame({"id": [1], "start_entidade": [0], "end_entidade": [11]})
        with pytest.raises(ValueError, match="Colunas ausentes"):
            set_predictions(bad_preds, logger=mock_logger)

    def test_missing_start_end_raises(
        self,
        mock_logger: logging.Logger,
    ) -> None:
        """Predições sem colunas de posição levantam ValueError."""
        bad_preds = pd.DataFrame({"id": [1], "tp_entidade": ["CPF"]})
        with pytest.raises(ValueError, match="Colunas ausentes"):
            set_predictions(bad_preds, logger=mock_logger)

    def test_empty_predictions_accepted(
        self,
        mock_logger: logging.Logger,
    ) -> None:
        """DataFrame vazio com colunas corretas passa na validação."""
        empty_preds = pd.DataFrame(columns=["id", "tp_entidade", "start_entidade", "end_entidade"])
        result = set_predictions(empty_preds, logger=mock_logger)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_extra_columns_preserved(
        self,
        mock_logger: logging.Logger,
    ) -> None:
        """Colunas extras são preservadas no resultado."""
        preds_with_extra = pd.DataFrame(
            {
                "id": [1],
                "tp_entidade": ["CPF"],
                "start_entidade": [0],
                "end_entidade": [11],
                "text_entidade": ["123.456.789-09"],
                "detected_by": ["regex"],
            }
        )
        result = set_predictions(preds_with_extra, logger=mock_logger)
        assert "text_entidade" in result.columns
        assert "detected_by" in result.columns

    def test_returns_same_dataframe(
        self,
        valid_predictions: pd.DataFrame,
        mock_logger: logging.Logger,
    ) -> None:
        """Função retorna o DataFrame recebido (ou uma cópia com mesmos dados)."""
        result = set_predictions(valid_predictions, logger=mock_logger)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), valid_predictions.reset_index(drop=True))
