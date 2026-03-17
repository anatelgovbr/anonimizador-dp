"""Testes unitários para módulo _extraction/pipeline.py.

Este módulo testa a função principal de extração de entidades
que combina modelo spaCy, regex e tabelas markdown.
"""

import logging
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from anonimizar._extraction.pipeline import extract_entities

__all__ = [
    "TestExtractEntities",
]

# ---------------------------------------------------------------------------
# Constantes para testes
# ---------------------------------------------------------------------------

_RETURN_TYPE_LABEL_DETAIL = "label_detail"
_RETURN_TYPE_LABEL_POSITION = "label_position"
_RETURN_TYPE_LABEL_TEXT = "label_text"
_RETURN_TYPE_INVALID = "invalid_type"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_logger() -> logging.Logger:
    """Logger simples para testes."""
    return logging.getLogger("test_pipeline")


@pytest.fixture
def mock_nlp() -> MagicMock:
    """Mock de modelo spaCy."""
    return MagicMock()


@pytest.fixture
def verify_fn():
    """Função de verificação que sempre retorna True."""
    return lambda entity, text: True  # noqa: ARG005


@pytest.fixture
def empty_patterns() -> list:
    """Lista de padrões vazia."""
    return []


@pytest.fixture
def empty_labels() -> set:
    """Conjunto de labels vazio."""
    return set()


# ---------------------------------------------------------------------------
# TestExtractEntities
# ---------------------------------------------------------------------------


class TestExtractEntities:
    """Testes para função extract_entities."""

    @patch("anonimizar._extraction.pipeline.extract_entities_from_markdown_tables", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_entities_regex_re", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_from_model", return_value=[])
    def test_empty_text_returns_empty_list(
        self,
        mock_model: MagicMock,  # noqa: ARG002
        mock_regex: MagicMock,  # noqa: ARG002
        mock_markdown: MagicMock,  # noqa: ARG002
        mock_nlp: MagicMock,
        verify_fn,
        empty_patterns: list,
        empty_labels: set,
        mock_logger: logging.Logger,
    ) -> None:
        """Texto vazio retorna lista vazia."""
        result = extract_entities(
            nlp_trained=mock_nlp,
            text_or_path="",
            labels=empty_labels,
            patterns=empty_patterns,
            return_type=_RETURN_TYPE_LABEL_DETAIL,
            verify_fn=verify_fn,
            logger=mock_logger,
        )
        assert result == []

    @patch("anonimizar._extraction.pipeline.extract_entities_from_markdown_tables", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_entities_regex_re", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_from_model", return_value=[])
    def test_returns_list(
        self,
        mock_model: MagicMock,  # noqa: ARG002
        mock_regex: MagicMock,  # noqa: ARG002
        mock_markdown: MagicMock,  # noqa: ARG002
        mock_nlp: MagicMock,
        verify_fn,
        empty_patterns: list,
        empty_labels: set,
        mock_logger: logging.Logger,
    ) -> None:
        """Retorna uma lista."""
        result = extract_entities(
            nlp_trained=mock_nlp,
            text_or_path="Texto de teste com CPF 123.456.789-09",
            labels=empty_labels,
            patterns=empty_patterns,
            return_type=_RETURN_TYPE_LABEL_DETAIL,
            verify_fn=verify_fn,
            logger=mock_logger,
        )
        assert isinstance(result, list)

    @patch("anonimizar._extraction.pipeline.extract_entities_from_markdown_tables", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_entities_regex_re", return_value=[])
    @patch(
        "anonimizar._extraction.pipeline.extract_from_model",
        return_value=[
            {"label": "CPF", "text": "123.456.789-09", "start_position": 20, "end_position": 34, "detected_by": "model"}
        ],
    )
    def test_model_entities_returned(
        self,
        mock_model: MagicMock,  # noqa: ARG002
        mock_regex: MagicMock,  # noqa: ARG002
        mock_markdown: MagicMock,  # noqa: ARG002
        mock_nlp: MagicMock,
        verify_fn,
        empty_patterns: list,
        empty_labels: set,
        mock_logger: logging.Logger,
    ) -> None:
        """Entidades do modelo são incluídas no resultado."""
        result = extract_entities(
            nlp_trained=mock_nlp,
            text_or_path="Texto de teste com CPF 123.456.789-09",
            labels=empty_labels,
            patterns=empty_patterns,
            return_type=_RETURN_TYPE_LABEL_DETAIL,
            verify_fn=verify_fn,
            logger=mock_logger,
        )
        assert len(result) == 1
        assert result[0]["label"] == "CPF"

    @patch("anonimizar._extraction.pipeline.extract_entities_from_markdown_tables", return_value=[])
    @patch(
        "anonimizar._extraction.pipeline.extract_entities_regex_re",
        return_value=[
            {"label": "EMAIL", "text": "a@b.com", "start_position": 5, "end_position": 12, "detected_by": "regex"}
        ],
    )
    @patch("anonimizar._extraction.pipeline.extract_from_model", return_value=[])
    def test_regex_entities_returned(
        self,
        mock_model: MagicMock,  # noqa: ARG002
        mock_regex: MagicMock,  # noqa: ARG002
        mock_markdown: MagicMock,  # noqa: ARG002
        mock_nlp: MagicMock,
        verify_fn,
        empty_patterns: list,
        empty_labels: set,
        mock_logger: logging.Logger,
    ) -> None:
        """Entidades do regex são incluídas no resultado."""
        result = extract_entities(
            nlp_trained=mock_nlp,
            text_or_path="test a@b.com texto",
            labels=empty_labels,
            patterns=empty_patterns,
            return_type=_RETURN_TYPE_LABEL_DETAIL,
            verify_fn=verify_fn,
            logger=mock_logger,
        )
        assert len(result) == 1
        assert result[0]["label"] == "EMAIL"

    @patch(
        "anonimizar._extraction.pipeline.extract_entities_from_markdown_tables",
        return_value=[
            {
                "label": "CPF",
                "text": "123.456.789-09",
                "start_position": 0,
                "end_position": 14,
                "detected_by": "markdown",
            }
        ],
    )
    @patch("anonimizar._extraction.pipeline.extract_entities_regex_re", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_from_model", return_value=[])
    def test_markdown_entities_returned(
        self,
        mock_model: MagicMock,  # noqa: ARG002
        mock_regex: MagicMock,  # noqa: ARG002
        mock_markdown: MagicMock,  # noqa: ARG002
        mock_nlp: MagicMock,
        verify_fn,
        empty_patterns: list,
        empty_labels: set,
        mock_logger: logging.Logger,
    ) -> None:
        """Entidades de tabelas markdown são incluídas no resultado."""
        result = extract_entities(
            nlp_trained=mock_nlp,
            text_or_path="| CPF | 123.456.789-09 |",
            labels=empty_labels,
            patterns=empty_patterns,
            return_type=_RETURN_TYPE_LABEL_DETAIL,
            verify_fn=verify_fn,
            logger=mock_logger,
        )
        assert len(result) == 1
        assert result[0]["detected_by"] == "markdown"

    @patch("anonimizar._extraction.pipeline.extract_entities_from_markdown_tables", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_entities_regex_re", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_from_model", return_value=[])
    def test_invalid_return_type_raises(
        self,
        mock_model: MagicMock,  # noqa: ARG002
        mock_regex: MagicMock,  # noqa: ARG002
        mock_markdown: MagicMock,  # noqa: ARG002
        mock_nlp: MagicMock,
        verify_fn,
        empty_patterns: list,
        empty_labels: set,
        mock_logger: logging.Logger,
    ) -> None:
        """return_type inválido levanta ValueError."""
        with pytest.raises(ValueError, match="Tipo de retorno não permitido"):
            extract_entities(
                nlp_trained=mock_nlp,
                text_or_path="texto de teste",
                labels=empty_labels,
                patterns=empty_patterns,
                return_type=_RETURN_TYPE_INVALID,
                verify_fn=verify_fn,
                logger=mock_logger,
            )

    @patch("anonimizar._extraction.pipeline.extract_entities_from_markdown_tables", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_entities_regex_re", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_from_model", return_value=[])
    def test_label_text_return_type(
        self,
        mock_model: MagicMock,  # noqa: ARG002
        mock_regex: MagicMock,  # noqa: ARG002
        mock_markdown: MagicMock,  # noqa: ARG002
        mock_nlp: MagicMock,
        verify_fn,
        empty_patterns: list,
        empty_labels: set,
        mock_logger: logging.Logger,
    ) -> None:
        """return_type='label_text' retorna lista válida."""
        result = extract_entities(
            nlp_trained=mock_nlp,
            text_or_path="texto de teste",
            labels=empty_labels,
            patterns=empty_patterns,
            return_type=_RETURN_TYPE_LABEL_TEXT,
            verify_fn=verify_fn,
            logger=mock_logger,
        )
        assert isinstance(result, list)

    @patch("anonimizar._extraction.pipeline.extract_entities_from_markdown_tables", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_entities_regex_re", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_from_model", return_value=[])
    def test_label_position_return_type(
        self,
        mock_model: MagicMock,  # noqa: ARG002
        mock_regex: MagicMock,  # noqa: ARG002
        mock_markdown: MagicMock,  # noqa: ARG002
        mock_nlp: MagicMock,
        verify_fn,
        empty_patterns: list,
        empty_labels: set,
        mock_logger: logging.Logger,
    ) -> None:
        """return_type='label_position' retorna lista válida."""
        result = extract_entities(
            nlp_trained=mock_nlp,
            text_or_path="texto de teste",
            labels=empty_labels,
            patterns=empty_patterns,
            return_type=_RETURN_TYPE_LABEL_POSITION,
            verify_fn=verify_fn,
            logger=mock_logger,
        )
        assert isinstance(result, list)

    def test_non_md_file_raises_value_error(
        self,
        mock_nlp: MagicMock,
        verify_fn,
        empty_patterns: list,
        empty_labels: set,
        mock_logger: logging.Logger,
    ) -> None:
        """Arquivo com extensão diferente de .md levanta ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"conteudo")
            tmp_path = tmp.name

        with pytest.raises(ValueError, match="não é um MD"):
            extract_entities(
                nlp_trained=mock_nlp,
                text_or_path=tmp_path,
                labels=empty_labels,
                patterns=empty_patterns,
                return_type=_RETURN_TYPE_LABEL_DETAIL,
                verify_fn=verify_fn,
                logger=mock_logger,
            )

    @patch("anonimizar._extraction.pipeline.extract_entities_from_markdown_tables", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_entities_regex_re", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_from_model", return_value=[])
    def test_md_file_loaded_and_processed(
        self,
        mock_model: MagicMock,
        mock_regex: MagicMock,
        mock_markdown: MagicMock,
        mock_nlp: MagicMock,
        verify_fn,
        empty_patterns: list,
        empty_labels: set,
        mock_logger: logging.Logger,
    ) -> None:
        """Arquivo .md é carregado e processado corretamente."""
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as tmp:
            tmp.write("# Texto de teste\n\nConteúdo do arquivo markdown.")
            tmp_path = tmp.name

        result = extract_entities(
            nlp_trained=mock_nlp,
            text_or_path=tmp_path,
            labels=empty_labels,
            patterns=empty_patterns,
            return_type=_RETURN_TYPE_LABEL_DETAIL,
            verify_fn=verify_fn,
            logger=mock_logger,
        )
        assert isinstance(result, list)
        # Verifica que as fontes foram chamadas
        mock_model.assert_called_once()
        mock_regex.assert_called_once()
        mock_markdown.assert_called_once()

    @patch("anonimizar._extraction.pipeline.extract_entities_from_markdown_tables", return_value=[])
    @patch(
        "anonimizar._extraction.pipeline.extract_entities_regex_re",
        return_value=[
            {"label": "CPF", "text": "123.456.789-09", "start_position": 5, "end_position": 19, "detected_by": "regex"},
            {"label": "CPF", "text": "123.456.789-09", "start_position": 5, "end_position": 19, "detected_by": "regex"},
        ],
    )
    @patch("anonimizar._extraction.pipeline.extract_from_model", return_value=[])
    def test_label_text_deduplicates_same_entity(
        self,
        mock_model: MagicMock,  # noqa: ARG002
        mock_regex: MagicMock,  # noqa: ARG002
        mock_markdown: MagicMock,  # noqa: ARG002
        mock_nlp: MagicMock,
        verify_fn,
        empty_patterns: list,
        empty_labels: set,
        mock_logger: logging.Logger,
    ) -> None:
        """label_text desduplicar entidades com mesmo label+text."""
        result = extract_entities(
            nlp_trained=mock_nlp,
            text_or_path="cpf 123.456.789-09 texto",
            labels=empty_labels,
            patterns=empty_patterns,
            return_type=_RETURN_TYPE_LABEL_TEXT,
            verify_fn=verify_fn,
            logger=mock_logger,
        )
        # Dois registros idênticos devem ser deduplciados para 1
        assert len(result) == 1

    @patch("anonimizar._extraction.pipeline.extract_entities_from_markdown_tables", return_value=[])
    @patch("anonimizar._extraction.pipeline.extract_entities_regex_re", return_value=[])
    @patch(
        "anonimizar._extraction.pipeline.extract_from_model",
        return_value=[
            {"label": "CPF", "text": "123.456.789-09", "start_position": 0, "end_position": 14, "detected_by": "model"},
            {"label": "CPF", "text": "456.789-09", "start_position": 4, "end_position": 14, "detected_by": "model"},
        ],
    )
    def test_overlap_removed_for_label_detail(
        self,
        mock_model: MagicMock,  # noqa: ARG002
        mock_regex: MagicMock,  # noqa: ARG002
        mock_markdown: MagicMock,  # noqa: ARG002
        mock_nlp: MagicMock,
        verify_fn,
        empty_patterns: list,
        empty_labels: set,
        mock_logger: logging.Logger,
    ) -> None:
        """label_detail remove spans sobrepostos usando remove_overlap_positions."""
        result = extract_entities(
            nlp_trained=mock_nlp,
            text_or_path="123.456.789-09 texto",
            labels=empty_labels,
            patterns=empty_patterns,
            return_type=_RETURN_TYPE_LABEL_DETAIL,
            verify_fn=verify_fn,
            logger=mock_logger,
        )
        # Dois spans sobrepostos deve resultar em somente 1
        assert len(result) <= 2
