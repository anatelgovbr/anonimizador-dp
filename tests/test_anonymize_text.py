import logging
from unittest.mock import MagicMock

from anonimizar._anonymization.text import _apply_substitutions, anonymize_text
from tests._helpers import _make_logger


class TestAnonymization:
    """Testes para anonimização de texto."""

    def test_anonymize_text_basic(self, anonymizer):
        text = "João, CPF 123.456.789-09, email: joao@email.com"
        entities = anonymizer.extract_entities(text, return_type="label_position")

        anonymized = anonymizer.anonymize_text(text, entities)

        assert "<|CPF|>" in anonymized
        assert "<|EMAIL|>" in anonymized
        assert "123.456.789-09" not in anonymized
        assert "joao@email.com" not in anonymized

    def test_anonymize_text_empty_entities(self, anonymizer):
        text = "Texto sem entidades sensíveis"
        anonymized = anonymizer.anonymize_text(text, [])
        assert anonymized == text

    def test_anonymize_text_preserves_prefixes_outside_documented_spans(self, anonymizer):
        text = "João, CPF 123.456.789-09, email: joao@email.com"
        entities = [
            {"label": "CPF", "start_position": 10, "end_position": 24},
            {"label": "EMAIL", "start_position": 33, "end_position": 47},
        ]

        assert anonymizer.anonymize_text(text, entities) == "João, CPF <|CPF|>, email: <|EMAIL|>"

    def test_documented_example_offsets_and_tags(self):
        """Mantém reproduzível o exemplo da docstring de anonymize_text."""
        text = "João, CPF 123.456.789-00, email: joao@email.com"
        entities = [
            {"label": "CPF", "start_position": 10, "end_position": 24},
            {"label": "EMAIL", "start_position": 33, "end_position": 47},
        ]

        assert text[10:24] == "123.456.789-00"
        assert text[33:47] == "joao@email.com"
        assert anonymize_text(text, entities, _make_logger()) == "João, CPF <|CPF|>, email: <|EMAIL|>"


class TestB26BoundsValidation:
    """B-26: _apply_substitutions deve ignorar posições inválidas em vez de falhar."""

    def test_start_negativo_ignorado(self):
        text = "texto normal"
        result = _apply_substitutions(text, [(-1, 5, "CPF")], _make_logger())
        assert result == text

    def test_end_maior_que_texto_ignorado(self):
        text = "texto"
        result = _apply_substitutions(text, [(0, 999, "CPF")], _make_logger())
        assert result == text

    def test_start_igual_end_ignorado(self):
        text = "texto normal"
        result = _apply_substitutions(text, [(5, 5, "CPF")], _make_logger())
        assert result == text

    def test_substituicao_valida_funciona(self):
        text = "cpf 123.456.789-09 aqui"
        result = _apply_substitutions(text, [(4, 18, "CPF")], _make_logger())
        assert "<|CPF|>" in result
        assert "123.456.789-09" not in result

    def test_mistura_validos_invalidos(self):
        text = "cpf 123.456.789-09 fim"
        substitutions = [
            (-1, 3, "CPF"),
            (4, 18, "CPF"),
            (0, 999, "CPF"),
        ]
        result = _apply_substitutions(text, substitutions, _make_logger())
        assert "<|CPF|>" in result
        assert "123.456.789-09" not in result

    def test_warning_emitido_para_bounds_invalidos(self):
        mock_logger = MagicMock(spec=logging.Logger)
        _apply_substitutions("texto", [(-1, 5, "CPF")], mock_logger)
        mock_logger.warning.assert_called_once()
