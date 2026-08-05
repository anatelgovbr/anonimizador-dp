"""Testes para métodos wrapper da classe Anonimizar (add_pattern_*, get_entity_attribute)."""

from unittest.mock import MagicMock, patch

import pytest

from anonimizar import Anonimizar


@pytest.fixture
def mock_anonymizer():
    """Cria Anonimizar com spacy.load mockado."""
    with (
        patch("anonimizar._anonymization.anonymizer.spacy.load") as mock_spacy,
        patch("anonimizar._anonymization.anonymizer.create_default_logger") as mock_logger_factory,
    ):
        mock_model = MagicMock()
        mock_model.pipe_names = ["ner"]
        mock_model.get_pipe.return_value.labels = ("CPF", "EMAIL")
        mock_model.max_length = 3000000
        mock_spacy.return_value = mock_model
        mock_logger_factory.return_value = MagicMock()
        anon = Anonimizar(model_path="fake", auto_patterns=False)
        yield anon


class TestGetEntityAttribute:
    """Testes para get_entity_attribute."""

    def test_with_object(self, mock_anonymizer):
        class DummyEntity:
            label = "CPF"
            text = "123.456.789-09"

        entity = DummyEntity()
        result = mock_anonymizer.get_entity_attribute(entity, "label")
        assert result == "CPF"

    def test_with_dict(self, mock_anonymizer):
        entity = {"label": "EMAIL", "text": "teste@email.com"}
        result = mock_anonymizer.get_entity_attribute(entity, "label")
        assert result == "EMAIL"

    def test_missing_attr_raises(self, mock_anonymizer):
        entity = {"label": "CPF"}
        with pytest.raises(KeyError):
            mock_anonymizer.get_entity_attribute(entity, "inexistente")


_PB = "anonimizar._anonymization.anonymizer"

_ADD_PATTERN_CASES = [
    ("add_pattern_cid", f"{_PB}._bp_add_pattern_cid"),
    ("add_pattern_cpf", f"{_PB}._bp_add_pattern_cpf"),
    ("add_pattern_endereco", f"{_PB}._bp_add_pattern_endereco"),
    ("add_pattern_geo_coord", f"{_PB}._bp_add_pattern_geo_coord"),
    ("add_pattern_rg", f"{_PB}._bp_add_pattern_rg"),
    ("add_pattern_rg_estrangeiro", f"{_PB}._bp_add_pattern_rg_estrangeiro"),
    ("add_pattern_titulo_eleitor", f"{_PB}._bp_add_pattern_titulo_eleitor"),
    ("add_pattern_passaporte", f"{_PB}._bp_add_pattern_passaporte"),
    ("add_pattern_passaporte_est", f"{_PB}._bp_add_pattern_passaporte_est"),
    ("add_pattern_siape", f"{_PB}._bp_add_pattern_siape"),
    ("add_pattern_cnh", f"{_PB}._bp_add_pattern_cnh"),
    ("add_pattern_dados_bancarios", f"{_PB}._bp_add_pattern_dados_bancarios"),
    ("add_pattern_email", f"{_PB}._bp_add_pattern_email"),
    ("add_pattern_telefone", f"{_PB}._bp_add_pattern_telefone"),
    ("add_pattern_data_nascimento", f"{_PB}._bp_add_pattern_data_nascimento"),
]


class TestAddPatternWrappers:
    """Testa que cada add_pattern_* delega para a função _bp_* correspondente."""

    @pytest.mark.parametrize(("method_name", "patch_target"), _ADD_PATTERN_CASES)
    def test_add_pattern_delegates(self, mock_anonymizer, method_name, patch_target):
        with patch(patch_target) as mock_fn:
            getattr(mock_anonymizer, method_name)()
            mock_fn.assert_called_once_with(mock_anonymizer.patterns, mock_anonymizer.logger)
