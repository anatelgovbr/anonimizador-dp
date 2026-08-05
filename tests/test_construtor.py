import pytest

from anonimizar import Anonimizar


class TestAnonimizarInit:
    """Testes para inicialização da classe."""

    def test_init_with_valid_model(self, model_path):
        anonymizer = Anonimizar(model_path)
        assert anonymizer.model_path == model_path
        assert anonymizer.nlp_trained is not None
        assert anonymizer.use_cpf_validator is True

    def test_init_with_empty_model_path(self):
        with pytest.raises(ValueError, match="É necessario ter o model_path preenchido"):
            Anonimizar("")

    def test_init_with_custom_labels(self, model_path):
        labels = ["CPF", "EMAIL"]
        anonymizer = Anonimizar(model_path, labels=labels)
        assert anonymizer.labels == set(labels)

    def test_init_with_custom_label_fistel(self, model_path):
        labels = ["FISTEL", "EMAIL"]
        anonymizer = Anonimizar(model_path, labels=["EMAIL", "FISTEL"])
        assert anonymizer.labels == set(labels)


class TestAutoPatterns:
    """Testes para o parâmetro auto_patterns do construtor."""

    def test_auto_patterns_true_aplica_patterns(self, model_path):
        anonymizer = Anonimizar(model_path=model_path)
        assert len(anonymizer.patterns) > 0

    def test_auto_patterns_false_nao_aplica(self, model_path):
        anonymizer = Anonimizar(model_path=model_path, auto_patterns=False)
        assert len(anonymizer.patterns) == 0

    def test_auto_patterns_permite_sobrescrever(self, model_path):
        anonymizer = Anonimizar(model_path=model_path)
        qtd_original = len(anonymizer.patterns)
        anonymizer.add_apply_patterns(["CPF"], replace_patterns=True)
        assert len(anonymizer.patterns) < qtd_original

    def test_auto_patterns_default_e_true(self, model_path):
        anonymizer = Anonimizar(model_path=model_path)
        labels_nos_patterns = {p["label"] for p in anonymizer.patterns}
        assert "CPF" in labels_nos_patterns
        assert "EMAIL" in labels_nos_patterns

    def test_auto_patterns_false_funciona_como_antes(self, model_path):
        anonymizer = Anonimizar(model_path=model_path, auto_patterns=False)
        assert len(anonymizer.patterns) == 0
        anonymizer.add_apply_patterns(["CPF", "EMAIL"])
        assert len(anonymizer.patterns) > 0
        labels_nos_patterns = {p["label"] for p in anonymizer.patterns}
        assert "CPF" in labels_nos_patterns
        assert "EMAIL" in labels_nos_patterns

    def test_auto_patterns_true_contem_cpf_e_email(self, model_path):
        anonymizer = Anonimizar(model_path=model_path)
        labels_com_pattern = {p["label"] for p in anonymizer.patterns}
        assert "CPF" in labels_com_pattern
        assert "EMAIL" in labels_com_pattern

    def test_auto_patterns_false_permite_add_manual(self, model_path):
        anonymizer = Anonimizar(model_path=model_path, auto_patterns=False)
        assert len(anonymizer.patterns) == 0
        anonymizer.add_apply_patterns(["CPF"])
        assert len(anonymizer.patterns) > 0
        assert any(p["label"] == "CPF" for p in anonymizer.patterns)
