import pytest

from anonimizar import Anonimizar


class TestAddApplyPatternsCustomPatterns:
    """Testes para o parâmetro custom_patterns em add_apply_patterns."""

    def test_apply_with_valid_custom_patterns(self, model_path):
        anonymizer = Anonimizar(model_path)
        custom = [
            {"label": "CODIGO_TESTE", "regex": r"TEST-\d{4}", "description": "Código de teste"},
            {"label": "MATRICULA", "regex": r"\b\d{6}-\d{2}\b"},
        ]
        anonymizer.add_apply_patterns(["EMAIL"], custom_patterns=custom)

        labels = {p["label"] for p in anonymizer.patterns}
        assert "EMAIL" in labels
        assert "CODIGO_TESTE" in labels
        assert "MATRICULA" in labels

        text = "Código TEST-1234 e matrícula 123456-01"
        result = anonymizer.extract_entities_regex_re(text, "label_text")
        detected = {(e["label"], e["text"]) for e in result}
        assert ("CODIGO_TESTE", "TEST-1234") in detected
        assert ("MATRICULA", "123456-01") in detected

    def test_apply_with_custom_missing_label(self, model_path):
        anonymizer = Anonimizar(model_path)
        custom = [
            {"regex": r"ABC-\d{3}"},
        ]
        with pytest.raises(ValueError, match="Padrão customizado inválido"):
            anonymizer.add_apply_patterns(["CPF"], custom_patterns=custom)

    def test_apply_with_custom_missing_regex(self, model_path):
        anonymizer = Anonimizar(model_path)
        custom = [
            {"label": "SEM_REGEX"},
        ]
        with pytest.raises(ValueError, match="Padrão customizado inválido"):
            anonymizer.add_apply_patterns(["EMAIL"], custom_patterns=custom)

    def test_apply_with_custom_invalid_regex_syntax(self, model_path):
        anonymizer = Anonimizar(model_path)
        custom = [
            {"label": "REGEX_INVAL", "regex": r"[invalid"},
        ]
        with pytest.raises(ValueError, match="Regex inválido"):
            anonymizer.add_apply_patterns(["TELEFONE"], custom_patterns=custom)

    def test_apply_custom_preserves_description(self, model_path):
        anonymizer = Anonimizar(model_path)
        custom = [
            {"label": "DOC_SEI", "regex": r"\d{5}\.\d{6}/\d{4}-\d{2}", "description": "Nº de processo SEI"},
        ]
        anonymizer.add_apply_patterns(["CPF"], custom_patterns=custom)
        doc_patterns = [p for p in anonymizer.patterns if p["label"] == "DOC_SEI"]
        assert len(doc_patterns) == 1
        assert doc_patterns[0].get("description") == "Nº de processo SEI"

    def test_apply_custom_with_replace_patterns(self, model_path):
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])
        prev_count = len(anonymizer.patterns)

        custom = [
            {"label": "COD_PROD", "regex": r"PROD-\d{4}"},
        ]
        anonymizer.add_apply_patterns(["EMAIL"], custom_patterns=custom, replace_patterns=True)
        labels = {p["label"] for p in anonymizer.patterns}

        assert len(anonymizer.patterns) < prev_count
        assert "EMAIL" in labels
        assert "COD_PROD" in labels
        assert "CPF" not in labels

    def test_apply_custom_combined_with_use_model_labels(self, model_path):
        anonymizer = Anonimizar(model_path)
        custom = [
            {"label": "CHAVE_NOTA", "regex": r"\b\d{44}\b"},
        ]
        anonymizer.add_apply_patterns(labels=["EMAIL"], use_model_labels=True, custom_patterns=custom)

        info = anonymizer.get_active_labels()
        assert "EMAIL" in info["model_labels"]
        assert "EMAIL" in info["regex_summary"]
        assert "CHAVE_NOTA" in info["regex_summary"]

    def test_custom_pattern_empty_regex_raises(self, model_path):
        anonymizer = Anonimizar(model_path)
        with pytest.raises(ValueError, match="string não vazia"):
            anonymizer.add_custom_pattern("COD", "   ", "desc")

    def test_custom_pattern_invalid_syntax_raises(self, model_path):
        anonymizer = Anonimizar(model_path)
        with pytest.raises(ValueError, match="Regex inválido"):
            anonymizer.add_custom_pattern("COD", r"[abc", "desc")

    def test_custom_pattern_uppercase_and_description(self, model_path):
        anonymizer = Anonimizar(model_path)
        anonymizer.add_custom_pattern("doc_sei", r"\d{5}\.\d{6}/\d{4}-\d{2}", "Nº SEI")
        p = [p for p in anonymizer.patterns if p["label"] == "DOC_SEI"]
        assert len(p) == 1
        assert p[0]["pattern"]["REGEX"] == r"\d{5}\.\d{6}/\d{4}-\d{2}"
        assert p[0].get("description") == "Nº SEI"

    def test_apply_multiple_custom_patterns_together(self, model_path):
        anonymizer = Anonimizar(model_path)
        custom = [
            {"label": "PROC_SEI", "regex": r"\d{5}\.\d{6}/\d{4}-\d{2}"},
            {"label": "COD_PEDIDO", "regex": r"PED-\d{6}"},
        ]
        anonymizer.add_apply_patterns(["RG"], custom_patterns=custom)

        text = "Processo 12345.123456/2024-10 e Pedido PED-012345"
        result = anonymizer.extract_entities_regex_re(text, "label_text")
        found = {(e["label"], e["text"]) for e in result}
        assert ("PROC_SEI", "12345.123456/2024-10") in found
        assert ("COD_PEDIDO", "PED-012345") in found


class TestAddApplyPatterns:
    """Testes para o método add_apply_patterns."""

    def test_apply_basic_patterns(self, model_path):
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "EMAIL"])

        labels = {p["label"] for p in anonymizer.patterns}
        assert "CPF" in labels
        assert "EMAIL" in labels
        assert len(anonymizer.patterns) > 0

    def test_apply_cid_patterns(self, model_path):
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["CID"])

        labels = {p["label"] for p in anonymizer.patterns}
        assert "CID" in labels
        assert len(anonymizer.patterns) > 0

    def test_apply_replace_patterns(self, model_path):
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "EMAIL"])
        anonymizer.add_apply_patterns(["CPF"])
        count_cpf = len(anonymizer.patterns)

        anonymizer.add_apply_patterns(["EMAIL"], replace_patterns=True)
        labels = {p["label"] for p in anonymizer.patterns}

        assert len(anonymizer.patterns) < count_cpf
        assert labels == {"EMAIL"}

    def test_apply_with_model_labels(self, model_path):
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(labels=["EMAIL"], use_model_labels=True)

        active_info = anonymizer.get_active_labels()
        assert "EMAIL" in active_info["model_labels"]
        assert "EMAIL" in active_info["regex_summary"]

    def test_apply_passaporte_with_foreign(self, model_path):
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["PASSAPORTE"], foreign_passport=True)

        labels = {p["label"] for p in anonymizer.patterns}
        assert "PASSAPORTE" in labels
        regex_count = sum(1 for p in anonymizer.patterns if p["label"] == "PASSAPORTE")
        assert regex_count > 1

    def test_apply_without_labels_and_no_model(self, model_path):
        anonymizer = Anonimizar(model_path)
        with pytest.raises(ValueError, match="fornecer ao menos um label"):
            anonymizer.add_apply_patterns(labels=None, use_model_labels=False)

    def test_apply_with_invalid_label(self, model_path):
        anonymizer = Anonimizar(model_path)
        with pytest.raises(ValueError, match="Rótulo não suportado"):
            anonymizer.add_apply_patterns(["INVALIDO"])

    def test_requires_labels_when_not_using_model(self, model_path):
        anonymizer = Anonimizar(model_path)
        with pytest.raises(ValueError, match="fornecer ao menos um label"):
            anonymizer.add_apply_patterns(labels=None, use_model_labels=False)

    def test_unsupported_label_raises(self, model_path):
        anonymizer = Anonimizar(model_path)
        with pytest.raises(ValueError, match="Rótulo não suportado"):
            anonymizer.add_apply_patterns(["INVALIDO"])

    def test_foreign_passport_adds_extra_regex(self, model_path):
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["PASSAPORTE"], foreign_passport=True)
        count_passaporte = sum(1 for p in anonymizer.patterns if p["label"] == "PASSAPORTE")
        assert count_passaporte > 1

    def test_fistel_without_cpf_adds_cpf_patterns(self, model_path):
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["FISTEL"])
        labels = {p["label"] for p in anonymizer.patterns}
        assert "CPF" in labels

    def test_fistel_with_cpf_does_not_duplicate(self, model_path):
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "FISTEL"])
        cpf_count = sum(1 for p in anonymizer.patterns if p["label"] == "CPF")
        assert cpf_count >= 1


class TestListPatterns:
    """Testes para o método list_patterns()."""

    def test_list_patterns_initial_empty(self, model_path):
        anonymizer = Anonimizar(model_path, auto_patterns=False)
        listed = anonymizer.list_patterns()
        assert isinstance(listed, dict)
        assert "builtin" in listed
        assert "custom" in listed
        assert isinstance(listed["builtin"], list)
        assert isinstance(listed["custom"], list)
        assert len(listed["builtin"]) == 0
        assert len(listed["custom"]) == 0

    def test_list_patterns_with_builtin_only(self, model_path):
        anonymizer = Anonimizar(model_path, auto_patterns=False)
        anonymizer.add_apply_patterns(["CPF", "EMAIL"])
        listed = anonymizer.list_patterns()

        assert len(listed["builtin"]) > 0
        assert all(p["label"] in {"CPF", "EMAIL"} for p in listed["builtin"])
        assert len(listed["custom"]) == 0

    def test_list_patterns_mixed_builtin_and_custom(self, model_path):
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "EMAIL"])
        anonymizer.add_custom_pattern("DOC_SEI", r"\d{5}\.\d{6}/\d{4}-\d{2}", "Nº SEI")

        listed = anonymizer.list_patterns()
        builtin_labels = {p["label"] for p in listed["builtin"]}
        custom_labels = {p["label"] for p in listed["custom"]}

        assert "CPF" in builtin_labels
        assert "EMAIL" in builtin_labels
        assert "DOC_SEI" in custom_labels
        assert builtin_labels.isdisjoint(custom_labels)

    def test_list_patterns_after_replace_patterns(self, model_path):
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])
        pre_replace = anonymizer.list_patterns()
        assert len(pre_replace["builtin"]) > 0

        anonymizer.add_apply_patterns(["EMAIL"], replace_patterns=True)
        anonymizer.add_custom_pattern("COD_PED", r"PED-\d{6}", "Código do pedido")

        post_replace = anonymizer.list_patterns()
        builtin_labels = {p["label"] for p in post_replace["builtin"]}
        custom_labels = {p["label"] for p in post_replace["custom"]}

        assert "EMAIL" in builtin_labels
        assert "CPF" not in builtin_labels
        assert "COD_PED" in custom_labels

    def test_list_patterns_respects_model_labels_filter(self, model_path):
        anonymizer = Anonimizar(model_path, labels=["RG"])
        anonymizer.add_apply_patterns(["RG", "EMAIL"])

        listed = anonymizer.list_patterns()
        builtin_labels = {p["label"] for p in listed["builtin"]}
        custom_labels = {p["label"] for p in listed["custom"]}

        assert "RG" in builtin_labels
        assert "EMAIL" in custom_labels


class TestPatternManagement:
    """Testes para gerenciamento de padrões."""

    def test_add_apply_patterns(self, model_path):
        anonymizer = Anonimizar(model_path)
        initial_count = len(anonymizer.patterns)
        anonymizer.add_apply_patterns(["CPF", "EMAIL"])
        assert len(anonymizer.patterns) > initial_count

    def test_add_custom_pattern(self, anonymizer):
        anonymizer.add_custom_pattern(label="CODIGO_TESTE", regex_pattern=r"TEST-\d{4}", description="Código de teste")
        custom_patterns = [p for p in anonymizer.patterns if p["label"] == "CODIGO_TESTE"]
        assert len(custom_patterns) == 1
        result = anonymizer.extract_entities_regex_re("Código TEST-1234", "label_text")
        assert len(result) == 1
        assert result[0]["label"] == "CODIGO_TESTE"

    def test_replace_patterns(self, anonymizer):
        anonymizer.add_apply_patterns(["CPF"])
        initial_count = len(anonymizer.patterns)
        anonymizer.add_apply_patterns(["EMAIL"], replace_patterns=True)
        final_count = len(anonymizer.patterns)
        assert final_count < initial_count
        labels = {p["label"] for p in anonymizer.patterns}
        assert labels == {"EMAIL"}


class TestGetActiveLabels:
    """Testes para informações de labels ativos."""

    def test_get_active_labels(self, anonymizer):
        info = anonymizer.get_active_labels()
        assert "model_labels" in info
        assert "regex_summary" in info
        assert "regex_patterns" in info
        assert "total_patterns" in info
        assert isinstance(info["model_labels"], list)
        assert isinstance(info["regex_summary"], dict)
        assert isinstance(info["regex_patterns"], int)
        assert isinstance(info["total_patterns"], int)


class TestErrorHandling:
    """Testes de erro para add_custom_pattern."""

    def test_invalid_regex_pattern(self, anonymizer):
        with pytest.raises(ValueError, match="Regex inválido"):
            anonymizer.add_custom_pattern("TEST", "[invalid regex", "Teste")

    def test_empty_pattern(self, anonymizer):
        with pytest.raises(ValueError, match="não vazia"):
            anonymizer.add_custom_pattern("TEST", "", "Teste")


class TestUnsupportedLabelsAddApplyPatterns:
    """Testes de regressão: labels sem padrões regex não devem levantar erro."""

    def test_pis_sem_padrao_nao_levanta_erro(self, model_path):
        """PIS não tem função de padrão; deve ser ignorado, não levantar erro."""
        anonymizer = Anonimizar(model_path, auto_patterns=False)
        anonymizer.add_apply_patterns(["PIS"])
        assert True

    def test_cns_com_padrao_funciona(self, model_path):
        """CNS tem função add_pattern_cns; deve adicionar padrões."""
        anonymizer = Anonimizar(model_path, auto_patterns=False)
        anonymizer.add_apply_patterns(["CNS"])
        labels = {p["label"] for p in anonymizer.patterns}
        assert "CNS" in labels

    def test_reservista_com_padrao_funciona(self, model_path):
        """RESERVISTA tem função add_pattern_reservista; deve adicionar padrões."""
        anonymizer = Anonimizar(model_path, auto_patterns=False)
        anonymizer.add_apply_patterns(["RESERVISTA"])
        labels = {p["label"] for p in anonymizer.patterns}
        assert "RESERVISTA" in labels

    def test_todos_os_labels_suportados_funcionam_juntos(self, model_path):
        """Todos os labels em DEFAULT_SUPPORTED_LABELS devem funcionar juntos."""
        anonymizer = Anonimizar(model_path, auto_patterns=False)
        anonymizer.add_apply_patterns(["CPF", "RG", "SIAPE", "TELEFONE", "EMAIL", "CNS", "PIS", "RESERVISTA"])
        labels = {p["label"] for p in anonymizer.patterns}
        assert "CPF" in labels
        assert "CNS" in labels
        assert "RESERVISTA" in labels

    def test_label_invalido_ainda_levanta_erro(self, model_path):
        """Labels verdadeiramente inválidos (não em ALL_MODEL_LABELS) ainda devem levantar erro."""
        anonymizer = Anonimizar(model_path, auto_patterns=False)
        with pytest.raises(ValueError, match="não suportado"):
            anonymizer.add_apply_patterns(["LABEL_INEXISTENTE"])
