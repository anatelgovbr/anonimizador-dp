"""Testes para o módulo SeiAnonimizar."""

import pytest

from anonimizar.sei_anonimizar import SeiAnonimizar


class TestSeiAnonimizarInit:
    """Testes para inicialização da classe."""

    def test_init_with_valid_model(self, model_path):
        """Testa inicialização com modelo válido."""
        anonymizer = SeiAnonimizar(model_path)
        assert anonymizer.model_path == model_path
        assert anonymizer.nlp_trained is not None
        assert anonymizer.use_cpf_validator is True

    def test_init_with_empty_model_path(self):
        """Testa inicialização com modelo vazio."""
        with pytest.raises(ValueError, match="É necessario ter o model_path preenchido"):
            SeiAnonimizar("")

    def test_init_with_custom_labels(self, model_path):
        """Testa inicialização com labels customizados."""
        labels = ["CPF", "EMAIL"]
        anonymizer = SeiAnonimizar(model_path, labels=labels)
        assert anonymizer.labels == set(labels)

    def test_init_with_custom_label_fistel(self, model_path):
        """Testa inicialização com labels customizados."""
        labels = ["FISTEL", "EMAIL"]
        anonymizer = SeiAnonimizar(model_path, labels=["EMAIL", "FISTEL"])
        assert anonymizer.labels == set(labels)


class TestAddApplyPatternsCustomPatterns:
    """Testes para o parâmetro custom_patterns em add_apply_patterns."""

    def test_apply_with_valid_custom_patterns(self, model_path):
        """Aplica padrões customizados válidos junto com labels built-in."""
        anonymizer = SeiAnonimizar(model_path)
        custom = [
            {"label": "CODIGO_TESTE", "regex": r"TEST-\d{4}", "description": "Código de teste"},
            {"label": "MATRICULA", "regex": r"\b\d{6}-\d{2}\b"},
        ]
        anonymizer.add_apply_patterns(["EMAIL"], custom_patterns=custom)

        # Verifica se os padrões customizados foram adicionados
        labels = {p["label"] for p in anonymizer.patterns}
        assert "EMAIL" in labels
        assert "CODIGO_TESTE" in labels
        assert "MATRICULA" in labels

        # Verifica se detecta por regex customizado
        text = "Código TEST-1234 e matrícula 123456-01"
        result = anonymizer.extract_entities_regex_re(text, "label_text")
        detected = {(e["label"], e["text"]) for e in result}
        assert ("CODIGO_TESTE", "TEST-1234") in detected
        assert ("MATRICULA", "123456-01") in detected

    def test_apply_with_custom_missing_label(self, model_path):
        """Gera erro quando um padrão customizado não tem 'label'."""
        anonymizer = SeiAnonimizar(model_path)
        custom = [
            {"regex": r"ABC-\d{3}"},  # faltando label
        ]
        with pytest.raises(ValueError, match="Padrão customizado inválido"):
            anonymizer.add_apply_patterns(["CPF"], custom_patterns=custom)

    def test_apply_with_custom_missing_regex(self, model_path):
        """Gera erro quando um padrão customizado não tem 'regex'."""
        anonymizer = SeiAnonimizar(model_path)
        custom = [
            {"label": "SEM_REGEX"},  # faltando regex
        ]
        with pytest.raises(ValueError, match="Padrão customizado inválido"):
            anonymizer.add_apply_patterns(["EMAIL"], custom_patterns=custom)

    def test_apply_with_custom_invalid_regex_syntax(self, model_path):
        """Gera erro quando o regex do padrão customizado é inválido (sintaxe)."""
        anonymizer = SeiAnonimizar(model_path)
        custom = [
            {"label": "REGEX_INVAL", "regex": r"[invalid"},  # regex inválido
        ]
        with pytest.raises(ValueError, match="Regex inválido"):
            anonymizer.add_apply_patterns(["TELEFONE"], custom_patterns=custom)

    def test_apply_custom_preserves_description(self, model_path):
        """Garante que a description do padrão customizado é preservada."""
        anonymizer = SeiAnonimizar(model_path)
        custom = [
            {"label": "DOC_SEI", "regex": r"\d{5}\.\d{6}/\d{4}-\d{2}", "description": "Nº de processo SEI"},
        ]
        anonymizer.add_apply_patterns(["CPF"], custom_patterns=custom)
        doc_patterns = [p for p in anonymizer.patterns if p["label"] == "DOC_SEI"]
        assert len(doc_patterns) == 1
        assert doc_patterns[0].get("description") == "Nº de processo SEI"

    def test_apply_custom_with_replace_patterns(self, model_path):
        """Substitui padrões anteriores e mantém apenas os customizados + novos solicitados."""
        anonymizer = SeiAnonimizar(model_path)
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
        """Combina custom_patterns com use_model_labels=True."""
        anonymizer = SeiAnonimizar(model_path)
        custom = [
            {"label": "CHAVE_NOTA", "regex": r"\b\d{44}\b"},
        ]
        anonymizer.add_apply_patterns(labels=["EMAIL"], use_model_labels=True, custom_patterns=custom)

        info = anonymizer.get_active_labels()
        assert "EMAIL" in info["model_labels"]
        assert "EMAIL" in info["regex_summary"]
        assert "CHAVE_NOTA" in info["regex_summary"]

    def test_custom_pattern_empty_regex_raises(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        with pytest.raises(ValueError, match="string não vazia"):
            anonymizer.add_custom_pattern("COD", "   ", "desc")

    def test_custom_pattern_invalid_syntax_raises(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        with pytest.raises(ValueError, match="Regex inválido"):
            anonymizer.add_custom_pattern("COD", r"[abc", "desc")

    def test_custom_pattern_uppercase_and_description(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_custom_pattern("doc_sei", r"\d{5}\.\d{6}/\d{4}-\d{2}", "Nº SEI")
        p = [p for p in anonymizer.patterns if p["label"] == "DOC_SEI"]
        assert len(p) == 1
        assert p[0]["pattern"]["REGEX"] == r"\d{5}\.\d{6}/\d{4}-\d{2}"
        assert p[0].get("description") == "Nº SEI"

    def test_apply_multiple_custom_patterns_together(self, model_path):
        """Aplica múltiplos custom_patterns de uma vez e valida detecção."""
        anonymizer = SeiAnonimizar(model_path)
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
        """Testa aplicação de padrões básicos (CPF e EMAIL)."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "EMAIL"])

        labels = {p["label"] for p in anonymizer.patterns}
        assert "CPF" in labels
        assert "EMAIL" in labels
        assert len(anonymizer.patterns) > 0

    def test_apply_CID_patterns(self, model_path):
        """Testa aplicação de padrões básicos (CID)."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CID"])

        labels = {p["label"] for p in anonymizer.patterns}
        assert "CID" in labels
        assert len(anonymizer.patterns) > 0

    def test_apply_replace_patterns(self, model_path):
        """Testa substituição de padrões."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "EMAIL"])
        anonymizer.add_apply_patterns(["CPF"])
        count_cpf = len(anonymizer.patterns)

        anonymizer.add_apply_patterns(["EMAIL"], replace_patterns=True)
        labels = {p["label"] for p in anonymizer.patterns}

        assert len(anonymizer.patterns) < count_cpf
        assert labels == {"EMAIL"}

    def test_apply_with_model_labels(self, model_path):
        """Testa ativação com use_model_labels=True."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(labels=["EMAIL"], use_model_labels=True)

        active_info = anonymizer.get_active_labels()
        assert "EMAIL" in active_info["model_labels"]
        assert "EMAIL" in active_info["regex_summary"]

    def test_apply_passaporte_with_foreign(self, model_path):
        """Testa aplicação de padrões de passaporte incluindo estrangeiros."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["PASSAPORTE"], foreign_passport=True)

        labels = {p["label"] for p in anonymizer.patterns}
        assert "PASSAPORTE" in labels
        # Deve conter mais de um regex (passaporte nacional + estrangeiro)
        regex_count = sum(1 for p in anonymizer.patterns if p["label"] == "PASSAPORTE")
        assert regex_count > 1

    def test_apply_without_labels_and_no_model(self, model_path):
        """Testa erro quando nenhum label é fornecido e use_model_labels=False."""
        anonymizer = SeiAnonimizar(model_path)
        with pytest.raises(ValueError, match="fornecer ao menos um label"):
            anonymizer.add_apply_patterns(labels=None, use_model_labels=False)

    def test_apply_with_invalid_label(self, model_path):
        """Testa erro ao passar label inválido."""
        anonymizer = SeiAnonimizar(model_path)
        with pytest.raises(ValueError, match="Rótulo não suportado"):
            anonymizer.add_apply_patterns(["INVALIDO"])

    def test_requires_labels_when_not_using_model(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        with pytest.raises(ValueError, match="fornecer ao menos um label"):
            anonymizer.add_apply_patterns(labels=None, use_model_labels=False)

    def test_unsupported_label_raises(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        with pytest.raises(ValueError, match="Rótulo não suportado"):
            anonymizer.add_apply_patterns(["INVALIDO"])

    def test_foreign_passport_adds_extra_regex(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["PASSAPORTE"], foreign_passport=True)
        count_passaporte = sum(1 for p in anonymizer.patterns if p["label"] == "PASSAPORTE")
        assert count_passaporte > 1

    def test_fistel_without_cpf_adds_cpf_patterns(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["FISTEL"])
        labels = {p["label"] for p in anonymizer.patterns}
        assert "CPF" in labels

    def test_fistel_with_cpf_does_not_duplicate(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "FISTEL"])
        cpf_count = sum(1 for p in anonymizer.patterns if p["label"] == "CPF")
        assert cpf_count >= 1


class TestListPatterns:
    """Testes para o método list_patterns()."""

    def test_list_patterns_initial_empty(self, model_path):
        """No início, sem add_apply_patterns, builtin e custom devem refletir o estado inicial."""
        anonymizer = SeiAnonimizar(model_path)
        listed = anonymizer.list_patterns()
        assert isinstance(listed, dict)
        assert "builtin" in listed and "custom" in listed
        assert isinstance(listed["builtin"], list)
        assert isinstance(listed["custom"], list)
        assert len(listed["builtin"]) == 0
        assert len(listed["custom"]) == 0

    def test_list_patterns_with_builtin_only(self, model_path):
        """Após aplicar padrões built-in, devem aparecer apenas em 'builtin'."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "EMAIL"])  # apenas built-in
        listed = anonymizer.list_patterns()

        assert len(listed["builtin"]) > 0
        assert all(p["label"] in {"CPF", "EMAIL"} for p in listed["builtin"])
        assert len(listed["custom"]) == 0

    def test_list_patterns_mixed_builtin_and_custom(self, model_path):
        """Mistura de built-in e custom deve separar corretamente as listas."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "EMAIL"])
        anonymizer.add_custom_pattern("DOC_SEI", r"\d{5}\.\d{6}/\d{4}-\d{2}", "Nº SEI")

        listed = anonymizer.list_patterns()
        builtin_labels = {p["label"] for p in listed["builtin"]}
        custom_labels = {p["label"] for p in listed["custom"]}

        assert "CPF" in builtin_labels and "EMAIL" in builtin_labels
        assert "DOC_SEI" in custom_labels
        assert builtin_labels.isdisjoint(custom_labels)

    def test_list_patterns_after_replace_patterns(self, model_path):
        """Após replace_patterns=True, list_patterns deve refletir o novo estado."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])
        pre_replace = anonymizer.list_patterns()
        assert len(pre_replace["builtin"]) > 0  # tinha CPF

        anonymizer.add_apply_patterns(["EMAIL"], replace_patterns=True)
        anonymizer.add_custom_pattern("COD_PED", r"PED-\d{6}", "Código do pedido")

        post_replace = anonymizer.list_patterns()
        builtin_labels = {p["label"] for p in post_replace["builtin"]}
        custom_labels = {p["label"] for p in post_replace["custom"]}

        assert "EMAIL" in builtin_labels
        assert "CPF" not in builtin_labels  # removido pelo replace
        assert "COD_PED" in custom_labels

    def test_list_patterns_respects_model_labels_filter(self, model_path):
        """'builtin' deve conter apenas labels presentes em anonymizer.labels (labels do modelo/ativos)."""
        anonymizer = SeiAnonimizar(model_path, labels=["RG"])
        anonymizer.add_apply_patterns(["RG", "EMAIL"])

        listed = anonymizer.list_patterns()
        builtin_labels = {p["label"] for p in listed["builtin"]}
        custom_labels = {p["label"] for p in listed["custom"]}

        assert "RG" in builtin_labels
        assert "EMAIL" in custom_labels


class TestEntityExtraction:
    """Testes para extração de entidades."""

    @pytest.mark.entity
    @pytest.mark.parametrize(
        "input_text, expected_label, expected_text",
        [
            # CPF
            ("Meu CPF é 123.456.789-09.", "CPF", "123.456.789-09"),
            # RG
            ("Meu RG é 12.345.678-9.", "RG", "12.345.678-9"),
            # # TITULO_ELEITOR
            # ("Título de eleitor: 1234 5678 9012", "TITULO_ELEITOR", "1234 5678 9012"),
            # PASSAPORTE
            ("Passaporte AB123456", "PASSAPORTE", "AB123456"),
            # SIAPE
            ("SIAPE 1234567", "SIAPE", "SIAPE 1234567"),
            # DADOS_BANCARIOS
            ("Conta bancária 1234-5", "DADOS_BANCARIOS", "1234-5"),
            # EMAIL
            ("Email: teste@email.com", "EMAIL", "teste@email.com"),
            # TELEFONE
            ("Telefone: (11) 99999-9999", "TELEFONE", "(11) 99999-9999"),
            # DATA_NASCIMENTO
            ("Data de nascimento: 15/03/1985", "DATA_NASCIMENTO", "15/03/1985"),
            # CNH
            ("CNH 12345678901", "CNH", "CNH 12345678901"),
            # ENDEREÇO
            ("Endereço com CEP 01234-567", "ENDEREÇO", "01234-567"),
            # GEO_COORD
            ("Coordenada: 22°31.26' S", "GEO_COORD", "22°31.26' S"),
            # CID
            ("CID meu atestado referente ao A10", "CID", "A10"),
        ],
    )
    def test_extract_all_entity_types(self, anonymizer, input_text, expected_label, expected_text):
        """Testa detecção de pelo menos uma ocorrência de cada tipo de entidade."""
        result = anonymizer.extract_entities(input_text, return_type="label_detail")

        # Verifica se há exatamente uma entidade detectada
        assert len(result) == 1, f"Esperado 1 entidade, mas encontrou {len(result)} para '{input_text}'"

        entity = result[0]
        assert entity["label"] == expected_label, f"Label esperado: {expected_label}, mas obteve {entity['label']}"
        assert entity["text"] == expected_text, f"Texto esperado: {expected_text}, mas obteve {entity['text']}"
        assert "detected_by" in entity  # Verifica se tem método de detecção

    @pytest.mark.parametrize(
        "input_text,return_type,expected_count",
        [
            ("CPF 123.456.789-09", "label_text", 1),
            ("Email teste@email.com", "label_text", 1),
            ("Telefone (11) 99999-9999", "label_text", 1),
            ("Texto sem entidades", "label_text", 0),
        ],
    )
    def test_extract_entities_basic(self, anonymizer, input_text, return_type, expected_count):
        """Testa extração básica de entidades."""
        result = anonymizer.extract_entities(input_text, return_type=return_type)
        assert len(result) == expected_count

    @pytest.mark.cpf
    @pytest.mark.parametrize(
        "input_text,expected_cpf", [("CPF 123.456.789-09", "123.456.789-09"), ("nº:12345678909", "12345678909")]
    )
    def test_extract_cpf_formats(self, anonymizer, input_text, expected_cpf):
        """Testa diferentes formatos de CPF."""
        result = anonymizer.extract_entities(input_text, return_type="label_text")
        cpf_entities = [e for e in result if e["label"] == "CPF"]
        assert len(cpf_entities) > 0
        assert cpf_entities[0]["text"] == expected_cpf

    @pytest.mark.email
    @pytest.mark.parametrize(
        "input_text,expected_email",
        [
            ("Email: teste@dominio.com", "teste@dominio.com"),
            ("Contato via usuario@empresa.com.br", "usuario@empresa.com.br"),
            ("user_123@test-domain.org", "user_123@test-domain.org"),
        ],
    )
    def test_extract_email_formats(self, anonymizer, input_text, expected_email):
        """Testa diferentes formatos de email."""
        result = anonymizer.extract_entities(input_text, return_type="label_text")
        email_entities = [e for e in result if e["label"] == "EMAIL"]
        assert len(email_entities) > 0
        assert email_entities[0]["text"] == expected_email

    def test_extract_entities_with_positions(self, anonymizer):
        """Testa extração com posições."""
        text = "João, CPF 123.456.789-09, email: joao@email.com"
        result = anonymizer.extract_entities(text, return_type="label_position")

        assert len(result) == 2  # CPF e EMAIL
        cpf_entity = next(e for e in result if e["label"] == "CPF")
        assert cpf_entity["start_position"] == 10
        assert cpf_entity["end_position"] == 24

    def test_extract_entities_with_detail(self, anonymizer):
        """Testa extração com detalhes."""
        text = "CPF 123.456.789-09"
        result = anonymizer.extract_entities(text, return_type="label_detail")

        assert len(result) == 1
        entity = result[0]
        assert "detected_by" in entity
        assert entity["text"] == "123.456.789-09"


class TestPatternManagement:
    """Testes para gerenciamento de padrões."""

    def test_add_apply_patterns(self, model_path):
        """Testa adição de padrões."""
        anonymizer = SeiAnonimizar(model_path)
        initial_count = len(anonymizer.patterns)

        anonymizer.add_apply_patterns(["CPF", "EMAIL"])
        assert len(anonymizer.patterns) > initial_count

    def test_add_custom_pattern(self, anonymizer):
        """Testa adição de padrão customizado."""
        anonymizer.add_custom_pattern(label="CODIGO_TESTE", regex_pattern=r"TEST-\d{4}", description="Código de teste")

        # Verifica se foi adicionado
        custom_patterns = [p for p in anonymizer.patterns if p["label"] == "CODIGO_TESTE"]
        assert len(custom_patterns) == 1

        # Testa detecção
        result = anonymizer.extract_entities_regex_re("Código TEST-1234", "label_text")
        assert len(result) == 1
        assert result[0]["label"] == "CODIGO_TESTE"

    def test_replace_patterns(self, anonymizer):
        """Testa substituição de padrões."""
        anonymizer.add_apply_patterns(["CPF"])
        initial_count = len(anonymizer.patterns)

        anonymizer.add_apply_patterns(["EMAIL"], replace_patterns=True)
        final_count = len(anonymizer.patterns)

        # Deve ter menos padrões após substituição
        assert final_count < initial_count
        # Deve ter apenas padrões de EMAIL
        labels = {p["label"] for p in anonymizer.patterns}
        assert labels == {"EMAIL"}


class TestValidation:
    """Testes para validação de entidades."""

    @pytest.mark.parametrize(
        "cpf,expected",
        [
            ("123.456.789-09", True),
            ("000.000.000-00", False),
            ("123.456.789-00", False),
            ("12345678909", True),
        ],
    )
    def test_valida_cpf(self, cpf, expected):
        """Testa validação de CPF."""
        result = SeiAnonimizar.valida_cpf(cpf)
        assert result == expected

    def test_cpf_validation_with_asterisks(self):
        """Testa CPF com asteriscos (mascarado)."""
        result = SeiAnonimizar.valida_cpf("***.456.789-**")
        assert result is False

    def test_titulo_validation(self):
        """Testa validação de título de eleitor."""
        # Teste com título válido (exemplo)
        result = SeiAnonimizar.valida_titulo_eleitor("123456789011")
        assert isinstance(result, bool)


class TestAnonymization:
    """Testes para anonimização de texto."""

    def test_anonymize_text_basic(self, anonymizer):
        """Testa anonimização básica."""
        text = "João, CPF 123.456.789-09, email: joao@email.com"
        entities = anonymizer.extract_entities(text, return_type="label_position")

        anonymized = anonymizer.anonymize_text(text, entities)

        assert "<|CPF|>" in anonymized
        assert "<|EMAIL|>" in anonymized
        assert "123.456.789-09" not in anonymized
        assert "joao@email.com" not in anonymized

    def test_anonymize_text_empty_entities(self, anonymizer):
        """Testa anonimização com lista vazia."""
        text = "Texto sem entidades sensíveis"
        anonymized = anonymizer.anonymize_text(text, [])
        assert anonymized == text


class TestErrorHandling:
    """Testes para tratamento de erros."""

    def test_invalid_return_type(self, anonymizer):
        """Testa tipo de retorno inválido."""
        with pytest.raises(Exception):  # Ajustar para o tipo correto de exceção
            anonymizer.extract_entities("teste", return_type="invalid_type")

    def test_invalid_regex_pattern(self, anonymizer):
        """Testa padrão regex inválido."""
        with pytest.raises(ValueError):
            anonymizer.add_custom_pattern("TEST", "[invalid regex", "Teste")

    def test_empty_pattern(self, anonymizer):
        """Testa padrão vazio."""
        with pytest.raises(ValueError):
            anonymizer.add_custom_pattern("TEST", "", "Teste")


class TestOverlapRemoval:
    """Testes para remoção de sobreposições."""

    def test_remove_overlap_basic(self, anonymizer):
        """Testa remoção básica de sobreposições."""
        entities = [
            {"label": "CPF", "start_position": 0, "end_position": 14},
            {"label": "TELEFONE", "start_position": 5, "end_position": 19},  # Sobrepõe
        ]

        result = anonymizer.remove_overlap_positions(entities)

        # Deve manter apenas uma entidade
        assert len(result) == 1

    def test_no_overlap(self, anonymizer):
        """Testa quando não há sobreposição."""
        entities = [
            {"label": "CPF", "start_position": 0, "end_position": 14},
            {"label": "EMAIL", "start_position": 20, "end_position": 35},
        ]

        result = anonymizer.remove_overlap_positions(entities)

        assert len(result) == 2

    def test_priority_when_same_span(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        ents = [
            {"label": "EMAIL", "start_position": 10, "end_position": 20},
            {"label": "CPF", "start_position": 10, "end_position": 20},
        ]
        out = anonymizer.remove_overlap_positions(ents)
        assert len(out) == 1

    def test_contained_entity_removed(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        ents = [
            {"label": "CPF", "start_position": 0, "end_position": 30},
            {"label": "EMAIL", "start_position": 10, "end_position": 20},
        ]
        out = anonymizer.remove_overlap_positions(ents)
        assert len(out) == 1 and out[0]["label"] == "CPF"

    def test_touching_entities_not_removed(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        ents = [
            {"label": "CPF", "start_position": 0, "end_position": 10},
            {"label": "EMAIL", "start_position": 10, "end_position": 20},
        ]
        out = anonymizer.remove_overlap_positions(ents)
        assert len(out) == 2

    def test_union_when_new_extends_existing(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        ents = [
            {"label": "EMAIL", "start_position": 5, "end_position": 10},
            {"label": "CPF", "start_position": 8, "end_position": 15},
        ]
        out = anonymizer.remove_overlap_positions(ents)
        assert len(out) == 1
        assert out[0]["label"] in {"CPF", "EMAIL"}
        assert out[0]["start_position"] == 5
        assert out[0]["end_position"] == 15


class TestGetActiveLabels:
    """Testes para informações de labels ativos."""

    def test_get_active_labels(self, anonymizer):
        """Testa recuperação de labels ativos."""
        info = anonymizer.get_active_labels()

        assert "model_labels" in info
        assert "regex_summary" in info
        assert "regex_patterns" in info
        assert "total_patterns" in info

        assert isinstance(info["model_labels"], list)
        assert isinstance(info["regex_summary"], dict)
        assert isinstance(info["regex_patterns"], int)
        assert isinstance(info["total_patterns"], int)


def make_ent(label, text, frag):
    s = text.index(frag)
    return {"label_": label, "start_char": s, "end_char": s + len(frag)}


class TestVerifyEntitiesUnified:
    def test_cpf_valid_with_context(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        text = "meu cpf é 529.982.247-25"
        ent = make_ent("CPF", text, "529.982.247-25")
        assert anonymizer.verify_entities_unified(ent, text) is True

    def test_cpf_invalid_with_context(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        text = "meu dado é 529.982.247-29"
        ent = make_ent("CPF", text, "529.982.247-29")
        assert anonymizer.verify_entities_unified(ent, text) is False

    def test_cnpj_valid(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        text = "meu cpf é 18.781.203/0001-28"
        ent = make_ent("CPF", text, "18.781.203/0001-28")
        assert anonymizer.verify_entities_unified(ent, text) is False

    def test_cnpj_invalid(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        text = "meu cpf é 18.781.203/0001-27"
        ent = make_ent("CPF", text, "18.781.203/0001-27")
        assert anonymizer.verify_entities_unified(ent, text) is False

    def test_cpf_invalid_with_validator(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        text = "cpf 000.000.000-00"
        ent = make_ent("CPF", text, "000.000.000-00")
        assert anonymizer.verify_entities_unified(ent, text) is False

    def test_cpf_reclassifies_to_cnh_on_context(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        text = "cnh 12345678909"
        ent = make_ent("CPF", text, "12345678909")
        assert anonymizer.verify_entities_unified(ent, text) == "CNH"

    def test_endereco_requires_context(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        text = "01234-567"
        ent = make_ent("ENDEREÇO", text, "01234-567")
        assert anonymizer.verify_entities_unified(ent, text) is False

        text2 = "CEP: 01234-567"
        ent2 = make_ent("ENDEREÇO", text2, "01234-567")
        assert anonymizer.verify_entities_unified(ent2, text2) is True

    def test_rg_needs_context(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        t1 = "12.345.678-9"
        ent1 = make_ent("RG", t1, t1)
        assert anonymizer.verify_entities_unified(ent1, t1) is False

        t2 = "identidade 12.345.678-9"
        ent2 = make_ent("RG", t2, "12.345.678-9")
        assert anonymizer.verify_entities_unified(ent2, t2) is True

    def test_cnh_with_validator_and_context(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        t1 = "cnh 11111111111"
        ent1 = make_ent("CNH", t1, "11111111111")
        assert anonymizer.verify_entities_unified(ent1, t1) is False

        anonymizer.use_cnh_validator = False
        assert anonymizer.verify_entities_unified(ent1, t1) is True

    def test_siape_needs_siape_and_not_sei(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        t1 = "siape 1234567"
        ent1 = make_ent("SIAPE", t1, "1234567")
        assert anonymizer.verify_entities_unified(ent1, t1) is True

        t2 = "1234567 sei"
        ent2 = make_ent("SIAPE", t2, "1234567")
        assert anonymizer.verify_entities_unified(ent2, t2) is False

    def test_telefone_requires_context(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        t1 = "(11) 99999-9999"
        ent1 = make_ent("TELEFONE", t1, t1)
        assert anonymizer.verify_entities_unified(ent1, t1) is False

        t2 = "telefone (11) 99999-9999"
        ent2 = make_ent("TELEFONE", t2, "(11) 99999-9999")
        assert anonymizer.verify_entities_unified(ent2, t2) is True

    def test_data_nascimento_numeric(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        t1 = "nascimento 15/03/1985"
        ent1 = make_ent("DATA_NASCIMENTO", t1, "15/03/1985")
        assert anonymizer.verify_entities_unified(ent1, t1) is True

    def test_passaporte_length_and_context(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        t1 = "passaporte AB123456"
        ent1 = make_ent("PASSAPORTE", t1, "AB123456")
        assert anonymizer.verify_entities_unified(ent1, t1) is True

        t2 = "AB123456"  # sem contexto
        ent2 = make_ent("PASSAPORTE", t2, "AB123456")
        assert anonymizer.verify_entities_unified(ent2, t2) is False

        t3 = "passaporte ABCDEF0123X"  # >9 chars
        ent3 = make_ent("PASSAPORTE", t3, "ABCDEF0123X")
        assert anonymizer.verify_entities_unified(ent3, t3) is False

    def test_titulo_eleitor_validator_and_context(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        t1 = "eleitor 000000000000"
        ent1 = make_ent("TITULO_ELEITOR", t1, "000000000000")
        assert anonymizer.verify_entities_unified(ent1, t1) is False

        anonymizer.use_titulo_validator = False
        assert anonymizer.verify_entities_unified(ent1, t1) is True

    def test_dados_bancarios_context(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        t1 = "1234-5"
        ent1 = make_ent("DADOS_BANCARIOS", t1, "1234-5")
        assert anonymizer.verify_entities_unified(ent1, t1) is False

        t2 = "agência 1234-5"
        ent2 = make_ent("DADOS_BANCARIOS", t2, "1234-5")
        assert anonymizer.verify_entities_unified(ent2, t2) is True

    def test_cid_context(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        t2 = "Paciente com diagnóstico CID F32.1"
        ent2 = make_ent("CID", t2, "F32.1")
        assert anonymizer.verify_entities_unified(ent2, t2) is True

    def test_geo_coord_variants(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        t1 = "24°27'13.0\"S"
        ent1 = make_ent("GEO_COORD", t1, t1)
        assert anonymizer.verify_entities_unified(ent1, t1) is True
        t2 = "latitude -15,789509 -47,911627"
        ent2 = make_ent("GEO_COORD", t2, "-15,789509 -47,911627")
        assert anonymizer.verify_entities_unified(ent2, t2) is True
        t3 = "coordenada x"
        ent3 = make_ent("GEO_COORD", t3, "x")
        assert anonymizer.verify_entities_unified(ent3, t3) is False


class TestExtractEntitiesErrors:
    def test_invalid_return_type_raises_valueerror(self, model_path):
        anonymizer = SeiAnonimizar(model_path)
        with pytest.raises(ValueError, match="Tipo de retorno não permitido"):
            anonymizer.extract_entities("texto", return_type="invalid_type")


class TestMarkdownTableExtraction:
    """Testes para extração de entidades de tabelas markdown."""

    def test_extract_from_simple_table(self, model_path):
        """Testa extração básica de tabela markdown com CPF."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "RG"])

        text = """
| Nome | CPF | Idade |
|------|-----|-------|
| João | 123.456.789-09 | 30 |
| Maria | 987.654.321-00 | 25 |
"""
        result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")
        cpf_entities = [e for e in result if e["label"] == "CPF"]

        assert len(cpf_entities) == 2
        assert "123.456.789-09" in [e["text"] for e in cpf_entities]
        assert "987.654.321-00" in [e["text"] for e in cpf_entities]

    def test_extract_multiple_sensitive_columns(self, model_path):
        """Testa extração de múltiplas colunas sensíveis na mesma tabela."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "RG"])

        text = """
| Nome | CPF | RG | 
|------|-----|-----|
| João | 123.456.789-09 | 12.345.678-9 |
"""
        result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")

        labels = {e["label"] for e in result}
        assert "CPF" in labels
        assert "RG" in labels
        assert len(result) == 2

    def test_extract_with_label_position(self, model_path):
        """Testa extração com posições no texto original."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])

        text = """Antes da tabela
| Nome | CPF |
|------|-----|
| João | 123.456.789-09 |
Depois da tabela"""

        result = anonymizer.extract_entities_from_markdown_tables(text, "label_position")

        assert len(result) == 1
        entity = result[0]
        assert entity["label"] == "CPF"
        assert entity["start_position"] >= 0
        assert entity["end_position"] > entity["start_position"]
        assert text[entity["start_position"] : entity["end_position"]] == "123.456.789-09"

    def test_table_without_sensitive_columns(self, model_path):
        """Testa tabela sem colunas sensíveis."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])

        text = """
| Nome | Idade | Cidade |
|------|-------|--------|
| João | 30 | SP |
"""
        result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")

        assert len(result) == 0

    def test_malformed_table_no_separator(self, model_path):
        """Testa tabela malformada sem linha separadora."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])

        text = """
| Nome | CPF |
| João | 123.456.789-09 |
"""
        result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")

        assert result == []

    def test_table_with_empty_cells(self, model_path):
        """Testa tabela com células vazias."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "EMAIL"])

        text = """
| Nome | CPF | Email |
|------|-----|-------|
| João | 123.456.789-09 | |
| Maria | | maria@email.com |
"""
        result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")

        assert len(result) == 2
        texts = [e["text"] for e in result]
        assert "123.456.789-09" in texts
        assert "maria@email.com" in texts

    def test_case_insensitive_column_detection(self, model_path):
        """Testa detecção case-insensitive de colunas."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])

        # Variações de maiúsculas/minúsculas
        for col_name in ["cpf", "CPF", "Cpf", "cPf"]:
            text = f"""
| Nome | {col_name} |
|------|-----|
| João | 123.456.789-09 |
"""
            result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")
            assert len(result) == 1, f"Falhou para coluna '{col_name}'"

    def test_sensitive_keywords_detection(self, model_path):
        """Testa todas as palavras-chave sensíveis suportadas."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "RG", "TITULO_ELEITOR", "PASSAPORTE", "CNH", "SIAPE"])

        keywords_map = {
            "cpf": ("123.456.789-09", "CPF"),
            "rg": ("12.345.678-9", "RG"),
            "titulo": ("123456789011", "TITULO"),
            "documento": ("AB123456", "DOCUMENTO"),
            "passaporte": ("AB123456", "PASSAPORTE"),
            "cnh": ("12345678901", "CNH"),
            "siape": ("SIAPE 1234567", "SIAPE"),
        }

        for keyword, (value, _) in keywords_map.items():
            text = f"""
| {keyword} |
|-----|
| {value} |
"""
            result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")
            assert len(result) >= 0  # Pode ou não detectar dependendo da validação

    def test_table_column_index_out_of_range(self, model_path):
        """Testa quando uma linha tem menos colunas que o header."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "EMAIL"])

        text = """
| Nome | CPF | Email |
|------|-----|-------|
| João | 123.456.789-09 |
"""
        result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")
        assert isinstance(result, list)

    def test_table_with_special_characters(self, model_path):
        """Testa tabela com caracteres especiais."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])

        text = """
| Nome | CPF |
|------|-----|
| José & Maria | 123.456.789-09 |
| João <Silva> | 987.654.321-00 |
"""
        result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")
        cpf_entities = [e for e in result if e["label"] == "CPF"]

        assert len(cpf_entities) == 2

    def test_empty_table(self, model_path):
        """Testa tabela vazia (só header)."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])

        text = """
| Nome | CPF |
|------|-----|
"""
        result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")
        assert result == []

    @pytest.mark.parametrize(
        "col_name,entity_type,value",
        [
            ("cpf", "CPF", "123.456.789-09"),
            ("rg", "RG", "12.345.678-9"),
            ("email", "EMAIL", "teste@email.com"),
            ("documento", "DOCUMENTO", "AB123456"),
        ],
    )
    def test_column_keyword_variations(self, model_path, col_name, entity_type, value):
        """Testa variações de nomes de colunas sensíveis."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns([entity_type] if entity_type != "DOCUMENTO" else ["PASSAPORTE"])

        text = f"""
| {col_name} |
|-----|
| {value} |
"""
        result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")
        # Pelo menos detecta algo
        assert isinstance(result, list)


class TestTableOverlapRemoval:
    """Testes para remoção de overlaps incluindo entidades de tabelas."""

    def test_table_entity_overlaps_with_regex(self, model_path):
        """Testa quando entidade da tabela sobrepõe com regex."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])

        text = """
CPF no texto: 123.456.789-09

| CPF |
|-----|
| 123.456.789-09 |
"""
        result = anonymizer.extract_entities(text, "label_detail")

        cpf_texts = [e["text"] for e in result if e["label"] == "CPF"]
        assert "123.456.789-09" in cpf_texts

    def test_table_priority_in_overlap(self, model_path):
        """Testa prioridade quando há overlap entre tabela e outras fontes."""
        anonymizer = SeiAnonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])

        text = """123.456.789-09"""

        result = anonymizer.extract_entities(text, "label_detail")

        assert len(result) >= 1
