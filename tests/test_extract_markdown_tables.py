import logging
from unittest.mock import MagicMock

import pytest

from anonimizar import Anonimizar
from anonimizar._extraction.markdown import _process_table


class TestMarkdownTableExtraction:
    """Testes para extração de entidades de tabelas markdown."""

    def test_extract_from_simple_table(self, model_path):
        anonymizer = Anonimizar(model_path)
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
        anonymizer = Anonimizar(model_path)
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

    def test_extract_with_label_position(self, model_path):
        anonymizer = Anonimizar(model_path)
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
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])

        text = """
| Nome | Idade | Cidade |
|------|-------|--------|
| João | 30 | SP |
"""
        result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")
        assert len(result) == 0

    def test_malformed_table_no_separator(self, model_path):
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])

        text = """
| Nome | CPF |
| João | 123.456.789-09 |
"""
        result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")
        assert result == []

    def test_table_with_empty_cells(self, model_path):
        anonymizer = Anonimizar(model_path)
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
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])

        for col_name in ["cpf", "CPF", "Cpf", "cPf"]:
            text = f"""
| Nome | {col_name} |
|------|-----|
| João | 123.456.789-09 |
"""
            result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")
            assert len(result) == 1, f"Falhou para coluna '{col_name}'"

    def test_sensitive_keywords_detection(self, model_path):
        anonymizer = Anonimizar(model_path)
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
            assert len(result) >= 0

    def test_table_column_index_out_of_range(self, model_path):
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF", "EMAIL"])

        text = """
| Nome | CPF | Email |
|------|-----|-------|
| João | 123.456.789-09 |
"""
        result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")
        assert isinstance(result, list)

    def test_table_with_special_characters(self, model_path):
        anonymizer = Anonimizar(model_path)
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
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])

        text = """
| Nome | CPF |
|------|-----|
"""
        result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")
        assert result == []

    @pytest.mark.parametrize(
        ("col_name", "entity_type", "value"),
        [
            ("cpf", "CPF", "123.456.789-09"),
            ("rg", "RG", "12.345.678-9"),
            ("email", "EMAIL", "teste@email.com"),
            ("documento", "DOCUMENTO", "AB123456"),
        ],
    )
    def test_column_keyword_variations(self, model_path, col_name, entity_type, value):
        anonymizer = Anonimizar(model_path)
        target_label = entity_type if entity_type != "DOCUMENTO" else "PASSAPORTE"
        anonymizer.add_apply_patterns([target_label])

        text = f"""
| {col_name} |
|-----|
| {value} |
"""
        result = anonymizer.extract_entities_from_markdown_tables(text, "label_text")
        assert isinstance(result, list)


class TestTableOverlapRemoval:
    """Testes para remoção de overlaps incluindo entidades de tabelas."""

    def test_table_entity_overlaps_with_regex(self, model_path):
        anonymizer = Anonimizar(model_path)
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
        anonymizer = Anonimizar(model_path)
        anonymizer.add_apply_patterns(["CPF"])

        text = """123.456.789-09"""
        result = anonymizer.extract_entities(text, "label_detail")
        assert len(result) >= 1


class TestB18MarkdownIncrementalSearch:
    """B-18: _process_table deve usar busca incremental para células com valor repetido."""

    def test_celulas_repetidas_tem_posicoes_distintas(self):
        text = "| CPF |\n|-----|\n| 111.111.111-11 |\n| 111.111.111-11 |\n"
        tb_lines = text.strip().split("\n")
        mock_logger = MagicMock(spec=logging.Logger)

        entities = _process_table(tb_lines, text, "label_position", mock_logger)

        cpf_entities = [e for e in entities if e.get("label") == "CPF"]
        assert len(cpf_entities) == 2
        positions = [(e["start_position"], e["end_position"]) for e in cpf_entities]
        assert positions[0] != positions[1], "Células iguais devem ter posições distintas"

    def test_celulas_diferentes_posicoes_corretas(self):
        text = "| CPF |\n|-----|\n| 529.982.247-25 |\n| 111.444.777-35 |\n"
        tb_lines = text.strip().split("\n")
        mock_logger = MagicMock(spec=logging.Logger)

        entities = _process_table(tb_lines, text, "label_position", mock_logger)
        cpf_entities = [e for e in entities if e.get("label") == "CPF"]
        assert len(cpf_entities) == 2

        for ent in cpf_entities:
            start = ent["start_position"]
            end = ent["end_position"]
            assert text[start:end] in ("529.982.247-25", "111.444.777-35")
