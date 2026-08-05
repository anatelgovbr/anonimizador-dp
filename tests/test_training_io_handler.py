"""Testes para módulo _training/io_handler.py."""

import json
from pathlib import Path

import pytest

from anonimizar._training.io_handler import (
    load_from_doccano_jsonl,
    load_jsonl_to_dataframes,
    save_to_doccano_jsonl,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestLoadFromDoccanoJSONL:
    """Testes para função load_from_doccano_jsonl()."""

    def test_load_basic_labels_format(self, tmp_path):
        """Testa carregamento de arquivo JSONL no formato 'labels'."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"text": "CPF 123.456.789-09", "labels": [[4, 18, "CPF"]]}\n'
            '{"text": "Email: teste@exemplo.com", "labels": [[7, 24, "EMAIL"]]}\n',
            encoding="utf-8",
        )

        data = load_from_doccano_jsonl(jsonl_file)

        assert len(data) == 2
        assert data[0][0] == "CPF 123.456.789-09"
        assert data[0][1] == {"entities": [(4, 18, "CPF")]}

    def test_load_entities_format(self, tmp_path):
        """Testa carregamento de arquivo JSONL no formato 'entities' (start_offset)."""
        jsonl_file = tmp_path / "test_entities.jsonl"
        jsonl_file.write_text(
            '{"text": "CPF 123", "entities": [{"start_offset": 4, "end_offset": 7, "label": "CPF"}]}\n',
            encoding="utf-8",
        )

        data = load_from_doccano_jsonl(jsonl_file)

        assert len(data) == 1
        assert data[0][1]["entities"] == [(4, 7, "CPF")]

    def test_load_file_not_found_raises(self):
        """Testa que arquivo inexistente levanta FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_from_doccano_jsonl("arquivo_que_nao_existe.jsonl")

    def test_load_sample_fixture(self):
        """Testa carregamento do fixture de exemplo."""
        data = load_from_doccano_jsonl(FIXTURES_DIR / "sample_doccano.jsonl")

        assert len(data) == 3
        texts = [item[0] for item in data]
        assert "CPF 123.456.789-09" in texts
        assert "Email: teste@exemplo.com" in texts

    def test_load_returns_spacy_format(self, tmp_path):
        """Testa que retorna formato spaCy (list of tuples)."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"text": "CPF 123", "labels": [[4, 7, "CPF"]]}\n',
            encoding="utf-8",
        )

        data = load_from_doccano_jsonl(jsonl_file)

        assert isinstance(data, list)
        assert isinstance(data[0], tuple)
        assert len(data[0]) == 2
        assert isinstance(data[0][0], str)
        assert isinstance(data[0][1], dict)
        assert "entities" in data[0][1]

    def test_load_skips_empty_text_lines(self, tmp_path):
        """Testa que linhas com texto vazio são puladas."""
        jsonl_file = tmp_path / "with_empty.jsonl"
        jsonl_file.write_text(
            '{"text": "Valid text", "labels": []}\n{"text": "", "labels": []}\n{"text": "Also valid", "labels": []}\n',
            encoding="utf-8",
        )

        data = load_from_doccano_jsonl(jsonl_file)

        assert len(data) == 2

    def test_load_empty_entities(self, tmp_path):
        """Testa carregamento de linha com entidades vazias."""
        jsonl_file = tmp_path / "empty_labels.jsonl"
        jsonl_file.write_text(
            '{"text": "Texto sem entidades", "labels": []}\n',
            encoding="utf-8",
        )

        data = load_from_doccano_jsonl(jsonl_file)

        assert len(data) == 1
        assert data[0][1]["entities"] == []

    def test_load_path_object(self, tmp_path):
        """Testa que aceita Path object além de string."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"text": "Texto", "labels": []}\n', encoding="utf-8")

        data = load_from_doccano_jsonl(jsonl_file)
        assert len(data) == 1


class TestSaveToDoccanoJSONL:
    """Testes para função save_to_doccano_jsonl()."""

    def test_save_basic(self, tmp_path):
        """Testa salvamento básico de dados."""
        output_file = tmp_path / "output.jsonl"
        data = [
            ("CPF 123.456.789-09", {"entities": [(4, 18, "CPF")]}),
            ("Email: teste@exemplo.com", {"entities": [(7, 24, "EMAIL")]}),
        ]

        save_to_doccano_jsonl(output_file, data)

        assert output_file.exists()
        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_save_and_reload(self, tmp_path):
        """Testa que salvar e recarregar preserva os dados."""
        output_file = tmp_path / "roundtrip.jsonl"
        original = [
            ("CPF 123.456.789-09", {"entities": [(4, 18, "CPF")]}),
            ("Email: teste@exemplo.com", {"entities": [(7, 24, "EMAIL")]}),
        ]

        save_to_doccano_jsonl(output_file, original)
        reloaded = load_from_doccano_jsonl(output_file)

        assert len(reloaded) == len(original)
        assert reloaded[0][0] == original[0][0]

    def test_save_creates_parent_directory(self, tmp_path):
        """Testa que cria diretório pai se não existir."""
        output_file = tmp_path / "subdir" / "nested" / "output.jsonl"
        data = [("Texto", {"entities": []})]

        save_to_doccano_jsonl(output_file, data)

        assert output_file.exists()

    def test_save_entities_format(self, tmp_path):
        """Testa salvamento no formato 'entities'."""
        output_file = tmp_path / "entities_format.jsonl"
        data = [("CPF 123", {"entities": [(4, 7, "CPF")]})]

        save_to_doccano_jsonl(output_file, data, format_type="entities")

        line = json.loads(output_file.read_text(encoding="utf-8").strip())
        assert "entities" in line
        assert line["entities"][0]["start_offset"] == 4

    def test_save_invalid_format_raises(self, tmp_path):
        """Testa que formato inválido levanta ValueError."""
        output_file = tmp_path / "output.jsonl"
        data = [("Texto", {"entities": []})]

        with pytest.raises(ValueError, match="format_type"):
            save_to_doccano_jsonl(output_file, data, format_type="invalid_format")

    def test_save_empty_entities(self, tmp_path):
        """Testa salvamento com entidades vazias."""
        output_file = tmp_path / "empty.jsonl"
        data = [("Texto sem entidades", {"entities": []})]

        save_to_doccano_jsonl(output_file, data)

        line = json.loads(output_file.read_text(encoding="utf-8").strip())
        assert line["text"] == "Texto sem entidades"
        assert line["labels"] == []


class TestLoadJSONLToDataFrames:
    """Testes para função load_jsonl_to_dataframes()."""

    def test_load_returns_two_dataframes(self, tmp_path):
        """Testa que retorna tupla com dois DataFrames."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"text": "CPF 123.456.789-09", "labels": [[4, 18, "CPF"]]}\n',
            encoding="utf-8",
        )

        df_textos, df_entidades = load_jsonl_to_dataframes(jsonl_file)

        assert df_textos is not None
        assert df_entidades is not None

    def test_load_textos_has_id_and_text(self, tmp_path):
        """Testa que df_textos tem colunas id e text."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"text": "CPF 123.456.789-09", "labels": [[4, 18, "CPF"]]}\n',
            encoding="utf-8",
        )

        df_textos, _ = load_jsonl_to_dataframes(jsonl_file)

        assert "id" in df_textos.columns
        assert "text" in df_textos.columns

    def test_load_entidades_has_required_columns(self, tmp_path):
        """Testa que df_entidades tem colunas id, start, end, entidade."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"text": "CPF 123.456.789-09", "labels": [[4, 18, "CPF"]]}\n',
            encoding="utf-8",
        )

        _, df_entidades = load_jsonl_to_dataframes(jsonl_file)

        assert "id" in df_entidades.columns
        assert "start" in df_entidades.columns
        assert "end" in df_entidades.columns
        assert "entidade" in df_entidades.columns

    def test_load_file_not_found_raises(self):
        """Testa que arquivo inexistente levanta FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_jsonl_to_dataframes("nao_existe.jsonl")

    def test_load_fixture(self):
        """Testa carregamento do fixture de exemplo."""
        df_textos, df_entidades = load_jsonl_to_dataframes(FIXTURES_DIR / "sample_doccano.jsonl")

        assert len(df_textos) == 3
        assert len(df_entidades) == 3
