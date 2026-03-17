"""Testes para _training/io.py.

Cobre branches e funções não exercitados pelos testes existentes:
- load_doccano_jsonl: linha vazia, texto vazio, JSON inválido, formato entities
- _extract_entities_from_record: labels inválido, sem chaves
- save_to_doccano_jsonl: formato 'entities', formato inválido
- load_jsonl_to_dataframes: formato entities e labels
- load_cv_input_data: branches de string e conversão
- _extract_entities_for_dataframe: formato entities
"""

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from anonimizar._training.io import (
    _extract_entities_for_dataframe,
    _extract_entities_from_record,
    load_cv_input_data,
    load_doccano_jsonl,
    load_jsonl_to_dataframes,
    save_to_doccano_jsonl,
)


@pytest.fixture
def logger():
    """Logger silencioso para testes."""
    log = logging.getLogger("test_training_io")
    log.setLevel(logging.DEBUG)
    return log


@pytest.fixture
def jsonl_labels(tmp_path) -> Path:
    """Arquivo JSONL com formato 'labels'."""
    f = tmp_path / "labels.jsonl"
    records = [
        {"text": "CPF 529.982.247-25", "labels": [[4, 18, "CPF"]]},
        {"text": "Email joao@email.com aqui", "labels": [[6, 20, "EMAIL"]]},
    ]
    f.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return f


@pytest.fixture
def jsonl_entities(tmp_path) -> Path:
    """Arquivo JSONL com formato 'entities' (Doccano)."""
    f = tmp_path / "entities.jsonl"
    records = [
        {
            "text": "RG 12.345.678-9",
            "entities": [{"start_offset": 3, "end_offset": 15, "label": "RG"}],
        },
        {
            "text": "Email jose@test.com aqui",
            "entities": [{"start": 6, "end": 19, "label": "EMAIL"}],
        },
    ]
    f.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return f


# =============================================================================
# load_doccano_jsonl
# =============================================================================


class TestLoadDoccanoJSONL:
    """Testa load_doccano_jsonl - linhas 49, 61, 65-68."""

    def test_arquivo_nao_encontrado(self, tmp_path) -> None:
        """FileNotFoundError quando arquivo não existe."""
        with pytest.raises(FileNotFoundError):
            load_doccano_jsonl(tmp_path / "nao_existe.jsonl")

    def test_carrega_formato_labels(self, jsonl_labels) -> None:
        """Carrega arquivo no formato 'labels'."""
        data = load_doccano_jsonl(jsonl_labels)
        assert len(data) == 2
        assert data[0][0] == "CPF 529.982.247-25"
        assert data[0][1]["entities"] == [(4, 18, "CPF")]

    def test_carrega_formato_entities(self, jsonl_entities) -> None:
        """Carrega arquivo no formato 'entities' do Doccano."""
        data = load_doccano_jsonl(jsonl_entities)
        assert len(data) == 2

    def test_ignora_linha_vazia(self, tmp_path) -> None:
        """Linhas vazias são ignoradas."""
        f = tmp_path / "test.jsonl"
        f.write_text('{"text": "CPF", "labels": []}\n\n{"text": "RG", "labels": []}', encoding="utf-8")
        data = load_doccano_jsonl(f)
        assert len(data) == 2

    def test_ignora_texto_vazio(self, tmp_path) -> None:
        """Registros com texto vazio são ignorados."""
        f = tmp_path / "test.jsonl"
        f.write_text('{"text": "", "labels": []}\n{"text": "ok", "labels": []}', encoding="utf-8")
        data = load_doccano_jsonl(f)
        assert len(data) == 1

    def test_ignora_json_invalido(self, tmp_path) -> None:
        """Linhas com JSON inválido são ignoradas."""
        f = tmp_path / "test.jsonl"
        f.write_text('NOT_JSON\n{"text": "valido", "labels": []}', encoding="utf-8")
        data = load_doccano_jsonl(f)
        assert len(data) == 1

    def test_ignora_sem_labels_ou_entities(self, tmp_path) -> None:
        """Registros sem 'labels' ou 'entities' retornam None."""
        f = tmp_path / "test.jsonl"
        f.write_text('{"text": "texto sem campo", "other": []}', encoding="utf-8")
        data = load_doccano_jsonl(f)
        assert data == []


# =============================================================================
# _extract_entities_from_record
# =============================================================================


class TestExtractEntitiesFromRecord:
    """Testa _extract_entities_from_record - linhas 90-91, 107-108, 113-114."""

    def test_formato_labels_invalido_retorna_none(self, logger) -> None:
        """'labels' não sendo lista retorna None."""
        data = {"text": "teste", "labels": "invalido"}
        result = _extract_entities_from_record(data, 1, logger)
        assert result is None

    def test_sem_labels_ou_entities_retorna_none(self, logger) -> None:
        """Sem 'labels' ou 'entities' retorna None."""
        data = {"text": "teste", "outro": []}
        result = _extract_entities_from_record(data, 1, logger)
        assert result is None

    def test_formato_entities_start_offset(self, logger) -> None:
        """Formato entities com start_offset é processado."""
        data = {"text": "CPF aqui", "entities": [{"start_offset": 4, "end_offset": 8, "label": "CPF"}]}
        result = _extract_entities_from_record(data, 1, logger)
        assert result == [(4, 8, "CPF")]

    def test_formato_entities_start_sem_offset(self, logger) -> None:
        """Formato entities com chave 'start' (sem _offset) é processado."""
        data = {"text": "CPF aqui", "entities": [{"start": 4, "end": 8, "label": "CPF"}]}
        result = _extract_entities_from_record(data, 1, logger)
        assert result == [(4, 8, "CPF")]

    def test_formato_entities_como_lista(self, logger) -> None:
        """Formato entities como lista de listas."""
        data = {"text": "CPF aqui", "entities": [[4, 8, "CPF"]]}
        result = _extract_entities_from_record(data, 1, logger)
        assert result == [(4, 8, "CPF")]


# =============================================================================
# save_to_doccano_jsonl
# =============================================================================


class TestSaveToDoccanoJSONL:
    """Testa save_to_doccano_jsonl - linhas 200-201, 208-211."""

    def test_formato_invalido_lanca_erro(self, tmp_path) -> None:
        """Formato inválido lança ValueError."""
        with pytest.raises(ValueError, match="format_type inválido"):
            save_to_doccano_jsonl(tmp_path / "out.jsonl", [], format_type="xml")

    def test_salva_formato_labels(self, tmp_path) -> None:
        """Salva no formato 'labels'."""
        data = [("texto", {"entities": [(0, 5, "CPF")]})]
        out = tmp_path / "out.jsonl"
        save_to_doccano_jsonl(out, data, format_type="labels")
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        record = json.loads(lines[0])
        assert "labels" in record

    def test_salva_formato_entities(self, tmp_path) -> None:
        """Salva no formato 'entities' do Doccano."""
        data = [("texto", {"entities": [(0, 5, "CPF")]})]
        out = tmp_path / "out.jsonl"
        save_to_doccano_jsonl(out, data, format_type="entities")
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        record = json.loads(lines[0])
        assert "entities" in record
        assert record["entities"][0]["start_offset"] == 0

    def test_cria_diretorio_pai(self, tmp_path) -> None:
        """Cria diretório pai se não existir."""
        out = tmp_path / "subdir" / "out.jsonl"
        save_to_doccano_jsonl(out, [("t", {"entities": []})], format_type="labels")
        assert out.exists()


# =============================================================================
# load_jsonl_to_dataframes
# =============================================================================


class TestLoadJSONLToDataFrames:
    """Testa load_jsonl_to_dataframes - linhas 254-267, 270-283."""

    def test_arquivo_nao_encontrado(self, tmp_path) -> None:
        """FileNotFoundError quando arquivo não existe."""
        with pytest.raises(FileNotFoundError):
            load_jsonl_to_dataframes(tmp_path / "nao_existe.jsonl")

    def test_carrega_formato_labels(self, jsonl_labels) -> None:
        """Carrega formato labels para DataFrames."""
        df_textos, _df_entidades = load_jsonl_to_dataframes(jsonl_labels)
        assert len(df_textos) == 2
        assert "id" in df_textos.columns
        assert "text" in df_textos.columns

    def test_carrega_formato_entities(self, jsonl_entities) -> None:
        """Carrega formato entities para DataFrames."""
        df_textos, _df_entidades = load_jsonl_to_dataframes(jsonl_entities)
        assert len(df_textos) == 2

    def test_ignora_texto_vazio(self, tmp_path) -> None:
        """Texto vazio é ignorado."""
        f = tmp_path / "test.jsonl"
        f.write_text('{"text": "", "labels": []}\n{"text": "ok", "labels": []}', encoding="utf-8")
        df_textos, _ = load_jsonl_to_dataframes(f)
        assert len(df_textos) == 1


# =============================================================================
# load_cv_input_data
# =============================================================================


class TestLoadCVInputData:
    """Testa load_cv_input_data - linhas 254-283 (branches de string)."""

    def test_formato_nao_suportado_lanca_erro(self, tmp_path, logger) -> None:
        """Arquivo com extensão não .jsonl lança ValueError."""
        with pytest.raises(ValueError, match="Formato não suportado"):
            load_cv_input_data(str(tmp_path / "arquivo.csv"), None, logger)

    def test_carrega_jsonl_sem_df_textos(self, jsonl_labels, logger) -> None:
        """Carrega JSONL como df_entidades quando df_textos=None."""
        df_textos, df_entidades = load_cv_input_data(str(jsonl_labels), None, logger)
        assert df_textos is not None
        assert df_entidades is not None

    def test_carrega_dois_jsonl(self, jsonl_labels, tmp_path, logger) -> None:
        """Carrega dois arquivos JSONL."""
        jsonl2 = tmp_path / "labels2.jsonl"
        jsonl2.write_text(json.dumps({"text": "RG 12345", "labels": [[3, 8, "RG"]]}), encoding="utf-8")
        df_textos, _df_entidades = load_cv_input_data(str(jsonl_labels), str(jsonl2), logger)
        assert df_textos is not None

    def test_df_textos_como_string_jsonl(self, jsonl_labels, logger) -> None:
        """df_textos como string JSONL enquanto df_entidades ja e DataFrame."""
        df_entidades = pd.DataFrame({"id": [1], "start": [0], "end": [5], "entidade": ["CPF"]})
        df_textos, _ = load_cv_input_data(df_entidades, str(jsonl_labels), logger)
        assert df_textos is not None

    def test_df_textos_formato_invalido_lanca_erro(self, tmp_path, logger) -> None:
        """df_textos como string nao .jsonl lanca ValueError."""
        df_entidades = pd.DataFrame({"id": [], "start": [], "end": [], "entidade": []})
        with pytest.raises(ValueError, match="Formato não suportado"):
            load_cv_input_data(df_entidades, str(tmp_path / "arquivo.csv"), logger)


# =============================================================================
# _extract_entities_for_dataframe
# =============================================================================


class TestExtractEntitiesForDataframe:
    """Testa _extract_entities_for_dataframe - linhas 309-332."""

    def test_formato_labels(self) -> None:
        """Extrai entidades do formato 'labels'."""
        data = {"labels": [[4, 18, "CPF"]]}
        result = _extract_entities_for_dataframe(data, 1)
        assert len(result) == 1
        assert result[0]["start"] == 4
        assert result[0]["entidade"] == "CPF"

    def test_formato_entities_start_offset(self) -> None:
        """Extrai entidades com start_offset."""
        data = {"entities": [{"start_offset": 3, "end_offset": 10, "label": "RG"}]}
        result = _extract_entities_for_dataframe(data, 1)
        assert len(result) == 1
        assert result[0]["start"] == 3

    def test_formato_entities_start_sem_offset(self) -> None:
        """Extrai entidades com chave 'start' (sem _offset)."""
        data = {"entities": [{"start": 3, "end": 10, "label": "RG"}]}
        result = _extract_entities_for_dataframe(data, 1)
        assert len(result) == 1

    def test_formato_entities_como_lista(self) -> None:
        """Extrai entidades de entities como lista de listas."""
        data = {"entities": [[3, 10, "RG"]]}
        result = _extract_entities_for_dataframe(data, 1)
        assert len(result) == 1

    def test_sem_labels_ou_entities_retorna_vazio(self) -> None:
        """Sem 'labels' ou 'entities', retorna lista vazia."""
        data = {"text": "sem campos de entidade"}
        result = _extract_entities_for_dataframe(data, 1)
        assert result == []
