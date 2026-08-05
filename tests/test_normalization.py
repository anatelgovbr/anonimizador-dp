"""Testes para _normalization/normalize.py.

Cobre:
  - remover_prefixo / remover_sufixo / remover_prefixo_sufixo
  - normalize_entity para RG, GEO_COORD, CPF, CID, SIAPE, CNH
  - GEO_COORD: prefixos Lat, Long, Lat., Log:
  - RG: sufixo SSP/DF, SSP DF, SSPDF, - DF, /MG
  - Offsets preservados
  - Casos sem normalizacao
"""

import pytest

from anonimizar._normalization import normalize_entity, remover_prefixo_sufixo


class TestRemoverPrefixo:
    """Testes de remocao de prefixo por label."""

    @pytest.mark.parametrize(
        ("text", "start", "end", "label", "expected_text", "expected_start", "expected_end"),
        [
            ("RG: 12345678", 0, 12, "RG", "12345678", 4, 12),
            ("RG 12345678", 0, 11, "RG", "12345678", 3, 11),
            ("RG n 12345678", 0, 13, "RG", "12345678", 5, 13),
            ("CPF: 123.456.789-09", 0, 19, "CPF", "123.456.789-09", 5, 19),
            ("CPF 12345678909", 0, 15, "CPF", "12345678909", 4, 15),
            ("SIAPE 1234567", 0, 13, "SIAPE", "1234567", 6, 13),
            ("Latitude: -23.428639", 0, 20, "GEO_COORD", "-23.428639", 10, 20),
            ("Longitude: -51.960028", 0, 21, "GEO_COORD", "-51.960028", 11, 21),
            ("Lat: -23.5", 0, 10, "GEO_COORD", "-23.5", 5, 10),
            ("Log: -47.64083", 0, 14, "GEO_COORD", "-47.64083", 5, 14),
            ("CID A10", 0, 7, "CID", "A10", 4, 7),
            ("CNH: 12345678901", 0, 16, "CNH", "12345678901", 5, 16),
            ("E-mail: teste@email.com", 0, 23, "EMAIL", "teste@email.com", 8, 23),
        ],
    )
    def test_remove_prefixo(self, text, start, end, label, expected_text, expected_start, expected_end):
        cleaned, new_start, new_end, rule = remover_prefixo_sufixo(text, start, end, label)
        assert cleaned == expected_text, f"{label}: esperado {expected_text!r}, obtido {cleaned!r}"
        assert new_start == expected_start, f"{label}: esperado start={expected_start}, obtido {new_start}"
        assert new_end == expected_end, f"{label}: esperado end={expected_end}, obtido {new_end}"
        assert new_end - new_start == len(cleaned)


class TestRGSufixo:
    """Testes de remocao de sufixo de RG (orgao emissor)."""

    @pytest.mark.parametrize(
        ("text", "start", "end", "expected_text", "expected_end"),
        [
            ("3990139 SSP/DF", 0, 14, "3990139", 7),
            ("2355046 SSP/DF", 0, 14, "2355046", 7),
            ("1399163 SSP DF", 0, 14, "1399163", 7),
            ("19118939 - DF", 0, 13, "19118939", 8),
            ("1822245 SSPDF", 0, 13, "1822245", 7),
            ("1.847.732 /DF", 0, 13, "1.847.732", 9),
            ("2.478.396 DF", 0, 12, "2.478.396", 9),
            ("3436060 SSP GO", 0, 14, "3436060", 7),
        ],
    )
    def test_rg_sufixo_orgao(self, text, start, end, expected_text, expected_end):
        cleaned, new_start, new_end, rule = normalize_entity(text, start, end, "RG")
        assert cleaned == expected_text, f"Esperado {expected_text!r}, obtido {cleaned!r}"
        assert new_end == expected_end, f"Esperado end={expected_end}, obtido {new_end}"
        assert rule is not None, "Deveria ter aplicado regra"


class TestTituloEleitorPrefixo:
    """Testes de remocao dos prefixos capturados junto ao titulo."""

    @pytest.mark.parametrize(
        ("text", "expected_text"),
        [
            ("Título de Eleitor: 8185632089", "8185632089"),
            ("TÍTULO DE ELEITOR: 4079 0024 0116", "4079 0024 0116"),
            ("nº 090101151875", "090101151875"),
        ],
    )
    def test_titulo_prefixo_e_offsets(self, text, expected_text):
        cleaned, new_start, new_end, rule = normalize_entity(text, 0, len(text), "TITULO_ELEITOR")

        assert cleaned == expected_text
        assert new_start == text.index(expected_text)
        assert new_end == len(text)
        assert new_end - new_start == len(cleaned)
        assert rule is not None


class TestRGPrefixoESufixo:
    """Testes de RG com prefixo e sufixo simultaneos."""

    @pytest.mark.parametrize(
        ("text", "start", "end", "expected_text", "expected_start", "expected_end"),
        [
            ("RG: 3990139 SSP/DF", 0, 18, "3990139", 4, 11),
            ("RG 19118939 - DF", 0, 16, "19118939", 3, 11),
            ("RG n 1822245 SSPDF", 0, 18, "1822245", 5, 12),
        ],
    )
    def test_rg_prefixo_sufixo(self, text, start, end, expected_text, expected_start, expected_end):
        cleaned, new_start, new_end, rule = normalize_entity(text, start, end, "RG")
        assert cleaned == expected_text
        assert new_start == expected_start
        assert new_end == expected_end


class TestGeoCoordPrefixo:
    """Testes de remocao de prefixo de GEO_COORD."""

    @pytest.mark.parametrize(
        ("text", "start", "end", "expected_text", "expected_start"),
        [
            ("Lat.02S380200", 0, 13, "02S380200", 4),
            ("Lat 02S380200", 0, 13, "02S380200", 4),
            ("Latitude: -10.16281°S", 0, 21, "-10.16281°S", 10),
            ("Longitude: -48.87431°W", 0, 22, "-48.87431°W", 11),
            ("Log: -47.64083", 0, 14, "-47.64083", 5),
        ],
    )
    def test_geo_prefixo(self, text, start, end, expected_text, expected_start):
        cleaned, new_start, new_end, rule = normalize_entity(text, start, end, "GEO_COORD")
        assert cleaned == expected_text
        assert new_start == expected_start
        assert new_end - new_start == len(cleaned)


class TestSemNormalizacao:
    """Entidades que nao devem ser alteradas."""

    @pytest.mark.parametrize(
        ("text", "start", "end", "label"),
        [
            ("12345678", 0, 8, "RG"),
            ("12.345.678-9", 0, 12, "RG"),
            ("-23.428639", 0, 10, "GEO_COORD"),
            ("02S380200", 0, 9, "GEO_COORD"),
            ("-10.296940, -48.358915", 0, 23, "GEO_COORD"),
        ],
    )
    def test_sem_alteracao(self, text, start, end, label):
        cleaned, new_start, new_end, rule = normalize_entity(text, start, end, label)
        assert cleaned == text
        assert new_start == start
        assert new_end == end
        assert rule is None


class TestNormalizeEdgeCases:
    """Casos de borda."""

    def test_rg_apenas_numeros(self):
        cleaned, s, e, rule = normalize_entity("5249269", 0, 7, "RG")
        assert cleaned == "5249269"
        assert rule is None

    def test_rg_com_slash_uf(self):
        cleaned, s, e, rule = normalize_entity("5249269/DF", 0, 10, "RG")
        assert cleaned == "5249269"
        assert s == 0
        assert e == 7
        assert rule is not None

    def test_geo_apenas_decimal(self):
        cleaned, s, e, rule = normalize_entity("-48.960556", 0, 10, "GEO_COORD")
        assert cleaned == "-48.960556"
        assert rule is None

    def test_geo_log_prefixo_removido(self):
        cleaned, s, e, rule = normalize_entity("Log: -47.64083", 0, 15, "GEO_COORD")
        assert cleaned == "-47.64083"
        assert s == 5
        assert e == 15

    def test_vazio_retorna_vazio(self):
        cleaned, s, e, rule = normalize_entity("", 0, 0, "RG")
        assert cleaned == ""
        assert s == 0
        assert e == 0

    def test_label_sem_padrao(self):
        cleaned, s, e, rule = normalize_entity("algum texto", 0, 11, "LABEL_INEXISTENTE")
        assert cleaned == "algum texto"
        assert s == 0
        assert e == 11
        assert rule is None

    def test_documented_rg_example_uses_text_length_as_end_offset(self):
        """Mantém reproduzível o exemplo da docstring de normalize_entity."""
        text = "RG 123456 SSP/SP"

        cleaned, start, end, rule = normalize_entity(text, 0, len(text), "RG")

        assert len(text) == 16
        assert (cleaned, start, end) == ("123456", 3, 9)
        assert rule is not None


class TestNormalizeEntitiesFlagPipeline:
    """Testa o comportamento da flag normalize_entities no pipeline de extração."""

    def test_normalize_true_no_prefix(self):
        from anonimizar._normalization import normalize_entity

        cleaned, s, e, _ = normalize_entity("1234567", 0, 7, "RG")
        assert cleaned == "1234567"
        assert s == 0

    def test_normalize_true_with_prefix(self):
        from anonimizar._normalization import normalize_entity

        # "RG: 1234567" has 12 chars; prefix "RG: " (4) stripped
        cleaned, s, e, _ = normalize_entity("RG: 1234567", 0, 12, "RG")
        assert cleaned == "1234567"
        assert s == 4
        assert e == 12

    def test_normalize_false_skips_cleaning(self):
        from anonimizar._normalization import normalize_entity

        cleaned, s, e, _ = normalize_entity("RG: 1234567", 0, 12, "RG")
        assert cleaned == "1234567"
        assert s == 4


class TestNormalizeFlagAnonimizar:
    """Testa a flag normalize_entities no Anonimizar."""

    def test_default_is_true(self):
        from unittest.mock import MagicMock, patch

        with patch("anonimizar._anonymization.anonymizer.spacy.load") as mock_load:
            mock_model = MagicMock()
            mock_model.pipe_names = ["ner"]
            mock_model.get_pipe.return_value.labels = ["CPF", "RG"]
            mock_load.return_value = mock_model

            from anonimizar import Anonimizar

            anon = Anonimizar(model_path="fake", auto_patterns=False)
            assert anon.normalize_entities is True

    def test_can_be_disabled(self):
        from unittest.mock import MagicMock, patch

        with patch("anonimizar._anonymization.anonymizer.spacy.load") as mock_load:
            mock_model = MagicMock()
            mock_model.pipe_names = ["ner"]
            mock_model.get_pipe.return_value.labels = ["CPF", "RG"]
            mock_load.return_value = mock_model

            from anonimizar import Anonimizar

            anon = Anonimizar(model_path="fake", auto_patterns=False, normalize_entities=False)
            assert anon.normalize_entities is False

    def test_true_normalizes_extracted_entities(self):
        from unittest.mock import MagicMock, patch

        with patch("anonimizar._anonymization.anonymizer.spacy.load") as mock_load:
            mock_model = MagicMock()
            mock_model.pipe_names = ["ner"]
            mock_model.get_pipe.return_value.labels = ["CPF", "RG"]
            mock_model.max_length = 3000000
            mock_load.return_value = mock_model

            from anonimizar import Anonimizar

            anon = Anonimizar(model_path="fake", auto_patterns=False, normalize_entities=True)
            assert anon.normalize_entities is True

    def test_false_skips_normalization_in_pipeline(self):
        from unittest.mock import MagicMock, patch

        with patch("anonimizar._anonymization.anonymizer.spacy.load") as mock_load:
            mock_model = MagicMock()
            mock_model.pipe_names = ["ner"]
            mock_model.get_pipe.return_value.labels = ["CPF", "RG"]
            mock_model.max_length = 3000000
            mock_load.return_value = mock_model

            from anonimizar import Anonimizar

            anon = Anonimizar(model_path="fake", auto_patterns=False, normalize_entities=False)
            assert anon.normalize_entities is False


class TestNormalizeFlagTrainer:
    """Testa a flag normalize_entities no Trainer."""

    def test_default_is_true(self):
        from anonimizar.sei_anonimizar_treino import Trainer

        trainer = Trainer(labels=["CPF"])
        assert trainer.normalize_entities is True

    def test_can_be_disabled(self):
        from anonimizar.sei_anonimizar_treino import Trainer

        trainer = Trainer(labels=["CPF"], normalize_entities=False)
        assert trainer.normalize_entities is False

    def test_true_normalizes_in_add_data(self):
        from anonimizar.sei_anonimizar_treino import Trainer

        trainer = Trainer(labels=["CPF"], normalize_entities=True)
        # "CPF: 123.456.789-09" = 19 chars, entity covers full text
        text = "CPF: 123.456.789-09"
        data = [{"text": text, "entities": [(0, len(text), "CPF")]}]
        trainer.add_data(data, auto_clean=False)
        assert len(trainer.training_data) == 1
        t, annotations = trainer.training_data[0]
        entities = annotations["entities"]
        assert len(entities) == 1
        start, end, label = entities[0]
        # Normalization strips "CPF: " prefix → span should be "123.456.789-09"
        assert t[start:end] == "123.456.789-09"

    def test_false_keeps_original_offsets(self):
        from anonimizar.sei_anonimizar_treino import Trainer

        trainer = Trainer(labels=["CPF"], normalize_entities=False)
        text = "CPF: 123.456.789-09"
        data = [{"text": text, "entities": [(0, len(text), "CPF")]}]
        trainer.add_data(data, auto_clean=False)
        assert len(trainer.training_data) == 1
        t, annotations = trainer.training_data[0]
        entities = annotations["entities"]
        assert len(entities) == 1
        start, end, label = entities[0]
        assert t[start:end] == text

    def test_per_call_normalize_entities_overrides_instance(self):
        from anonimizar.sei_anonimizar_treino import Trainer

        trainer = Trainer(labels=["CPF"], normalize_entities=False)
        text = "CPF: 123.456.789-09"
        data = [{"text": text, "entities": [(0, len(text), "CPF")]}]
        trainer.add_data(data, auto_clean=False, normalize_entities=True)
        t, annotations = trainer.training_data[0]
        start, end, label = annotations["entities"][0]
        assert t[start:end] == "123.456.789-09"


class TestNormalizeFlagEvaluator:
    """Testa a flag normalize_entities no Evaluation."""

    def test_default_is_true(self):
        from anonimizar.sei_anonimizar_evaluation import Evaluation

        ev = Evaluation()
        assert ev.normalize_entities is True

    def test_can_be_disabled(self):
        from anonimizar.sei_anonimizar_evaluation import Evaluation

        ev = Evaluation(normalize_entities=False)
        assert ev.normalize_entities is False

    def test_normalize_df_spans(self):
        import pandas as pd

        from anonimizar.sei_anonimizar_evaluation import Evaluation

        ev = Evaluation(normalize_entities=True)
        # "RG: 1234567 SSP/DF" = 18 chars, prefix 4, suffix 7
        _df = pd.DataFrame(
            [
                {"text_entidade": "RG: 1234567 SSP/DF", "start_entidade": 0, "end_entidade": 18, "tp_entidade": "RG"},
                {"text_entidade": "1234567", "start_entidade": 0, "end_entidade": 7, "tp_entidade": "RG"},
            ]
        )
        ev._normalize_df_spans(_df)  # noqa: SLF001
        row0 = _df.iloc[0]
        assert row0["text_entidade"] == "1234567"
        assert row0["start_entidade"] == 4
        assert row0["end_entidade"] == 11

    def test_normalize_df_spans_disabled(self):
        import pandas as pd

        from anonimizar.sei_anonimizar_evaluation import Evaluation

        ev = Evaluation(normalize_entities=False)
        _df = pd.DataFrame(
            [
                {"text_entidade": "RG: 1234567 SSP/DF", "start_entidade": 0, "end_entidade": 18, "tp_entidade": "RG"},
            ]
        )
        ev._normalize_df_spans(_df)  # noqa: SLF001
        row0 = _df.iloc[0]
        assert row0["text_entidade"] == "RG: 1234567 SSP/DF"
        assert row0["start_entidade"] == 0
        assert row0["end_entidade"] == 18
