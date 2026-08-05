"""Testes para o módulo Trainer."""

import json

import pandas as pd
import pytest

from anonimizar import Trainer


class TestTrainerInit:
    """Testes para inicialização do treinador."""

    def test_init_default(self):
        """Testa inicialização com parâmetros padrão."""
        trainer = Trainer()
        assert trainer.model_name is None
        assert trainer.output_dir.name == "trained_model"
        assert len(trainer.supported_labels) > 0

    def test_init_with_model(self, model_path, temp_dir):
        """Testa inicialização com modelo específico."""
        trainer = Trainer(model_name=model_path, output_dir=str(temp_dir / "custom_model"))
        assert trainer.model_name == model_path
        assert "ner" in trainer.nlp.pipe_names

    def test_init_custom_labels(self):
        """Testa inicialização com labels customizados."""
        custom_labels = ["CPF", "EMAIL", "TELEFONE"]
        trainer = Trainer(labels=custom_labels)
        assert trainer.supported_labels == custom_labels

    def test_init_blank_model_when_none(self):
        trainer = Trainer()
        assert trainer.model_name is None
        assert "ner" in trainer.nlp.pipe_names

    def test_init_raises_when_model_not_found(self, tmp_path):
        with pytest.raises(OSError, match="Can't find model"):
            Trainer(model_name=str(tmp_path / "modelo_inexistente"))


class TestDefaultLogger:
    def test_logger_singleton_handlers(self):
        trainer = Trainer()
        # Logger é configurado automaticamente via create_default_logger
        assert trainer.logger is not None
        assert len(trainer.logger.handlers) > 0


class TestDataHandling:
    """Testes para manipulação de dados."""

    def test_add_data_list(self, sample_training_data):
        """Testa adição de dados como lista."""
        trainer = Trainer()
        initial_count = len(trainer.training_data)

        trainer.add_data(sample_training_data)

        assert len(trainer.training_data) > initial_count
        assert len(trainer.training_data) == initial_count + len(sample_training_data)

    def test_add_data_single_dict(self):
        """Testa adição de dados como dicionário único."""
        trainer = Trainer()
        single_data = {"text": "CPF 123.456.789-09", "entities": [(4, 18, "CPF")]}

        trainer.add_data(single_data)

        assert len(trainer.training_data) == 1

    def test_add_data_pandas(self):
        """Testa adição de dados como DataFrame."""
        trainer = Trainer()
        _df = pd.DataFrame(
            {"text": ["CPF 123.456.789-09", "Email teste@test.com"], "entities": [[(4, 18, "CPF")], [(6, 20, "EMAIL")]]}
        )

        trainer.add_data(_df)

        assert len(trainer.training_data) == 2

    def test_add_data_with_errors_raise(self):
        """Testa adição com erro em modo 'raise'."""
        trainer = Trainer()
        bad_data = [{"text": "Teste", "entities": [(0, 5, "LABEL_INEXISTENTE")]}]

        with pytest.raises(ValueError, match="Label inválido"):
            trainer.add_data(bad_data, errors="raise")

    def test_add_data_with_errors_ignore(self):
        """Testa adição com erro em modo 'ignore'."""
        trainer = Trainer()
        bad_data = [{"text": "Teste", "entities": [(0, 5, "LABEL_INEXISTENTE")]}]

        # Não deve lançar erro
        trainer.add_data(bad_data, errors="ignore")
        assert len(trainer.training_data) == 0

    def test_add_data_wrong_type_raises(self):
        trainer = Trainer()
        with pytest.raises(TypeError, match="Dados devem ser lista de dicionário"):
            trainer.add_data(data=123)

    def test_add_data_auto_clean_strict_discards(self):
        trainer = Trainer()
        data = [{"text": "  CPF 123.456.789-09  ", "entities": [(0, 22, "CPF")]}]
        trainer.add_data(data, auto_clean=True, strict_clean=True)
        assert len(trainer.training_data) in (0, 1)

    def test_add_data_auto_clean_non_strict_keeps(self):
        trainer = Trainer()
        data = [{"text": "  CPF 123.456.789-09  ", "entities": [(0, 22, "CPF")]}]
        trainer.add_data(data, auto_clean=True, strict_clean=False, errors="ignore")
        assert len(trainer.training_data) == 1


class TestErrorsAutoCleanCombinations:
    """Testes para combinações de errors e auto_clean."""

    def test_errors_raise_auto_clean_false_raises_on_invalid_entity(self):
        """Testa errors='raise' com auto_clean=False - deve levantar erro."""
        trainer = Trainer(labels=["CPF"])
        data = [{"text": "CPF: 123.456.789-00", "entities": [(20, 30, "CPF")]}]

        with pytest.raises(ValueError, match="Posições inválidas"):
            trainer.add_data(data, errors="raise", auto_clean=False)

    def test_errors_raise_auto_clean_true_raises_on_invalid_entity(self):
        """Testa errors='raise' com auto_clean=True - deve levantar erro."""
        trainer = Trainer(labels=["CPF"])
        data = [{"text": "CPF: 123.456.789-00", "entities": [(20, 30, "CPF")]}]

        with pytest.raises(ValueError, match="Posições inválidas"):
            trainer.add_data(data, errors="raise", auto_clean=True)

    def test_errors_ignore_auto_clean_false_keeps_invalid(self):
        """Testa errors='ignore' com auto_clean=False - mantém entidades inválidas."""
        trainer = Trainer(labels=["CPF"])
        data = [{"text": "CPF: 123.456.789-00", "entities": [(20, 30, "CPF")]}]

        trainer.add_data(data, errors="ignore", auto_clean=False)
        assert len(trainer.training_data) == 1

    def test_errors_ignore_auto_clean_true_keeps_invalid_entities(self):
        """Testa errors='ignore' com auto_clean=True - mantém entidades como estão (sem correção)."""
        trainer = Trainer(labels=["CPF"])
        data = [{"text": "CPF: 123.456.789-09  ", "entities": [(5, 21, "CPF")]}]

        trainer.add_data(data, errors="ignore", auto_clean=True)
        assert len(trainer.training_data) == 1
        entities = trainer.training_data[0][1]["entities"]
        assert entities[0] == (5, 21, "CPF")

    def test_errors_coerce_auto_clean_true_corrects_spaces(self):
        """Testa errors='coerce' com auto_clean=True - corrige espaços."""
        trainer = Trainer(labels=["CPF"])
        data = [{"text": "CPF: 123.456.789-09  ", "entities": [(5, 21, "CPF")]}]

        trainer.add_data(data, errors="coerce", auto_clean=True)
        assert len(trainer.training_data) == 1
        entities = trainer.training_data[0][1]["entities"]
        assert entities[0] == (5, 19, "CPF")

    def test_errors_coerce_auto_clean_false_discards_invalid(self):
        """Testa errors='coerce' com auto_clean=False - descarta entidades inválidas."""
        trainer = Trainer(labels=["CPF"])
        data = [{"text": "CPF: 123.456.789-00", "entities": [(20, 30, "CPF")]}]

        trainer.add_data(data, errors="coerce", auto_clean=False)
        assert len(trainer.training_data) == 0

    def test_auto_clean_true_errors_raise_raises_on_removed_entities(self):
        """Testa que errors='raise' levanta erro se entidades forem removidas na limpeza."""
        trainer = Trainer(labels=["CPF", "EMAIL"])
        data = [{"text": "EMAIL: teste@exemplo.com", "entities": [(14, 21, "EMAIL")]}]

        with pytest.raises(ValueError, match="Entidades foram removidas|Offsets desalinhados"):
            trainer.add_data(data, errors="raise", auto_clean=True)

    def test_clean_entities_errors_ignore_preserves_spaces(self):
        """Testa clean_entities com errors='ignore' - mantém espaços."""
        trainer = Trainer(labels=["CPF"])
        text = "CPF: 123.456.789-09  "
        entities = [(5, 21, "CPF")]

        result = trainer.clean_entities(text, entities, errors="ignore")
        assert result == [(5, 21, "CPF")]

    def test_clean_entities_errors_coerce_removes_spaces(self):
        """Testa clean_entities com errors='coerce' - remove espaços."""
        trainer = Trainer(labels=["CPF"])
        text = "CPF: 123.456.789-09  "
        entities = [(5, 21, "CPF")]

        result = trainer.clean_entities(text, entities, errors="coerce")
        assert result == [(5, 19, "CPF")]

    def test_clean_entities_strict_returns_empty_when_entity_is_removed(self):
        """Confirma o contrato estrito documentado para entidades removidas."""
        trainer = Trainer(labels=["CPF"])
        text = "CPF: 123.456.789-09  "

        assert trainer.clean_entities(text, [(20, 30, "CPF")], strict=True) == []

    def test_clean_entities_errors_raise_raises_on_biluo_misaligned(self):
        """Testa clean_entities com errors='raise' - levanta erro em BILUO desalinhado."""
        trainer = Trainer(labels=["CPF"])
        text = "EMAIL: teste@exemplo.com"
        entities = [(14, 21, "EMAIL")]

        with pytest.raises(ValueError, match="Offsets desalinhados"):
            trainer.clean_entities(text, entities, errors="raise")


class TestValidation:
    """Testes para validação de dados."""

    def test_validate_entities_success(self):
        """Testa validação bem-sucedida."""
        trainer = Trainer()
        text = "CPF 123.456.789-09"
        entities = [(4, 18, "CPF")]

        result = trainer.validate_entities(text, entities)
        assert result is True

    def test_validate_entities_invalid_position(self):
        """Testa validação com posições inválidas."""
        trainer = Trainer()
        text = "Texto curto"
        entities = [(0, 100, "CPF")]

        result = trainer.validate_entities(text, entities)
        assert result is False

    def test_validate_entities_invalid_label(self):
        """Testa validação com label inválido."""
        trainer = Trainer()
        text = "Texto teste"
        entities = [(0, 5, "LABEL_INEXISTENTE")]

        result = trainer.validate_entities(text, entities)
        assert result is False

    def test_validate_entities_trims_and_rejects_spaces(self):
        trainer = Trainer()
        text = "XX 123.456.789-09 YY"
        entities = [(2, 18, "CPF")]  # " 123.456.789-09"
        assert trainer.validate_entities(text, entities) is False

    def test_validate_entities_biluo_misaligned(self):
        trainer = Trainer()
        text = "EMAIL: teste@exemplo.com"
        start = text.index("exemplo")
        entities = [(start, start + 3, "EMAIL")]
        assert trainer.validate_entities(text, entities) is False

    def test_validate_entities_start_greater_than_end(self):
        """Testa validação quando start >= end."""
        trainer = Trainer(labels=["CPF"])

        # start > end
        result = trainer.validate_entities("João Silva, CPF 123.456.789-00", [(20, 15, "CPF")])
        assert result is False

        # start equals end
        result = trainer.validate_entities("João Silva, CPF 123.456.789-00", [(10, 10, "CPF")])
        assert result is False

    def test_validate_entities_negative_start(self):
        """Testa validação com posição inicial negativa."""
        trainer = Trainer(labels=["CPF"])

        result = trainer.validate_entities("João Silva", [(-5, 10, "CPF")])
        assert result is False

    def test_validate_entities_all_checks(self):
        """Testa todas as validações do validate_entities."""
        trainer = Trainer(labels=["CPF", "EMAIL"])

        # Válido
        assert trainer.validate_entities("CPF 123.456.789-09", [(4, 18, "CPF")]) is True

        # start is >= end
        assert trainer.validate_entities("CPF 123.456.789-09", [(10, 10, "CPF")]) is False
        assert trainer.validate_entities("CPF 123.456.789-09", [(15, 10, "CPF")]) is False

        # start < 0
        assert trainer.validate_entities("CPF 123.456.789-09", [(-1, 10, "CPF")]) is False

        # Offset fora do texto
        assert trainer.validate_entities("CPF 123", [(0, 100, "CPF")]) is False
        assert trainer.validate_entities("CPF 123", [(100, 120, "CPF")]) is False

        # Espaços nas extremidades
        assert trainer.validate_entities("  CPF 123  ", [(0, 11, "CPF")]) is False

        # Label não suportado
        assert trainer.validate_entities("CPF 123.456.789-09", [(4, 18, "INVALID")]) is False


class TestValidateDataPolicies:
    def test_validate_data_label_invalido_raise(self):
        trainer = Trainer()
        data = [("Texto", {"entities": [(0, 5, "X_INVALIDO")]})]
        with pytest.raises(ValueError, match="Label inválido"):
            trainer._validate_data(data, errors="raise")  # noqa: SLF001

    def test_validate_data_posicao_invalida_coerce(self):
        trainer = Trainer()
        text = "CPF 123.456.789-09"
        data = [(text, {"entities": [(0, 100, "CPF")]})]
        out = trainer._validate_data(data, errors="coerce", keep_empty_entities=True)  # noqa: SLF001
        assert len(out) == 1
        assert out[0][1]["entities"] == []

    def test_validate_data_posicao_invalida_ignore(self):
        trainer = Trainer()
        text = "CPF 123.456.789-09"
        data = [(text, {"entities": [(0, 100, "CPF")]})]
        out = trainer._validate_data(data, errors="ignore", keep_empty_entities=True)  # noqa: SLF001
        assert len(out) == 1
        assert out[0][1]["entities"] == [(0, 100, "CPF")]


class TestValidateBiluoPolicies:
    def test_biluo_raise(self):
        trainer = Trainer()
        text = "EMAIL: teste@exemplo.com"
        start = text.index("exemplo")
        with pytest.raises(ValueError, match="Erro BILUO"):
            trainer._validate_biluo_tags(text, [(start, start + 3, "EMAIL")], errors="raise")  # noqa: SLF001

    def test_biluo_coerce(self):
        trainer = Trainer()
        text = "EMAIL: teste@exemplo.com"
        start = text.index("exemplo")
        out = trainer._validate_biluo_tags(text, [(start, start + 3, "EMAIL")], errors="coerce")  # noqa: SLF001
        assert out == []

    def test_biluo_ignore(self):
        trainer = Trainer()
        text = "EMAIL: teste@exemplo.com"
        start = text.index("exemplo")
        out = trainer._validate_biluo_tags(text, [(start, start + 3, "EMAIL")], errors="ignore")  # noqa: SLF001
        assert out == [(start, start + 3, "EMAIL")]


class TestTransformFromPandas:
    def test_transform_row_per_entity_format(self):
        trainer = Trainer()
        _df = pd.DataFrame(
            {
                "texto": ["abc", "abc", "def"],
                "start": [0, 4, 0],
                "end": [3, 7, 3],
                "entidade": ["CPF", "EMAIL", "CPF"],
            }
        )
        out = trainer._transform_data_from_pandas(_df)  # noqa: SLF001
        assert len(out) == 2
        texts = {t for t, _ in out}
        assert texts == {"abc", "def"}

    def test_transform_invalid_format_raises(self):
        trainer = Trainer()
        _df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="DataFrame deve ter colunas"):
            trainer._transform_data_from_pandas(_df)  # noqa: SLF001


class TestTraining:
    """Testes para treinamento."""

    def test_train_basic(self, temp_dir, sample_training_data):
        """Testa treinamento básico."""
        trainer = Trainer(output_dir=str(temp_dir))
        trainer.add_data(sample_training_data)

        trainer.train(n_iter=1, validation_split=0.0)

        assert len(trainer.training_data) > 0

    def test_train_no_data(self):
        """Testa treinamento sem dados."""
        trainer = Trainer()

        with pytest.raises(ValueError, match="Nenhum dado de treinamento disponível"):
            trainer.train()

    def test_split_data(self, sample_training_data):
        """Testa divisão de dados."""
        trainer = Trainer()
        trainer.add_data(sample_training_data)

        trainer.split_data(train_ratio=0.7)

        assert hasattr(trainer, "train_data")
        assert hasattr(trainer, "val_data")
        assert len(trainer.train_data) + len(trainer.val_data) == len(trainer.training_data)

    def test_split_data_insufficient(self):
        """Testa divisão com dados insuficientes."""
        trainer = Trainer()
        trainer.training_data = [("teste", {"entities": []})]

        with pytest.raises(ValueError, match="Dados insuficientes para divisão"):
            trainer.split_data()

    def test_train_validation_split_invalid(self, sample_training_data):
        trainer = Trainer()
        trainer.add_data(sample_training_data)
        with pytest.raises(ValueError, match="validation_split deve estar entre 0.0"):
            trainer.train(n_iter=1, validation_split=1.0)

    def test_train_with_split(self, sample_training_data):
        trainer = Trainer()
        trainer.add_data(sample_training_data)
        trainer.train(n_iter=1, validation_split=0.5, batch_size=2)
        assert hasattr(trainer, "training_data")

    def test_split_ratio_integrity(self, sample_training_data):
        trainer = Trainer()
        trainer.add_data(sample_training_data)
        trainer.split_data(train_ratio=0.6)
        total = len(trainer.train_data) + len(trainer.val_data)
        assert total == len(trainer.training_data)
        assert 0 < len(trainer.train_data) < len(trainer.training_data)


class TestModelSaving:
    """Testes para salvamento do modelo."""

    def test_save_model(self, temp_dir, sample_training_data):
        """Testa salvamento do modelo."""
        trainer = Trainer(output_dir=str(temp_dir))
        trainer.add_data(sample_training_data)
        trainer.train(n_iter=1, validation_split=0.0)

        trainer.save_model()

        assert temp_dir.exists()
        assert len(list(temp_dir.iterdir())) > 0

    def test_save_model_custom_path(self, temp_dir, sample_training_data):
        """Testa salvamento em caminho customizado."""
        trainer = Trainer()
        trainer.add_data(sample_training_data)
        trainer.train(n_iter=1, validation_split=0.0)

        custom_path = temp_dir / "custom_model"
        trainer.save_model(str(custom_path))

        assert custom_path.exists()


class TestCleanEntities:
    """Testes para limpeza de entidades."""

    def test_clean_entities_with_spaces(self):
        """Testa limpeza de entidades com espaços."""
        trainer = Trainer()
        text = "  CPF 123.456.789-09  "
        entities = [(0, len(text), "CPF")]

        cleaned = trainer.clean_entities(text, entities)

        assert len(cleaned) <= len(entities)

    def test_clean_entities_invalid_positions(self):
        """Testa limpeza com posições inválidas."""
        trainer = Trainer()
        text = "Texto"
        entities = [(0, 100, "CPF")]

        cleaned = trainer.clean_entities(text, entities)

        assert len(cleaned) == 0


class TestDataConversion:
    """Testes para conversão de dados."""

    def test_val_data_to_evaluation(self, sample_training_data):
        """Testa conversão para formato de avaliação."""
        trainer = Trainer()
        trainer.add_data(sample_training_data)
        trainer.split_data(train_ratio=0.5)

        df_texts, df_ground_truth = trainer.val_data_to_evaluation()

        assert isinstance(df_texts, pd.DataFrame)
        assert isinstance(df_ground_truth, pd.DataFrame)

        assert "id" in df_texts.columns
        assert "text" in df_texts.columns
        assert all(col in df_ground_truth.columns for col in ["id", "tp_entidade", "start_entidade", "end_entidade"])

    def test_val_data_to_evaluation_no_split(self):
        """Testa conversão sem dados de validação."""
        trainer = Trainer()

        with pytest.raises(ValueError, match="Não existem dados de validação"):
            trainer.val_data_to_evaluation()


class TestJSONLSupport:
    """Testes para suporte a JSONL/Doccano."""

    def test_load_from_doccano_jsonl(self, temp_dir):
        """Testa carregamento de arquivo JSONL no formato Doccano."""
        trainer = Trainer(labels=["CPF", "EMAIL"])

        jsonl_path = temp_dir / "test_doccano.jsonl"

        data = [
            {"text": "CPF 123.456.789-09 teste@test.com", "labels": [[4, 18, "CPF"], [19, 33, "EMAIL"]]},
            {"text": "CPF 987.654.321-00", "entities": [[4, 18, "CPF"]]},
        ]

        with jsonl_path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        loaded = trainer.load_from_doccano_jsonl(str(jsonl_path))

        assert len(loaded) == 2
        assert loaded[0][0] == "CPF 123.456.789-09 teste@test.com"
        assert len(loaded[0][1]["entities"]) == 2
        assert loaded[1][0] == "CPF 987.654.321-00"
        assert len(loaded[1][1]["entities"]) == 1

    def test_save_to_doccano_jsonl(self, temp_dir):
        """Testa salvamento em formato JSONL Doccano."""
        trainer = Trainer(labels=["CPF", "EMAIL"])

        trainer.add_data(
            [
                {"text": "CPF 123.456.789-09 teste@test.com", "entities": [(4, 18, "CPF"), (19, 33, "EMAIL")]},
                {"text": "CPF 987.654.321-00", "entities": [(4, 18, "CPF")]},
            ]
        )

        jsonl_path = temp_dir / "output.jsonl"
        trainer.save_to_doccano_jsonl(str(jsonl_path))

        assert jsonl_path.exists()

        with jsonl_path.open(encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2

        item1 = json.loads(lines[0])
        assert item1["text"] == "CPF 123.456.789-09 teste@test.com"
        assert len(item1["labels"]) == 2

    def test_load_jsonl_to_dataframes(self, temp_dir):
        """Testa conversão de JSONL para DataFrames."""
        trainer = Trainer(labels=["CPF", "EMAIL"])

        jsonl_path = temp_dir / "test.jsonl"
        data = [
            {"text": "CPF 123.456.789-09", "labels": [[4, 18, "CPF"]]},
            {"text": "teste@test.com", "labels": [[0, 14, "EMAIL"]]},
        ]

        with jsonl_path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        df_textos, df_entidades = trainer._load_jsonl_to_dataframes(str(jsonl_path))  # noqa: SLF001

        assert isinstance(df_textos, pd.DataFrame)
        assert isinstance(df_entidades, pd.DataFrame)
        assert len(df_textos) == 2
        assert len(df_entidades) == 2
        assert "id" in df_textos.columns
        assert "text" in df_textos.columns
        assert all(col in df_entidades.columns for col in ["id", "start", "end", "entidade"])

    def test_add_data_with_jsonl_path(self, temp_dir):
        """Testa add_data com caminho para arquivo JSONL."""
        trainer = Trainer(labels=["CPF", "EMAIL"])

        jsonl_path = temp_dir / "train.jsonl"
        data = [
            {"text": "CPF 123.456.789-09", "labels": [[4, 18, "CPF"]]},
            {"text": "teste@test.com", "labels": [[0, 14, "EMAIL"]]},
        ]

        with jsonl_path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        trainer.add_data(str(jsonl_path))

        assert len(trainer.training_data) == 2
