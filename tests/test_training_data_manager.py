"""Testes para módulo _training/data_manager.py."""

import logging

import pandas as pd
import pytest

from anonimizar._training.data_manager import NERDataManager


class TestNERDataManagerInit:
    """Testes para inicialização do NERDataManager."""

    def test_init_default(self):
        """Testa inicialização com parâmetros padrão."""
        manager = NERDataManager()

        assert manager.training_data == []
        assert manager.train_data == []
        assert manager.val_data == []
        assert manager.supported_labels == []
        assert manager.logger is not None

    def test_init_with_labels(self):
        """Testa inicialização com labels suportados."""
        labels = ["CPF", "EMAIL", "TELEFONE"]
        manager = NERDataManager(supported_labels=labels)

        assert manager.supported_labels == labels

    def test_init_with_custom_logger(self):
        """Testa inicialização com logger customizado."""
        custom_logger = logging.getLogger("test_logger")
        manager = NERDataManager(logger=custom_logger)

        assert manager.logger is custom_logger

    def test_init_empty_labels_list(self):
        """Testa que lista vazia de labels resulta em lista vazia."""
        manager = NERDataManager(supported_labels=[])
        assert manager.supported_labels == []


class TestNERDataManagerAddData:
    """Testes para método add_data()."""

    def test_add_data_from_list_of_dicts(self):
        """Testa adicionar dados de lista de dicts com 'text' e 'entities'."""
        manager = NERDataManager()

        data = [
            {"text": "CPF 123.456.789-09", "entities": [(4, 18, "CPF")]},
            {"text": "Email: teste@exemplo.com", "entities": [(7, 24, "EMAIL")]},
        ]

        manager.add_data(data)

        assert len(manager.training_data) == 2
        assert manager.training_data[0][0] == "CPF 123.456.789-09"

    def test_add_data_from_dict(self):
        """Testa adicionar um único dict."""
        manager = NERDataManager()

        data = {"text": "CPF 123", "entities": [(4, 7, "CPF")]}
        manager.add_data(data)

        assert len(manager.training_data) == 1

    def test_add_data_from_dataframe_text_entities(self):
        """Testa adicionar dados de DataFrame com colunas text/entities."""
        manager = NERDataManager()

        df_test = pd.DataFrame(
            [
                {"text": "CPF 123", "entities": [(4, 7, "CPF")]},
            ]
        )

        manager.add_data(df_test)

        assert len(manager.training_data) == 1

    def test_add_data_from_none_raises(self):
        """Testa que None levanta TypeError."""
        manager = NERDataManager()

        with pytest.raises(TypeError):
            manager.add_data(None)

    def test_add_data_concatenates_multiple_calls(self):
        """Testa que múltiplas chamadas concatenam dados."""
        manager = NERDataManager()

        manager.add_data([{"text": "A", "entities": []}])
        assert len(manager.training_data) == 1

        manager.add_data([{"text": "B", "entities": []}])
        assert len(manager.training_data) == 2

    def test_add_data_from_jsonl_file(self, tmp_path):
        """Testa adicionar dados de arquivo JSONL."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"text": "CPF 123.456.789-09", "labels": [[4, 18, "CPF"]]}\n'
            '{"text": "Email: teste@exemplo.com", "labels": [[7, 24, "EMAIL"]]}\n',
            encoding="utf-8",
        )

        manager = NERDataManager()
        manager.add_data(str(jsonl_file))

        assert len(manager.training_data) == 2

    def test_add_data_from_nonexistent_file_raises(self):
        """Testa que arquivo inexistente levanta FileNotFoundError."""
        manager = NERDataManager()

        with pytest.raises(FileNotFoundError):
            manager.add_data("arquivo_que_nao_existe.jsonl")


class TestNERDataManagerSplitData:
    """Testes para método split_data()."""

    def test_split_data_default_ratio(self):
        """Testa divisão com ratio padrão."""
        manager = NERDataManager()
        data = [{"text": f"Texto {i}", "entities": []} for i in range(10)]
        manager.add_data(data)

        train, val = manager.split_data()

        assert len(train) + len(val) == 10
        assert len(train) > 0
        assert len(val) > 0

    def test_split_data_80_20(self):
        """Testa divisão 80/20."""
        manager = NERDataManager()
        data = [{"text": f"Texto {i}", "entities": []} for i in range(100)]
        manager.add_data(data)

        train, val = manager.split_data(train_ratio=0.8)

        assert len(train) == 80
        assert len(val) == 20

    def test_split_data_no_data_raises(self):
        """Testa que sem dados levanta ValueError."""
        manager = NERDataManager()

        with pytest.raises(ValueError, match="Nenhum dado"):
            manager.split_data()

    def test_split_data_invalid_ratio_above_1_raises(self):
        """Testa que ratio > 1 levanta ValueError."""
        manager = NERDataManager()
        manager.add_data([{"text": "A", "entities": []}])

        with pytest.raises(ValueError, match="train_ratio"):
            manager.split_data(train_ratio=1.5)

    def test_split_data_invalid_ratio_zero_raises(self):
        """Testa que ratio = 0 levanta ValueError."""
        manager = NERDataManager()
        manager.add_data([{"text": "A", "entities": []}])

        with pytest.raises(ValueError, match="train_ratio"):
            manager.split_data(train_ratio=0.0)

    def test_split_data_persists_on_instance(self):
        """Testa que split_data persiste train_data e val_data."""
        manager = NERDataManager()
        data = [{"text": f"Texto {i}", "entities": []} for i in range(10)]
        manager.add_data(data)

        train, val = manager.split_data(train_ratio=0.8)

        assert manager.train_data == train
        assert manager.val_data == val

    def test_split_data_returns_tuple(self):
        """Testa que split_data retorna tuple com dois elementos."""
        manager = NERDataManager()
        manager.add_data([{"text": f"T{i}", "entities": []} for i in range(5)])

        result = manager.split_data(train_ratio=0.8)

        assert isinstance(result, tuple)
        assert len(result) == 2


class TestNERDataManagerProperties:
    """Testes para propriedades do NERDataManager."""

    def test_n_examples_empty(self):
        """Testa n_examples com manager vazio."""
        manager = NERDataManager()
        assert manager.n_examples == 0

    def test_n_examples_after_add(self):
        """Testa n_examples após adicionar dados."""
        manager = NERDataManager()
        manager.add_data([{"text": "A", "entities": []}, {"text": "B", "entities": []}])
        assert manager.n_examples == 2

    def test_is_empty_initially(self):
        """Testa is_empty quando manager está vazio."""
        manager = NERDataManager()
        assert manager.is_empty is True

    def test_is_empty_after_add(self):
        """Testa is_empty após adicionar dados."""
        manager = NERDataManager()
        manager.add_data([{"text": "A", "entities": []}])
        assert manager.is_empty is False
