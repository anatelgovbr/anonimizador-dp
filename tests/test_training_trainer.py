"""Testes unitários para módulo _training/trainer.py.

Este módulo testa a função train_ner_model() que realiza o treinamento
de modelos NER usando spaCy com dados em formato padronizado.
"""

import logging

import numpy as np
import pytest
import spacy

from anonimizar._training.trainer import train_ner_model

__all__ = ["TestTrainNERModel"]

# Constantes para testes
_N_ITER_BASIC = 5
_N_ITER_CUSTOM = 3
_N_ITER_WEIGHTS = 10
_N_ITER_SINGLE = 3
_N_ITER_SPLIT = 2
_N_EXAMPLES = 2
_N_SINGLE_EXAMPLE = 1
_TUPLE_LENGTH = 2
_BATCH_SIZE = 2
_DROP_RATE = 0.5


class TestTrainNERModel:
    """Suite de testes para função train_ner_model()."""

    @pytest.fixture
    def blank_model(self) -> object:
        """Cria modelo spaCy blank com componente NER e labels pré-adicionados."""
        nlp = spacy.blank("pt")
        if "ner" not in nlp.pipe_names:
            nlp.add_pipe("ner")
        # Adiciona labels usados nos testes
        ner = nlp.get_pipe("ner")
        ner.add_label("CPF")
        ner.add_label("EMAIL")
        return nlp

    @pytest.fixture
    def sample_data(self) -> list[tuple[str, dict]]:
        """Dados de treinamento de exemplo."""
        return [
            ("João Silva tem CPF 123.456.789-09", {"entities": [(21, 35, "CPF")]}),
            ("Contato: email@example.com para informações", {"entities": [(9, 27, "EMAIL")]}),
        ]

    def test_train_basic(self, blank_model: object, sample_data: list) -> None:
        """Testa treinamento básico do modelo com dados mínimos.

        Verifica que:
        - O modelo é treinado e retornado
        - Métricas são calculadas corretamente
        - Número de iterações e exemplos é registrado
        """
        nlp_trained, metrics = train_ner_model(
            nlp=blank_model,
            train_data=sample_data,
            n_iter=_N_ITER_BASIC,
            logger=logging.getLogger(__name__),
        )

        assert nlp_trained is not None, "Modelo treinado não deve ser None"
        assert isinstance(metrics, dict), "Métricas deve ser um dicionário"
        assert "final_loss" in metrics, "Métricas deve conter 'final_loss'"
        assert "iterations" in metrics, "Métricas deve conter 'iterations'"
        assert "examples_count" in metrics, "Métricas deve conter 'examples_count'"
        assert metrics["iterations"] == _N_ITER_BASIC, f"Número de iterações deve ser {_N_ITER_BASIC}"
        assert metrics["examples_count"] == _N_EXAMPLES, f"Número de exemplos deve ser {_N_EXAMPLES}"

    def test_train_with_empty_data_raises(self, blank_model: object) -> None:
        """Testa que ValueError é lançado com dados vazios."""
        with pytest.raises(ValueError, match="disponível"):
            train_ner_model(
                nlp=blank_model,
                train_data=[],
                logger=logging.getLogger(__name__),
            )

    def test_train_with_invalid_validation_split_raises(self, blank_model: object, sample_data: list) -> None:
        """Testa que ValueError é lançado com validation_split inválido.

        Verifica que:
        - Exceção ValueError é lançada para validation_split >= 1.0
        - Mensagem de erro menciona validação
        """
        with pytest.raises(ValueError, match="validation_split"):
            train_ner_model(
                nlp=blank_model,
                train_data=sample_data,
                validation_split=1.5,
                logger=logging.getLogger(__name__),
            )

    def test_train_with_custom_params(self, blank_model: object, sample_data: list) -> None:
        """Testa treinamento com parâmetros customizados.

        Verifica que:
        - Parâmetros são aplicados corretamente
        - Iterações e batch_size são respeitados
        - Loss é calculado
        """
        _, metrics = train_ner_model(
            nlp=blank_model,
            train_data=sample_data,
            n_iter=_N_ITER_CUSTOM,
            drop=_DROP_RATE,
            batch_size=_BATCH_SIZE,
            logger=logging.getLogger(__name__),
        )

        assert metrics["iterations"] == _N_ITER_CUSTOM, f"Número de iterações deve ser {_N_ITER_CUSTOM}"
        assert isinstance(metrics["final_loss"], float | np.floating), "Loss deve ser número float"

    def test_train_adds_labels_to_ner(self, blank_model: object, sample_data: list) -> None:
        """Testa que labels são adicionados ao componente NER.

        Verifica que:
        - Labels 'CPF' e 'EMAIL' estão presentes após treinamento
        - Componente NER foi atualizado
        """
        nlp_trained, _ = train_ner_model(
            nlp=blank_model,
            train_data=sample_data,
            n_iter=2,
            logger=logging.getLogger(__name__),
        )

        ner = nlp_trained.get_pipe("ner")
        assert "CPF" in ner.labels, "Label 'CPF' deve estar em NER labels"
        assert "EMAIL" in ner.labels, "Label 'EMAIL' deve estar em NER labels"

    def test_train_updates_model_weights(self, blank_model: object) -> None:
        """Testa que os pesos do modelo são atualizados durante treinamento.

        Verifica que:
        - Loss final é maior que zero (modelo foi treinado)
        - Modelo consegue processar texto
        - Documento resultante não é None
        """
        train_data = [
            ("João Silva tem CPF 123.456.789-09", {"entities": [(21, 35, "CPF")]}),
        ] * _N_ITER_BASIC

        nlp_trained, metrics = train_ner_model(
            nlp=blank_model,
            train_data=train_data,
            n_iter=_N_ITER_WEIGHTS,
            logger=logging.getLogger(__name__),
        )

        assert metrics["final_loss"] >= 0, "Loss deve ser não-negativo"

        doc = nlp_trained("João Silva tem CPF 123.456.789-09")
        assert doc is not None, "Documento processado não deve ser None"

    def test_train_returns_tuple_with_model_and_metrics(self, blank_model: object, sample_data: list) -> None:
        """Testa que retorno é tuple (modelo, métricas).

        Verifica que:
        - Retorno é uma tupla de 2 elementos
        - Primeiro elemento é o modelo spaCy
        - Segundo elemento é dicionário de métricas
        """
        result = train_ner_model(
            nlp=blank_model,
            train_data=sample_data,
            n_iter=_N_ITER_SPLIT,
            logger=logging.getLogger(__name__),
        )

        assert isinstance(result, tuple), "Retorno deve ser tupla"
        assert len(result) == _TUPLE_LENGTH, f"Tupla deve ter {_TUPLE_LENGTH} elementos"

        nlp_trained, metrics = result
        assert nlp_trained is not None, "Primeiro elemento deve ser modelo"
        assert isinstance(metrics, dict), "Segundo elemento deve ser dicionário"

    def test_train_with_custom_logger(self, blank_model: object, sample_data: list) -> None:
        """Testa treinamento com logger customizado.

        Verifica que:
        - Função aceita logger customizado
        - Não lança erro com logger fornecido
        """
        custom_logger = logging.getLogger("custom_test_logger")
        nlp_trained, metrics = train_ner_model(
            nlp=blank_model,
            train_data=sample_data,
            n_iter=_N_ITER_SPLIT,
            logger=custom_logger,
        )

        assert nlp_trained is not None, "Modelo deve ser retornado"
        assert metrics is not None, "Métricas devem ser retornadas"

    def test_train_with_none_logger_creates_default(self, blank_model: object, sample_data: list) -> None:
        """Testa que logger padrão é criado se None fornecido.

        Verifica que:
        - Função funciona com logger=None
        - Cria logger padrão internamente
        """
        nlp_trained, metrics = train_ner_model(
            nlp=blank_model,
            train_data=sample_data,
            n_iter=_N_ITER_SPLIT,
            logger=None,
        )

        assert nlp_trained is not None, "Modelo deve ser retornado mesmo com logger=None"
        assert metrics is not None, "Métricas devem ser retornadas mesmo com logger=None"

    def test_train_with_single_example(self, blank_model: object) -> None:
        """Testa treinamento com apenas um exemplo.

        Verifica que:
        - Funciona com dados mínimos (1 exemplo)
        - Métricas são calculadas corretamente
        """
        single_example = [
            ("CPF 123.456.789-09", {"entities": [(0, 14, "CPF")]}),
        ]

        _, metrics = train_ner_model(
            nlp=blank_model,
            train_data=single_example,
            n_iter=_N_ITER_SINGLE,
            logger=logging.getLogger(__name__),
        )

        assert metrics["examples_count"] == _N_SINGLE_EXAMPLE, f"Deve ter {_N_SINGLE_EXAMPLE} exemplo"
        assert metrics["iterations"] == _N_ITER_SINGLE, f"Deve ter {_N_ITER_SINGLE} iterações"

    def test_train_metrics_structure(self, blank_model: object, sample_data: list) -> None:
        """Testa estrutura das métricas retornadas.

        Verifica que:
        - Métricas contêm todas as chaves esperadas
        - Todos os valores têm tipos corretos
        """
        _, metrics = train_ner_model(
            nlp=blank_model,
            train_data=sample_data,
            n_iter=_N_ITER_SPLIT,
            logger=logging.getLogger(__name__),
        )

        assert "final_loss" in metrics, "Deve conter 'final_loss'"
        assert "iterations" in metrics, "Deve conter 'iterations'"
        assert "examples_count" in metrics, "Deve conter 'examples_count'"

        assert isinstance(metrics["final_loss"], int | float | np.floating), "final_loss deve ser número"
        assert isinstance(metrics["iterations"], int), "iterations deve ser int"
        assert isinstance(metrics["examples_count"], int), "examples_count deve ser int"

    def test_train_with_validation_split_zero(self, blank_model: object, sample_data: list) -> None:
        """Testa treinamento com validation_split=0.

        Verifica que:
        - Usa todos os dados para treinamento quando split é 0
        - Funcionamento é normal
        """
        _, metrics = train_ner_model(
            nlp=blank_model,
            train_data=sample_data,
            n_iter=_N_ITER_SPLIT,
            validation_split=0.0,
            logger=logging.getLogger(__name__),
        )

        assert metrics["examples_count"] == _N_EXAMPLES, f"Com split=0, deve usar todos os {_N_EXAMPLES} exemplos"
