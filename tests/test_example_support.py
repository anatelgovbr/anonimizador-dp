"""Cobertura dos utilitários usados pelos exemplos de avaliação."""

import logging

import pandas as pd

from anonimizar._common.logging import create_default_logger
from anonimizar._evaluation.predictor import extract_predictions, load_predictions, save_predictions


class _ExampleAnonymizer:
    """Anonimizador mínimo para exercitar o exemplo de predições."""

    def extract_entities(self, text_or_path: str, return_type: str) -> list[dict]:
        """Retorna uma entidade determinística para o texto do exemplo."""
        assert return_type == "label_detail"
        if "CPF" not in text_or_path:
            return []
        text = "123.456.789-09"
        start = text_or_path.index(text)
        return [
            {
                "label": "CPF",
                "text": text,
                "start_position": start,
                "end_position": start + len(text),
                "detected_by": "regex",
            }
        ]


def test_prediction_example_flow_roundtrip(tmp_path) -> None:
    """Cobre extração e persistência CSV do fluxo documentado."""
    logger = logging.getLogger("test_example_support")
    texts = pd.DataFrame({"id": [1, 2], "text": ["CPF 123.456.789-09", "Sem entidade"]})
    predictions = extract_predictions(texts, _ExampleAnonymizer(), {"CPF": "CPF"}, logger)

    assert list(predictions["tp_entidade"]) == ["CPF"]
    output_path = tmp_path / "predictions.csv"
    save_predictions(predictions, output_path, logger)
    loaded = load_predictions(output_path, logger)

    assert loaded.to_dict("records") == predictions.to_dict("records")


def test_prediction_example_flow_empty_result_has_expected_contract() -> None:
    """Cobre o aviso e o retorno vazio do exemplo sem entidades."""
    logger = logging.getLogger("test_example_support_empty")
    texts = pd.DataFrame({"id": [1], "text": ["Sem entidade"]})

    predictions = extract_predictions(texts, _ExampleAnonymizer(), {}, logger)

    assert predictions.empty
    assert list(predictions.columns) == []


def test_default_logger_keeps_preconfigured_logger() -> None:
    """Cobre a configuração padrão e a preservação de handlers existentes."""
    name = "test_example_support_logger"
    logger = logging.getLogger(name)
    logger.handlers.clear()

    configured = create_default_logger(name)
    assert configured.handlers
    assert configured.level == logging.INFO

    configured_again = create_default_logger(name)
    assert configured_again is configured

    logger.handlers.clear()
