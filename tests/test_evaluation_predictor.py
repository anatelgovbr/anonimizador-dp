"""Testes de contratos do módulo ``_evaluation.predictor``."""

import logging
from unittest.mock import MagicMock

import pandas as pd

from anonimizar._evaluation.predictor import extract_predictions


def test_extract_predictions_empty_result_has_no_columns() -> None:
    """Nenhuma entidade extraída retorna DataFrame vazio sem schema."""
    anonymizer = MagicMock()
    anonymizer.extract_entities.return_value = []
    texts = pd.DataFrame({"id": [1], "text": ["Sem entidades"]})

    result = extract_predictions(texts, anonymizer, {}, logging.getLogger(__name__))

    assert result.empty
    assert list(result.columns) == []
