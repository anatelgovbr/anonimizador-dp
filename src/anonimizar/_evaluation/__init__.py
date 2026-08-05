"""Utilitários internos para avaliação de modelos NER.

O pacote reúne carregamento de dados, extração de predições, comparação de
spans, métricas e exportação de relatórios. Os reexports abaixo formam a
superfície estável para consumidores internos; a fachada pública é
``Evaluation``.
"""

from anonimizar._evaluation.comparison import compare_reports, generate_comparison_data
from anonimizar._evaluation.data_loader import load_data, load_data_from_files, set_predictions
from anonimizar._evaluation.metrics import evaluate_multiple_thresholds, generate_evaluation_report
from anonimizar._evaluation.predictor import extract_predictions, load_predictions, save_predictions
from anonimizar._evaluation.reporter import export_dataframe, get_classification_cases, get_error_analysis

__all__ = [
    "compare_reports",
    "evaluate_multiple_thresholds",
    "export_dataframe",
    "extract_predictions",
    "generate_comparison_data",
    "generate_evaluation_report",
    "get_classification_cases",
    "get_error_analysis",
    "load_data",
    "load_data_from_files",
    "load_predictions",
    "save_predictions",
    "set_predictions",
]
