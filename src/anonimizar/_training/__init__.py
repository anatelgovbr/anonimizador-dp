"""Recursos internos para treinamento de modelos NER.

Reexporta o gerenciador de dados, o executor de cross-validation, utilitários
de I/O JSONL compatíveis com Doccano, a preparação de dados em janelas de
dificuldade (curriculum learning) e as funções de treinamento spaCy.
"""

from anonimizar._training.curriculum_data import (
    build_curriculum_datasets,
    load_curriculum_datasets,
    save_curriculum_datasets,
)
from anonimizar._training.cv_manager import NERCrossValidator
from anonimizar._training.data_manager import NERDataManager
from anonimizar._training.io_handler import load_from_doccano_jsonl, load_jsonl_to_dataframes, save_to_doccano_jsonl
from anonimizar._training.trainer import train_ner_model, train_ner_model_curriculum

__all__ = [
    "NERCrossValidator",
    "NERDataManager",
    "build_curriculum_datasets",
    "load_curriculum_datasets",
    "load_from_doccano_jsonl",
    "load_jsonl_to_dataframes",
    "save_curriculum_datasets",
    "save_to_doccano_jsonl",
    "train_ner_model",
    "train_ner_model_curriculum",
]
