"""Fixtures compartilhadas para todos os testes."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

from anonimizar.sei_anonimizar import SeiAnonimizar

load_dotenv()


@pytest.fixture
def model_path():
    """Caminho do modelo spaCy."""
    return os.getenv("SPACY_MODEL_PATH", "X:/sei-anonimizar-bckp/nlp_treinado_v5")


@pytest.fixture
def anonymizer(model_path) -> SeiAnonimizar:
    """Fixture para SeiAnonimizar configurado."""
    anonymizer = SeiAnonimizar(model_path=model_path)
    anonymizer.add_apply_patterns(
        [
            "CPF",
            "RG",
            "CNH",
            "TITULO_ELEITOR",
            "PASSAPORTE",
            "SIAPE",
            "DATA_NASCIMENTO",
            "DADOS_BANCARIOS",
            "EMAIL",
            "TELEFONE",
            "ENDEREÇO",
            "CID",
        ]
    )
    return anonymizer


@pytest.fixture
def sample_texts():
    """Textos de exemplo para testes."""
    return [
        "João Silva, CPF 123.456.789-09, email: joao@email.com",
        "Maria Santos, RG 12.345.678-9, telefone: (11) 98765-4321",
        "Pedro Costa, SIAPE 1234567, CNH 12345678901",
        "Ana Oliveira mora na Rua das Flores, 123, CEP 01234-567",
        "Passaporte AB123456, Data de nascimento: 15/03/1985",
    ]


@pytest.fixture
def sample_training_data():
    """Dados de exemplo para treinamento."""
    return [
        {"text": "CPF 123.456.789-09", "entities": [(4, 18, "CPF")]},
        {"text": "Outro exemplo de texto com CPF 123.456.789-09.", "entities": [(31, 45, "CPF")]},
        {
            "text": "João Silva, CPF 123.456.789-09, mora na Rua das Flores, 123.",
            "entities": [(16, 30, "CPF"), (40, 59, "ENDEREÇO")],
        },
    ]


@pytest.fixture
def temp_dir():
    """Diretório temporário para testes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_evaluation_data():
    """Dados de exemplo para avaliação."""
    texts = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "text": ["CPF 123.456.789-09 é válido", "Email teste@email.com funciona", "Telefone (11) 99999-9999"],
        }
    )

    ground_truth = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "tp_entidade": ["CPF", "EMAIL", "TELEFONE"],
            "start_entidade": [4, 6, 9],
            "end_entidade": [18, 21, 25],
            "text_entidade": ["123.456.789-09", "teste@email.com", "(11) 99999-9999"],
        }
    )

    return texts, ground_truth


@pytest.fixture
def sample_markdown_tables():
    """Tabelas markdown de exemplo para testes."""
    return {
        "simple": """
| Nome | CPF | Email |
|------|-----|-------|
| João | 123.456.789-09 | joao@email.com |
""",
        "multiple_rows": """
| CPF | RG |
|-----|-----|
| 123.456.789-09 | 12.345.678-9 |
| 987.654.321-00 | 98.765.432-1 |
| 111.222.333-44 | 11.222.333-4 |
""",
        "mixed_content": """
Texto antes da tabela com CPF 555.666.777-88

| Nome | Documento |
|------|-----------|
| Maria | 999.888.777-66 |

Texto depois da tabela.
""",
        "malformed": """
| CPF |
| 123.456.789-09 |
""",
        "empty": """
| CPF | Email |
|-----|-------|
""",
    }
