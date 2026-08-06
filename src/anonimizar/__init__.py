"""Ferramentas para detectar, anonimizar, treinar e avaliar dados pessoais.

APIs públicas:
    - ``Anonimizar``: extrai e anonimiza dados pessoais.
    - ``Trainer``: treina modelos spaCy NER.
    - ``Evaluation``: avalia predições de modelos NER.

Nota sobre dados:
    **Dados Fictícios:** todos os dados usados nos exemplos e no treinamento dos
    modelos NER distribuídos no projeto são fictícios. Eles preservam somente
    padrões estruturais existentes em documentos reais, sem reproduzir dados
    pessoais ou conteúdos reais. **Apenas parecem dados reais**, mas são
    totalmente fictícios e fora do contexto do conteúdo do texto de origem
    utilizado no treinamento dessa biblioteca.

Pipeline de Detecção:
    O pipeline combina três fontes de extração em etapas sequenciais:

        1. Modelo spaCy NER — identifica entidades via aprendizado de máquina
        2. Padrões REGEX — captura padrões conhecidos (CPF, RG, email, etc.)
        3. Tabelas Markdown — extrai colunas com cabeçalhos relacionados a dados pessoais

    Os resultados são unificados, conflitos de sobreposição são resolvidos por
    prioridade de label e cada entidade passa por um validador específico:

    - Validação algorítmica: confere dígitos verificadores (CPF, CNH,
      TITULO_ELEITOR, PIS, CNS)
    - Validação estrutural: verifica formato esperado (EMAIL, CID, GEO_COORD, RG)
    - Validação contextual: busca palavras-chave no entorno (SIAPE, TELEFONE,
      PASSAPORTE, etc.)

Entidades Suportadas:
    +------------------+---------------------------------------------------+--------------------------------------+
    | Rótulo           | Descrição                                         | Método de Detecção                   |
    +==================+===================================================+======================================+
    | CPF              | Cadastro de Pessoa Física                         | REGEX + NER + validação alg. (DV)    |
    +------------------+---------------------------------------------------+--------------------------------------+
    | RG               | Registro Geral (inclui RNE/CRNM)                  | REGEX + NER + validação estrutural   |
    +------------------+---------------------------------------------------+--------------------------------------+
    | CNH              | Carteira Nacional de Habilitação                  | REGEX + NER + validação alg. (DV)    |
    +------------------+---------------------------------------------------+--------------------------------------+
    | TITULO_ELEITOR   | Título de Eleitor                                 | REGEX + NER + validação alg. (DV+UF) |
    +------------------+---------------------------------------------------+--------------------------------------+
    | PASSAPORTE       | Passaporte                                        | REGEX + NER + validação contextual   |
    +------------------+---------------------------------------------------+--------------------------------------+
    | SIAPE            | Sistema Integrado de Adm. de Pessoal              | REGEX + NER + validação contextual   |
    +------------------+---------------------------------------------------+--------------------------------------+
    | DATA_NASCIMENTO  | Data de Nascimento                                | REGEX + NER + validação contextual   |
    +------------------+---------------------------------------------------+--------------------------------------+
    | DADOS_BANCARIOS  | Dados Bancários                                   | REGEX + NER + validação contextual   |
    +------------------+---------------------------------------------------+--------------------------------------+
    | EMAIL            | Endereço de E-mail                                | REGEX + NER + validação estrutural   |
    +------------------+---------------------------------------------------+--------------------------------------+
    | TELEFONE         | Número de Telefone                                | REGEX + NER + validação contextual   |
    +------------------+---------------------------------------------------+--------------------------------------+
    | ENDEREÇO         | Endereço Postal / CEP                             | REGEX + NER + validação contextual   |
    +------------------+---------------------------------------------------+--------------------------------------+
    | CID              | Classificação Internacional de Doenças            | REGEX + NER + validação estrutural   |
    +------------------+---------------------------------------------------+--------------------------------------+
    | GEO_COORD        | Coordenadas Geográficas                           | REGEX + NER + validação estrutural   |
    +------------------+---------------------------------------------------+--------------------------------------+
    | PIS              | PIS/PASEP/NIT                                     | REGEX + NER + validação alg. (DV)    |
    +------------------+---------------------------------------------------+--------------------------------------+
    | CNS              | Cartão Nacional de Saúde                          | REGEX + NER + validação alg. (DV)    |
    +------------------+---------------------------------------------------+--------------------------------------+
    | RESERVISTA       | Certificado de Reservista                         | REGEX + NER + validação contextual   |
    +------------------+---------------------------------------------------+--------------------------------------+

Tipos de Retorno:
    ``label_position``
        Para pré-processamento interativo — exiba as marcações para o usuário
        revisar antes da anonimização definitiva. Retorna ``[{"label": "CPF",
        "start_position": 10, "end_position": 24}]``.

    ``label_text``
        Para auditoria rápida — apenas os valores encontrados, sem
        coordenadas. Retorna ``[{"label": "CPF", "text": "123.456.789-09"}]``.

    ``label_detail``
        Para depuração — mostra qual método detectou cada entidade (modelo,
        regex ou tabela). Retorna ``[{"label": "CPF", "start_position": 10,
        "end_position": 24, "text": "123.456.789-09", "detected_by": "regex"}]``.

    ``anonymize_text()``
        Para processamento massivo — substitui automaticamente entidades por
        tags como ``<|CPF|>``, ``<|RG|>``. Sem intervenção humana.

Fluxo de treinamento, avaliação e uso:
    ```python
    from anonimizar import Anonimizar, Evaluation, Trainer

    trainer = Trainer(output_dir="./modelo_treinado", labels=["CPF", "EMAIL"])
    dados = [
        {"text": "CPF 123.456.789-09", "entities": [(4, 18, "CPF")]},
        {"text": "E-mail joao@empresa.com.br", "entities": [(7, 26, "EMAIL")]},
    ]
    trainer.add_data(dados, errors="coerce")
    trainer.train(n_iter=30, validation_split=0.2)
    trainer.save_model()

    anonymizer = Anonimizar("./modelo_treinado")
    evaluator = Evaluation(texts_path="texts.parquet", ground_truth_path="gt.parquet")
    evaluator.extract_predictions(anonymizer)
    resultados = evaluator.evaluate_model()
    ```

Arquitetura do Pacote:
    ::

        anonimizar/
        ├── __init__.py              → API pública
        ├── _anonymization/          → anonimização + CLI
        ├── _extraction/             → pipeline de detecção
        ├── _patterns/               → padrões regex built-in + custom
        ├── _validators/             → validadores contextuais + algorítmicos
        ├── _training/               → treinamento NER
        ├── _evaluation/             → avaliação de modelos
        ├── _normalization/          → normalização offset-safe
        ├── _constants/              → labels, prioridades, thresholds
        └── _common/                 → logging, overlap
"""

import warnings

from anonimizar._anonymization.anonymizer import Anonimizar
from anonimizar._evaluation.evaluation import Evaluation
from anonimizar._training.trainer_facade import Trainer

_DEPRECATED_ALIASES = {
    "SeiAnonimizar": Anonimizar,
    "SeiAnonimizarEvaluation": Evaluation,
    "SeiAnonimizarNERTrainer": Trainer,
}


def __getattr__(name: str) -> type:
    """Resolve alias deprecados e levanta AttributeError para nomes desconhecidos."""
    if name in _DEPRECATED_ALIASES:
        warnings.warn(
            f"anonimizar.{name} está depreciado; use o nome sem o prefixo Sei.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_ALIASES[name]
    raise AttributeError(name)


__all__ = [
    "Anonimizar",
    "Evaluation",
    "SeiAnonimizar",
    "SeiAnonimizarEvaluation",
    "SeiAnonimizarNERTrainer",
    "Trainer",
]

__version__ = "1.0.4"
