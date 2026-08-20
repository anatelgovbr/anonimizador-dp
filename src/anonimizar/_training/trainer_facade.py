"""Módulo trainer_facade.

Este módulo fornece funcionalidades robustas para o treinamento de modelos NER (Named Entity Recognition)
usando spaCy, com foco em anonimização de dados pessoais.

O módulo oferece validação avançada de dados, incluindo verificação de esquema BILUO, transformação
de dados de diferentes formatos, controle flexível de erros durante o processo de treinamento, e suporte
completo para importação/exportação de anotações no formato JSONL (Doccano).

Labels suportados por padrão:
- CPF, RG, SIAPE, ENDEREÇO, TELEFONE, EMAIL, GEO_COORD, CID
- DADOS_BANCARIOS, CNH, PASSAPORTE, TITULO_ELEITOR, DATA_NASCIMENTO

Principais funcionalidades:
- Suporte a modelos pré-treinados do spaCy ou criação de modelo em branco
- Validação rigorosa de dados com esquema BILUO
- Múltiplos formatos de entrada (dicionário, lista, DataFrame Pandas, JSONL)
- Importação/exportação de anotações no formato JSONL (compatível com Doccano)
- Estratégias flexíveis de tratamento de erros ('raise', 'coerce', 'ignore')
- Limpeza automática de entidades com resolução de conflitos
- Cross-validation estratificada com paralelização opcional
- Holdout test set para avaliação consistente entre folds
- Rastreabilidade completa de IDs de documentos em cada fold (JSON e CSV)
- Logging detalhado do processo de treinamento
- Divisão automática de dados para treino e validação
- Integração direta com Evaluation
- Suporte nativo para formato JSONL (Doccano) em add_data e cross_validate
- Métodos load_from_doccano_jsonl e save_to_doccano_jsonl
- Método _load_jsonl_to_dataframes para conversão JSONL → DataFrames
- Parâmetro holdout_test_size no cross_validate para separar conjunto de teste fixo
- Parâmetro holdout_stratify para estratificar o holdout test
- Salvamento de IDs de documentos por fold (fold_ids.json, train_ids.csv, val_ids.csv)
- Salvamento de métricas detalhadas do holdout test
- Validação adicional: start >= end e start < 0 em validate_data
- Curriculum learning por fases (train_curriculum): progressão de dificuldade
  (janelas w00/w0/w1/w2/full) e/ou de qualidade dos dados; fluxo end-to-end
  (jsonl ou dois DataFrames + fases por dataset/conjunto) ou com dados prontos
  por fase (joblib/tuplas); registro automático de labels novos

Exemplos:
    Exemplo de uso básico:

        >>> import pandas as pd
        >>> from anonimizar import Trainer
        >>>
        >>> # Inicializar treinador
        >>> trainer = Trainer(
        ...     model_name="pt_core_news_sm",
        ...     output_dir="./meu_modelo_ner",
        ...     labels=["CPF", "RG", "EMAIL"]
        ... )
        >>>
        >>> # Adicionar dados de treinamento
        >>> dados = [
        ...     {
        ...         "text": "João Silva, CPF 123.456.789-00, email: joao@email.com",
        ...         "entities": [(12, 26, "CPF"), (35, 50, "EMAIL")]
        ...     },
        ...     {
        ...         "text": "Maria Santos, RG 12.345.678-9",
        ...         "entities": [(14, 26, "RG")]
        ...     }
        ... ]
        >>> trainer.add_data(dados, errors='coerce')
        >>>
        >>> # Treinar modelo
        >>> trainer.train(n_iter=20, validation_split=0.2)
        >>>
        >>> # Salvar modelo treinado
        >>> trainer.save_model()


    Exemplo com resolução automática de conflitos:

        >>> # Dados com entidades problemáticas
        >>> dados_conflitos = [
        ...     {
        ...         "text": "João Silva com CPF duplicado",
        ...         "entities": [(15, 25, "CPF"), (15, 25, "CPF")]  # Duplicata
        ...     },
        ...     {
        ...         "text": "Maria com sobreposição de dados",
        ...         "entities": [(10, 20, "CPF"), (15, 25, "RG")]  # Sobreposição
        ...     }
        ... ]
        >>>
        >>> # Limpeza automática com resolução de conflitos
        >>> trainer.add_data(
        ...     dados_conflitos,
        ...     auto_clean=True,
        ...     strict_clean=False,
        ...     resolve_conflicts='coerce'  # Resolve automaticamente
        ... )


    Exemplo com DataFrame Pandas:

        >>> import pandas as pd
        >>>
        >>> df = pd.DataFrame({
        ...     'text': [
        ...         "Pedro Oliveira, SIAPE 1234567",
        ...         "Ana Costa, telefone (11) 98765-4321"
        ...     ],
        ...     'entities': [
        ...         [(17, 24, "SIAPE")],
        ...         [(20, 36, "TELEFONE")]
        ...     ]
        ... })
        >>>
        >>> trainer.add_data(df, errors='ignore', keep_empty_entities=False)


    Exemplo com tratamento de erros:

        >>> # Dados com problemas intencionais
        >>> dados_problematicos = [
        ...     {
        ...         "text": "Teste com label inválido",
        ...         "entities": [(0, 5, "LABEL_INEXISTENTE")]
        ...     }
        ... ]
        >>>
        >>> # Diferentes estratégias de erro
        >>> trainer.add_data(dados_problematicos, errors='ignore')  # Ignora erros
        >>> trainer.add_data(dados_problematicos, errors='coerce')  # Corrige/remove erros
        >>> trainer.add_data(dados_problematicos, errors='raise')   # Lança exceção


    Exemplo com JSONL/Doccano:

        >>> # Importar dados de arquivo JSONL (Doccano)
        >>> trainer = Trainer(labels=['CPF', 'EMAIL', 'TELEFONE'])
        >>> trainer.add_data("./anotacoes_doccano.jsonl", errors='coerce')
        >>> print(f"Dados carregados: {len(trainer.training_data)} exemplos")
        >>>
        >>> # Exportar dados de treinamento para JSONL
        >>> trainer.save_to_doccano_jsonl("./dados_exportados.jsonl")
        >>>
        >>> # Converter JSONL em DataFrames para análise
        >>> df_textos, df_entidades = trainer._load_jsonl_to_dataframes("./anotacoes.jsonl")
        >>> print(f"Textos: {len(df_textos)}, Entidades: {len(df_entidades)}")
        >>> # df_entidades tem colunas: id, start, end, entidade


    Exemplo com Cross-Validation básico:

        >>> # Exemplo de uso completo
        >>> trainer = Trainer(
        ...     model_name=None,  # modelo em branco
        ...     output_dir="./base_model",
        ...     labels=['CPF', 'RG', 'EMAIL', 'TELEFONE', 'SIAPE']
        ... )
        >>>
        >>> # Parâmetros de treinamento e avaliação
        >>> train_params = {
        ...     'n_iter': 30,
        ...     'drop': 0.2,
        ...     'batch_size': 8
        ... }
        >>>
        >>> eval_params = {
        ...     'overlap_threshold': 0.8,
        ...     'beta': 2.0
        ... }
        >>>
        >>> # Executa cross validation
        >>> reports, summaries, results, holdout = trainer.cross_validate(
        ...     df_entidades=df_entidades,
        ...     df_textos=df_textos,
        ...     n_splits=5,
        ...     stratified=True,
        ...     features=features,
        ...     output_dir="./cv_results",
        ...     n_jobs=3,
        ...     train_params={'n_iter': 30, 'drop': 0.2, 'batch_size': 8},
        ...     eval_params={'overlap_threshold': 0.8, 'beta': 2.0},
        ...     add_data_params={
        ...         'errors': 'coerce',
        ...         'auto_clean': True,
        ...         'strict_clean': True,
        ...         'keep_empty_entities': False
        ... },
        ... )
        >>>
        >>> # Análise dos resultados
        >>> print("Métricas por fold:")
        >>> for summary in summaries:
        >>>     if summary:
        >>>         print(f"Fold {summary['fold']}: F1={summary['fbeta']:.3f}")


    Exemplo com cross-validation estratificada avançada:

        >>> # Cross-validation com balanceamento por tipo de entidade
        >>> reports, summaries, results, holdout = trainer.cross_validate(
        ...     df_entidades=df_entidades,
        ...     df_textos=df_textos,
        ...     n_splits=5,
        ...     stratified=True,
        ...     features=['CPF', 'EMAIL', 'TELEFONE'],  # Estratificar por estes tipos
        ...     output_dir="./cv_estratificado",
        ...     n_jobs=2,  # Paralelização
        ...     random_state=42,
        ...     add_data_params={
        ...         'auto_clean': True,
        ...         'resolve_conflicts': 'coerce',
        ...         'strict_clean': False
        ...     },
        ...     replace=True  # Substitui diretório existente
        ... )
        >>>
        >>> # Análise detalhada dos resultados
        >>> for fold_idx, summary in enumerate(summaries, 1):
        ...     if summary:
        ...         print(f"Fold {fold_idx}:")
        ...         print(f"  Precision: {summary['precision']:.3f}")
        ...         print(f"  Recall: {summary['recall']:.3f}")
        ...         print(f"  F-beta: {summary['fbeta']:.3f}")


    Exemplo com Cross-Validation + Holdout Test:

        >>> # CV com holdout test set separado (20% dos dados)
        >>> reports, summaries, results, holdout_results = trainer.cross_validate(
        ...     df_entidades=df_entidades,
        ...     df_textos=df_textos,
        ...     n_splits=5,
        ...     output_dir="./cv_with_holdout",
        ...     holdout_test_size=0.2,  # Separar 20% para test fixo
        ...     holdout_stratify=True,   # Estratificar o holdout também
        ...     train_params={'n_iter': 25},
        ...     replace=True
        ... )
        >>>
        >>> # Analisar métricas no holdout test (comparável entre folds)
        >>> if holdout_results is not None:
        >>>     print("\n=== Métricas no Holdout Test ===")
        >>>     for result in holdout_results:
        >>>         print(f"Fold {result['fold']}: F-beta={result['fbeta']:.3f}")


    Exemplo com CV usando JSONL:

        >>> # Cross-validation direto de arquivo JSONL
        >>> reports, summaries, _, _ = trainer.cross_validate(
        ...     df_entidades="./anotacoes_completas.jsonl",  # Arquivo JSONL
        ...     df_textos=None,  # Mesmo arquivo contém textos e entidades
        ...     n_splits=5,
        ...     output_dir="./cv_from_jsonl",
        ...     train_params={'n_iter': 15},
        ...     replace=True
        ... )


    Exemplo de rastreabilidade de folds:

        >>> # Após executar CV, verificar IDs usados em cada fold
        >>> import json
        >>> from pathlib import Path
        >>>
        >>> cv_dir = Path("./cv_results")
        >>> for fold_dir in sorted(cv_dir.glob("fold_*")):
        >>>     if fold_dir.is_dir():
        >>>         # Carregar metadados do fold
        >>>         with open(fold_dir / "fold_ids.json", 'r') as f:
        >>>             fold_info = json.load(f)
        >>>
        >>>         print(f"\n{fold_dir.name}:")
        >>>         print(f"  Train IDs: {len(fold_info['train_ids'])} docs")
        >>>         print(f"  Val IDs: {len(fold_info['val_ids'])} docs")
        >>>
        >>>         # Também disponível em CSV
        >>>         train_csv = pd.read_csv(fold_dir / "train_ids.csv")
        >>>         val_csv = pd.read_csv(fold_dir / "val_ids.csv")


Estrutura de arquivos gerados pelo Cross-Validation:
    cv_results/
    ├── fold_1/
    │   ├── model/                    # Modelo treinado no fold 1
    │   ├── fold_ids.json             # Metadados (train_ids, val_ids, counts)
    │   ├── train_ids.csv             # IDs de treino (com coluna 'split')
    │   ├── val_ids.csv               # IDs de validação
    │   ├── detailed_report.parquet   # Relatório detalhado do fold
    │   └── summary.json              # Métricas agregadas do fold
    ├── fold_2/
    │   └── ...
    ├── all_folds_detailed.parquet    # Todos os folds consolidados
    ├── fold_summaries.parquet        # Sumário de todos os folds
    ├── holdout_test_ids.json         # IDs separados para holdout test (se usado)
    ├── holdout_test_ids.csv          # IDs do holdout em CSV
    ├── holdout_test_summary.parquet  # Métricas agregadas no holdout
    └── holdout_test_stats.json       # Estatísticas gerais do holdout


Métodos principais:
    - _create_default_logger: Cria logger padrão quando não fornecido.
    - _add_labels: Adiciona labels ao pipeline NER do spaCy.
    - add_data: Adiciona dados de treinamento (dict, lista, DataFrame, ou JSONL) com validação.
    - _validate_data: Valida dados de entrada com verificação de labels e posições.
    - _validate_biluo_tags: Valida entidades usando esquema BILUO do spaCy.
    - _transform_data_from_pandas: Transforma DataFrame Pandas para formato spaCy.
    - train: Executa o treinamento iterativo do modelo NER.
    - train_curriculum: Executa treinamento por curriculum learning em fases
      sequenciais (fluxo end-to-end com janelas ou dados prontos por fase).
    - save_model: Salva o modelo treinado em disco.
    - split_data: Divide dados em conjuntos de treino e validação.
    - val_data_to_evaluation: Transforma dados de validação para formato do Evaluation
    - validate_entities: Valida entidades anotadas retornando True/False
    - clean_entities: Limpa e corrige entidades automaticamente incluindo resolução de conflitos
    - detect_entity_conflicts: Detecta conflitos entre entidades (duplicatas e sobreposições)
    - debug_entities: Debug detalhado de problemas em offsets/labels
    - make_folds_by_id: Cria folds baseados em IDs de documento para cross-validation
    - make_stratified_folds_by_id: Cria folds estratificados balanceando distribuição de entidades
    - cross_validate: Executa cross validation completo com paralelização opcional
    - load_from_doccano_jsonl: Carrega dados de arquivo JSONL (formato Doccano) [NOVO v0.0.9]
    - save_to_doccano_jsonl: Salva training_data em arquivo JSONL (formato Doccano) [NOVO v0.0.9]
    - _load_jsonl_to_dataframes: Converte JSONL em DataFrames (textos e entidades) [NOVO v0.0.9]
"""  # noqa: D301

import logging
import time  # noqa: F401
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import spacy

from anonimizar._common.logging import create_default_logger
from anonimizar._constants import (
    ALL_ENTITIES_KEY,  # noqa: F401 — kept for backward compat with existing tests
    DEFAULT_BATCH_SIZE,
    DEFAULT_BETA,  # noqa: F401 — used indirectly via cv_manager
    DEFAULT_CURRICULUM_WINDOWS,
    DEFAULT_DROP,
    DEFAULT_INITIAL_BASE_FRAC,
    DEFAULT_N_ITER,
    DEFAULT_N_JOBS,
    DEFAULT_N_SPLITS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OVERLAP_THRESHOLD,  # noqa: F401 — used indirectly via cv_manager
    DEFAULT_RANDOM_STATE,
    DEFAULT_SUPPORTED_LABELS,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VALIDATION_SPLIT,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
    SPACY_LANGUAGE,
)
from anonimizar._normalization import normalize_entity
from anonimizar._training import cross_validation as cv
from anonimizar._training import data_loader, data_validator
from anonimizar._training import io as training_io
from anonimizar._training.curriculum_data import build_curriculum_datasets as _build_curriculum_datasets
from anonimizar._training.cv_manager import NERCrossValidator as _NERCrossValidator
from anonimizar._training.data_manager import NERDataManager as _NERDataManager
from anonimizar._training.io_handler import load_from_doccano_jsonl as _load_jsonl
from anonimizar._training.io_handler import save_to_doccano_jsonl as _save_jsonl
from anonimizar._training.trainer import train_ner_model as _train_ner_model
from anonimizar._training.trainer import train_ner_model_curriculum as _train_ner_model_curriculum

warnings.filterwarnings("ignore", message=r"\[W030\] Some entities could not be aligned.*")

_TAM_EXEMPLO_TUPLA = 2


class Trainer:
    """Classe para treinamento de modelos NER especializados em anonimização de dados pessoais.

    Esta classe oferece funcionalidades completas para treinar modelos de reconhecimento
    de entidades nomeadas (NER) usando spaCy, com foco em identificação de dados pessoais
    para anonimização.

    Attributes:
        model_name (str | None): Nome do modelo base do spaCy ou None para modelo em branco
        output_dir (Path): Diretório onde o modelo treinado será salvo
        logger (logging.Logger): Logger para registro de atividades
        nlp (spacy.Language): Pipeline spaCy configurado
        ner (spacy.pipeline.EntityRecognizer): Componente NER do pipeline
        training_data (list): Dados de treinamento no formato spaCy
        supported_labels (list[str]): Lista de labels suportados pelo modelo.
        train_data (list): Dados de treinamento criados por ``split_data()``.
        val_data (list): Dados de validação criados por ``split_data()``.

    Example:
        >>> trainer = Trainer(
        ...     model_name="pt_core_news_sm",
        ...     output_dir="./modelo_personalizado",
        ...     labels=["CPF", "RG", "EMAIL"]
        ... )
        >>> dados = {"text": "João, CPF 123.456.789-00", "entities": [(6, 20, "CPF")]}
        >>> trainer.add_data(dados)
        >>> trainer.train(n_iter=10)
        >>> trainer.save_model()
    """

    def __init__(
        self,
        model_name: str | None = None,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        logger: logging.Logger | None = None,
        labels: list[str] | None = None,
        *,
        normalize_entities: bool = True,
    ) -> None:
        """Inicializa o treinador NER com configurações personalizadas.

        Args:
            model_name (str | None, optional): Nome do modelo base do spaCy (ex: "pt_core_news_sm"),
                ou caminho de um modelo ja treinado.
                Se None, cria um modelo em branco. Defaults to None.
            output_dir (str, optional): Diretório para salvar o modelo treinado.
                Defaults to "./trained_model".
            logger (logging.Logger | None, optional): Logger personalizado para registro de atividades.
                Se None, cria um logger padrão. Defaults to None.
            labels (list[str] | None, optional): Lista de labels personalizados para o modelo.
                Se None, usa labels padrão para documentos do SEI. Defaults to None.
            normalize_entities (bool): Se True (padrão), normaliza entidades dos dados de
                treinamento removendo prefixos/sufixos textuais (ex: "RG: 1234567 SSP/DF"
                vira "1234567"), ajustando offsets automaticamente.

        Raises:
            OSError: Se o modelo especificado em model_name não for encontrado

        Example:
            >>> # Modelo com configuração padrão
            >>> trainer = Trainer()
            >>>
            >>> # Modelo com configuração personalizada
            >>> trainer = Trainer(
            ...     model_name="pt_core_news_lg",
            ...     output_dir="./meu_modelo",
            ...     labels=["CPF", "RG", "EMAIL", "TELEFONE"]
            ... )
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.logger = logger or create_default_logger(__name__)
        msg = f"Trainer inicializado com model_name={self.model_name}, output_dir={self.output_dir}"
        self.logger.info(msg)
        self.normalize_entities = normalize_entities
        self._blank_nlp = spacy.blank(SPACY_LANGUAGE)

        if not model_name:
            self.logger.warning("\tInicializando pipeline spaCy em branco para 'pt'. Nenhum modelo base fornecido.")
            self.nlp = spacy.blank(SPACY_LANGUAGE)
        else:
            msg = f"\tCarregando modelo spaCy '{model_name}' para treinamento NER"
            self.logger.debug(msg)
            self.nlp = spacy.load(model_name)

        if "ner" not in self.nlp.pipe_names:
            self.nlp.add_pipe("ner")
        self.other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        self.ner = self.nlp.get_pipe("ner")

        self.training_data = []
        if labels:
            self.supported_labels = labels
        else:
            msg = "Nao foram definidos labels, serão usados os labels padrão"
            self.logger.warning(msg)
            self.supported_labels = list(DEFAULT_SUPPORTED_LABELS)
        self._add_labels()
        msg = f"Labels suportados: {self.supported_labels}"
        self.logger.debug(msg)

        # Inicializar gerenciador de dados centralizado
        self._data_manager = _NERDataManager(
            supported_labels=self.supported_labels,
            logger=self.logger,
        )

        # Inicializar gerenciador de cross-validation
        self._cv_manager = _NERCrossValidator(logger=self.logger)

    def validate_entities(self, text: str, entities: list[tuple[int, int, str]]) -> bool:
        """Valida entidades anotadas com base no texto.

        Retorna True se todas estiverem corretas,
        ou False se houver qualquer erro (offset inválido, label não suportado, desalinhamento BILUO etc.).

        Args:
            text (str): Texto completo a ser inspecionado.
            entities (list[tuple[int, int, str]]): Lista de tuplas (start, end, label).

        Returns:
            bool: True se todas as entidades forem válidas, False caso contrário.

        Examples:
            Validar entidades corretas:

                >>> trainer = Trainer(labels=["CPF"])
                >>> text = "CPF: 123.456.789-00"
                >>> entities = [(5, 19, "CPF")]  # 14 caracteres: 123.456.789-00
                >>> trainer.validate_entities(text, entities)
                True

            Validar entidades com problemas:

                >>> # Offset com espaços extras
                >>> text = "CPF: 123.456.789-00  "
                >>> entities = [(5, 21, "CPF")]  # Inclui espaços extras
                >>> trainer.validate_entities(text, entities)
                False

                >>> # Offset desalinhado aos tokens
                >>> text = "EMAIL: teste@exemplo.com"
                >>> entities = [(14, 21, "EMAIL")]  # Não alinhado ao token
                >>> trainer.validate_entities(text, entities)
                False
        """
        return data_validator.validate_entities(text, entities, self.supported_labels, self.nlp, self.logger)

    def _add_labels(self) -> None:
        """Adiciona labels ao pipeline NER do spaCy.

        Este método registra todos os labels suportados no componente NER do pipeline spaCy,
        permitindo que o modelo reconheça essas entidades durante o treinamento.

        Note:
            - Método interno chamado automaticamente durante a inicialização
            - Cada label é registrado individualmente no pipeline NER
            - Labels duplicados são automaticamente ignorados pelo spaCy
            - Logs informativos são gerados para cada label adicionado

        Example:
            >>> # Chamado automaticamente durante inicialização
            >>> trainer = Trainer(labels=["CPF", "RG"])
            >>> # Logs: "Adicionado label CPF ao modelo", "Adicionado label RG ao modelo"
        """
        for label in self.supported_labels:
            self.ner.add_label(label)
            msg = f"Adicionado label {label} ao modelo NER"
            self.logger.debug(msg)

    def train(
        self,
        n_iter: int = DEFAULT_N_ITER,
        drop: float = DEFAULT_DROP,
        batch_size: int = DEFAULT_BATCH_SIZE,
        validation_split: float = DEFAULT_VALIDATION_SPLIT,
        initial_base_frac: float = DEFAULT_INITIAL_BASE_FRAC,
    ) -> dict[str, Any] | None:
        """Treina o modelo NER usando os dados previamente carregados.

        Args:
            n_iter (int, optional): Número de iterações (épocas) de treinamento.
                Padrão = 20.
            drop (float, optional): Taxa de *dropout* aplicada durante a atualização
                dos pesos. Padrão = 0.35.
            batch_size (int, optional): Tamanho da minilote utilizada nas
                atualizações. Padrão = 8.
            validation_split (float, optional): Fração (0-1) dos dados reservada
                para validação. Se 0, todo o conjunto é usado para treinamento.
                Padrão = 0.2.
            initial_base_frac (float, optional): Fração inicial do conjunto de
                treinamento usada para compor o lote de exemplos passado a
                ``nlp.initialize()``. Padrão = 1.0.

        Raises:
            ValueError: Se ``self.training_data`` estiver vazio.
            Exception: Repassa qualquer exceção gerada durante o processo de
                treinamento.

        Returns:
            dict[str, Any]: Dicionário com métricas de treinamento incluindo
                'final_loss', 'iterations' e 'examples_count'.
        """
        try:
            if not self.training_data:
                msg = "Nenhum dado de treinamento disponível"
                self.logger.exception(msg)
                raise ValueError(msg)  # noqa: TRY301
            if not 0.0 <= validation_split < 1.0:
                msg = f"validation_split deve estar entre 0.0 e 1.0(não incluso), recebido: {validation_split}"
                self.logger.exception(msg)
                raise ValueError(msg)  # noqa: TRY301

            # Preparar dados para treinamento
            if validation_split > 0:
                self.split_data(train_ratio=1 - validation_split)
                train_data = self.train_data
            else:
                train_data = self.training_data

            self.logger.info("Delegando treinamento para módulo trainer especializado")

            # Delegar treinamento para função standalone
            self.nlp, metrics = _train_ner_model(
                nlp=self.nlp,
                train_data=train_data,
                n_iter=n_iter,
                drop=drop,
                batch_size=batch_size,
                validation_split=0.0,  # Já foi dividido acima
                initial_base_frac=initial_base_frac,
                logger=self.logger,
                other_pipes=self.other_pipes,
            )

            self.logger.info("Treinamento concluído com sucesso via módulo trainer")

        except Exception as e:
            msg = f"Erro durante treinamento: {e!s}"
            self.logger.exception(msg)
            raise e  # noqa: TRY201
        else:
            return metrics

    def train_curriculum(
        self,
        phases: list[dict],
        *,
        df_textos: pd.DataFrame | str | None = None,
        df_entidades: pd.DataFrame | str | None = None,
        windows: tuple[str, ...] | None = None,
        include_pure: bool = False,
        oversample: dict[str, int] | None = None,
        drop: float = DEFAULT_DROP,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_compounding: tuple[float, float, float] | None = None,
        initial_base_frac: float = DEFAULT_INITIAL_BASE_FRAC,
        errors: str = "raise",
        auto_clean: bool = True,
        strict_clean: bool = True,
        resolve_conflicts: str = "coerce",
        keep_empty_entities: bool = False,
    ) -> dict[str, Any] | None:
        """Treina o modelo NER por curriculum learning com fases sequenciais.

        Cada fase treina um subconjunto de dados por um número de épocas, na
        ordem informada — reproduzindo o padrão de curriculum learning dos
        experimentos da estória 942 (junções de janelas de dificuldade e
        conjuntos de qualidade). Há dois fluxos de entrada, misturáveis entre
        fases, ambos usando a chave ``dataset``:

        1. **End-to-end**: informe ``df_textos`` + ``df_entidades`` (ou um
           caminho ``.jsonl`` em ``df_textos``) e referencie cada fase pelo
           nome da janela em ``dataset`` (e ``conjunto`` opcional). As janelas
           são geradas internamente e uma única vez.
        2. **Separado**: informe dados prontos em ``dataset`` (dicts, listas,
           DataFrames, JSONL ou tuplas ``(texto, {"entities": ...})`` do
           formato dos joblibs da 942).

        Um valor string de ``dataset`` que não termina em ``.jsonl`` é
        interpretado como nome de janela (fluxo e2e); qualquer outro valor
        (lista, dict, DataFrame, tuplas ou caminho ``.jsonl``) é tratado como
        dado pronto (fluxo separado).

        Labels novos encontrados nas fases (ex.: ``PIS``, ``CNS``,
        ``RESERVISTA``) são registrados automaticamente no modelo e em
        ``supported_labels``.

        Args:
            phases: Lista de fases. Cada fase é um dicionário com:
                - ``dataset`` (obrigatório): dados da fase — nome da janela a
                  usar no fluxo e2e (``w0``, ``w1``, ``w2``, ``full`` ou
                  ``w00`` quando gerada) ou dados prontos do fluxo separado
                  (formatos aceitos por ``add_data`` ou tuplas).
                - ``conjunto`` (opcional, fluxo e2e): conjunto referenciado;
                  padrão ``"default"``.
                - ``epochs`` (int, obrigatório): épocas da fase (>= 1).
                - ``name`` (str, opcional): nome da fase em logs e métricas.
            df_textos: DataFrame com ``id``/``text`` ou caminho ``.jsonl``
                (Doccano) com textos e entidades.
            df_entidades: DataFrame com ``id``/``start``/``end``/``entidade``
                (ou nomes da 942). Obrigatório quando ``df_textos`` é
                DataFrame; ignore quando ``df_textos`` é ``.jsonl``.
            windows: Janelas a gerar no fluxo e2e (padrão
                ``("w0", "w1", "w2", "full")``); janelas referenciadas nas
                fases são acrescentadas automaticamente.
            include_pure: Se True no fluxo e2e, gera também a janela ``"w00"``
                (entidade pura isolada).
            oversample: Oversampling do ``w00`` (label → fator, ex.:
                ``{"ENDEREÇO": 3}``).
            drop: Taxa de dropout. Padrão = 0.35.
            batch_size: Tamanho do minilote. Padrão = 8.
            batch_compounding: Tupla ``(start, stop, rate)`` para minilotes
                crescentes (ex.: ``(8.0, 32.0, 1.001)`` da 942).
            initial_base_frac: Fração da 1ª fase usada em ``nlp.initialize()``.
            errors: Política de erro do preparo dos dados ('raise', 'coerce',
                'ignore').
            auto_clean: Se True, aplica limpeza automática nas entidades.
            strict_clean: Se True, descarta exemplos com entidades removidas.
            resolve_conflicts: Como resolver conflitos de entidades.
            keep_empty_entities: Se True, mantém textos sem entidades válidas.

        Raises:
            ValueError: Se ``phases`` for vazio, fase sem ``data``/``dataset``,
                ``epochs`` inválido, ``dataset`` sem fonte de dados, janela/
                conjunto inexistente ou fase vazia após o preparo.

        Returns:
            dict[str, Any]: Métricas com 'final_loss', 'iterations',
                'examples_count', 'total_epochs' e 'phases' (detalhe por fase).

        Example:
            Fluxo end-to-end (jsonl ou dois DataFrames + fases por janela)::

                >>> trainer = Trainer()
                >>> trainer.train_curriculum(
                ...     df_textos="./anotacoes.jsonl",
                ...     phases=[
                ...         {"dataset": "w0", "epochs": 2},
                ...         {"dataset": "w1", "epochs": 1},
                ...         {"conjunto": "default", "dataset": "full", "epochs": 8},
                ...     ],
                ... )

            Fluxo separado (joblib preparado fora)::

                >>> datasets = load_curriculum_datasets("./datasets_treino_2.joblib")
                >>> trainer.train_curriculum(
                ...     phases=[
                ...         {"dataset": datasets["sujo"]["w0"], "epochs": 2},
                ...         {"dataset": datasets["ouro"]["full"], "epochs": 8,
                ...          "name": "ouro/full"},
                ...     ],
                ... )
        """
        if not phases:
            msg = "Nenhuma fase de curriculum informada"
            self.logger.exception(msg)
            raise ValueError(msg)

        precisa_e2e = any(
            isinstance(fase.get("dataset"), str) and not fase["dataset"].endswith(".jsonl") for fase in phases
        )
        datasets = None
        if precisa_e2e:
            if df_textos is None:
                msg = (
                    "Fases referenciam janelas de 'dataset', mas df_textos/df_entidades (ou jsonl) não foram informados"
                )
                self.logger.exception(msg)
                raise ValueError(msg)
            df_t, df_e = self._carregar_fontes_curriculum(df_textos, df_entidades)
            janelas_pedidas = list(windows) if windows else list(DEFAULT_CURRICULUM_WINDOWS)
            for fase in phases:
                janela = fase.get("dataset")
                if isinstance(janela, str) and not janela.endswith(".jsonl") and janela not in janelas_pedidas:
                    janelas_pedidas.append(janela)
            datasets = _build_curriculum_datasets(
                df_t,
                df_e,
                windows=tuple(janelas_pedidas),
                include_pure=include_pure,
                oversample=oversample,
                logger=self.logger,
            )

        dados_resolvidos = [
            (fase, self._resolver_dados_fase_curriculum(idx, fase, datasets))
            for idx, fase in enumerate(phases, start=1)
        ]
        self._registrar_labels_curriculum(dados_resolvidos)
        fases_treino = self._preparar_fases_treino_curriculum(
            dados_resolvidos,
            errors=errors,
            keep_empty_entities=keep_empty_entities,
            auto_clean=auto_clean,
            strict_clean=strict_clean,
            resolve_conflicts=resolve_conflicts,
        )

        try:
            self.logger.info("Delegando curriculum para módulo trainer especializado")
            self.nlp, metrics = _train_ner_model_curriculum(
                nlp=self.nlp,
                phases=fases_treino,
                drop=drop,
                batch_size=batch_size,
                batch_compounding=batch_compounding,
                initial_base_frac=initial_base_frac,
                logger=self.logger,
                other_pipes=self.other_pipes,
            )
            self.logger.info("Curriculum concluído com sucesso via módulo trainer")
        except Exception as e:
            msg = f"Erro durante treinamento por curriculum: {e!s}"
            self.logger.exception(msg)
            raise e  # noqa: TRY201
        else:
            return metrics

    def _carregar_fontes_curriculum(
        self,
        df_textos: pd.DataFrame | str | None,
        df_entidades: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Valida e carrega as fontes de dados do fluxo e2e do curriculum."""
        if isinstance(df_textos, str):
            if df_entidades is not None:
                msg = "Quando df_textos é um caminho .jsonl, df_entidades deve ser None"
                self.logger.exception(msg)
                raise ValueError(msg)
            return self._load_jsonl_to_dataframes(df_textos)
        if isinstance(df_textos, pd.DataFrame) and df_entidades is None:
            msg = "df_entidades é obrigatório quando df_textos é um DataFrame"
            self.logger.exception(msg)
            raise ValueError(msg)
        return df_textos, df_entidades

    def _resolver_dados_fase_curriculum(
        self,
        idx: int,
        fase: dict,
        datasets: dict[str, dict[str, list]] | None,
    ) -> object:
        """Resolve os dados de uma fase do curriculum: dado pronto ou nome de janela e2e."""
        epochs = fase.get("epochs")
        if not isinstance(epochs, int) or epochs < 1:
            msg = f"Fase {idx} deve ter epochs inteiro >= 1, recebido: {epochs}"
            self.logger.exception(msg)
            raise ValueError(msg)
        if "data" in fase:
            msg = f"Fase {idx}: a chave 'data' foi unificada em 'dataset'"
            self.logger.exception(msg)
            raise ValueError(msg)
        tem_dataset = "dataset" in fase
        if tem_dataset:
            valor = fase["dataset"]
            if isinstance(valor, str) and not valor.endswith(".jsonl"):
                if datasets is None:
                    msg = (
                        f"Fase {idx} referencia a janela {valor!r}, mas df_textos/df_entidades "
                        "(ou jsonl) não foram informados"
                    )
                    self.logger.exception(msg)
                    raise ValueError(msg)
                conjunto = str(fase.get("conjunto", "default"))
                janela = valor
                if conjunto not in datasets:
                    conjuntos = sorted(datasets)
                    msg = f"Fase {idx}: conjunto {conjunto!r} inexistente. Disponíveis: {conjuntos}"
                    self.logger.exception(msg)
                    raise ValueError(msg)
                if janela not in datasets[conjunto]:
                    janelas = sorted(datasets[conjunto])
                    msg = f"Fase {idx}: dataset {janela!r} inexistente no conjunto {conjunto!r}. Disponíveis: {janelas}"
                    self.logger.exception(msg)
                    raise ValueError(msg)
                dados = datasets[conjunto][janela]
            else:
                dados = valor
        else:
            msg = f"Fase {idx} deve informar 'dataset'"
            self.logger.exception(msg)
            raise ValueError(msg)
        if not dados:
            msg = f"Fase {idx} sem dados"
            self.logger.exception(msg)
            raise ValueError(msg)
        return dados

    def _registrar_labels_curriculum(self, dados_resolvidos: list[tuple[dict, object]]) -> None:
        """Registra labels novos encontrados nas fases antes do preparo (estilo labels_do_curriculum da 942)."""
        labels_descob: set[str] = set()
        for _, dados in dados_resolvidos:
            for _, annotations in self._convert_to_tuples(dados, errors="coerce"):
                for _, _, label in annotations.get("entities", []):
                    labels_descob.add(str(label))
        for label in sorted(labels_descob - set(self.supported_labels)):
            self.ner.add_label(label)
            self.supported_labels.append(label)
            self.logger.info("Label novo do curriculum registrado: %s", label)

    def _preparar_fases_treino_curriculum(
        self,
        dados_resolvidos: list[tuple[dict, object]],
        *,
        errors: str,
        keep_empty_entities: bool,
        auto_clean: bool,
        strict_clean: bool,
        resolve_conflicts: str,
    ) -> list[dict]:
        """Prepara os dados de cada fase (normalizar + auto_clean + validar BILUO)."""
        fases_treino: list[dict] = []
        for idx, (fase, dados) in enumerate(dados_resolvidos, start=1):
            preparados = self._prepare_data(
                dados,
                errors=errors,
                keep_empty_entities=keep_empty_entities,
                auto_clean=auto_clean,
                strict_clean=strict_clean,
                resolve_conflicts=resolve_conflicts,
            )
            if not preparados:
                msg = f"Fase {idx} ficou sem exemplos válidos após o preparo"
                self.logger.exception(msg)
                raise ValueError(msg)
            fases_treino.append(
                {"name": fase.get("name") or f"Fase {idx}", "train_data": preparados, "epochs": int(fase["epochs"])}
            )
        return fases_treino

    def _format_time(self, seconds: float) -> str:
        """Formata tempo em segundos para formato legível.

        Args:
            seconds (float): Tempo em segundos

        Returns:
            str: Tempo formatado (ex: "1h 23m 45s", "2m 30s", "15s")

        Example:
            >>> self._format_time(3661.5)
            '1h 1m 1s'
            >>> self._format_time(90.2)
            '1m 30s'
            >>> self._format_time(15.8)
            '16s'
        """
        if seconds < SECONDS_PER_MINUTE:
            return f"{int(seconds)}s"
        if seconds < SECONDS_PER_HOUR:
            minutes = int(seconds // SECONDS_PER_MINUTE)
            secs = int(seconds % SECONDS_PER_MINUTE)
            return f"{minutes}m {secs}s"
        hours = int(seconds // SECONDS_PER_HOUR)
        minutes = int((seconds % SECONDS_PER_HOUR) // SECONDS_PER_MINUTE)
        secs = int(seconds % SECONDS_PER_MINUTE)
        return f"{hours}h {minutes}m {secs}s"

    def add_data(
        self,
        data: list | dict | pd.DataFrame | str | None = None,
        *,
        errors: str = "raise",
        keep_empty_entities: bool = False,
        auto_clean: bool = True,
        strict_clean: bool = True,
        resolve_conflicts: str = "coerce",
        normalize_entities: bool | None = None,
    ) -> None:
        """Adiciona dados de treinamento à classe com validação e limpeza opcional.

        Args:
            data: Dados a serem adicionados, caminho para arquivo .jsonl (Doccano).
            errors: Política de tratamento de erro: "raise", "coerce" ou "ignore".
                - "raise": Lança exceção quando encontra entidades inválidas.
                - "coerce": Remove entidades inválidas silenciosamente.
                - "ignore": Mantém entidades como estão, sem correção.
            keep_empty_entities: Se True, mantém textos mesmo quando todas as entidades
                são descartadas na validação.
            auto_clean: Se True, aplica limpeza automática nas entidades:
                - Remove espaços extras nos limites das entidades.
                - Valida alinhamento BILUO (offsets devem corresponder a tokens spaCy).
                - Corrige offsets quando possível.
            strict_clean: Se True, descarta exemplos onde alguma entidade foi removida
                durante a limpeza. Se False, mantém exemplos mesmo com menos entidades.
            resolve_conflicts: Como resolver conflitos de entidades:
                - "raise": Lança erro quando há conflitos (duplicatas ou sobreposições).
                - "ignore": Remove documentos com conflitos.
                - "coerce": Tenta resolver conflitos automaticamente.
            normalize_entities (bool | None): Se True, normaliza entidades removendo
                prefixos/sufixos (ex: "RG: 1234567 SSP/DF" vira "1234567").
                Se None, usa o valor de ``self.normalize_entities``. Defaults to None.

        Examples:
            Adicionar dados com comportamento padrão (limpeza automática):

                >>> trainer.add_data([{"text": "CPF: 123.456.789-00", "entities": [(5, 20, "CPF")]}])

            Ignorar erros (mantém dados inválidos sem correção):

                >>> trainer.add_data(bad_data, errors="ignore", auto_clean=False)
                >>> # Entidades inválidas são mantidas como estão

            Ignorar erros com limpeza (corrige espaços mas não levanta erro):

                >>> trainer.add_data(bad_data, errors="ignore", auto_clean=True)
                >>> # Espaços são corrigidos, mas BILUO inválido é mantido

            Corrigir erros automaticamente:

                >>> trainer.add_data(bad_data, errors="coerce", auto_clean=True)
                >>> # Espaços são corrigidos, entidades com BILUO inválido são removidas

            Levantar erro em dados inválidos:

                >>> trainer.add_data(bad_data, errors="raise", auto_clean=True)
                >>> # Levanta ValueError se houver problemas não corrigíveis

            Sem limpeza automática:

                >>> trainer.add_data(bad_data, errors="raise", auto_clean=False)
                >>> # Valida entidades sem aplicar correções automáticas

            Resolver conflitos de entidades:

                >>> # Dados com sobreposição
                >>> data = [{"text": "CPF 123.456.789-00", "entities": [(4, 16, "CPF"), (4, 20, "CPF")]}]
                >>> trainer.add_data(data, resolve_conflicts="coerce")  # Remove duplicata
        """
        if data is None:
            self.logger.warning("Nenhum dado fornecido para adicionar.")
            return

        validated_data = self._prepare_data(
            data,
            errors=errors,
            keep_empty_entities=keep_empty_entities,
            auto_clean=auto_clean,
            strict_clean=strict_clean,
            resolve_conflicts=resolve_conflicts,
            normalize_entities=normalize_entities,
        )

        self.training_data.extend(validated_data)
        # Sincronizar com o gerenciador de dados centralizado
        self._data_manager.training_data = list(self.training_data)
        msg = (
            f"Dados adicionados com sucesso. Exemplos válidos adicionados: {len(validated_data)}. "
            f"Total de exemplos no conjunto de treinamento: {len(self.training_data)}"
        )
        self.logger.debug(msg)

    def _convert_to_tuples(self, data: list | dict | pd.DataFrame | str | tuple, errors: str = "raise") -> list:
        """Converte dados de entrada em tuplas ``(texto, {"entities": [...]})`` sem validar.

        Além dos formatos aceitos por ``data_loader.convert_input_data``
        (dict, lista de dicts, DataFrame, caminho JSONL), aceita tuplas
        ``(texto, {"entities": [...]})`` — formato usado nos joblibs dos
        experimentos da estória 942 — isolando-as antes da conversão dos
        demais itens.

        Args:
            data: Dados em qualquer formato aceito.
            errors: Política de erro para a conversão ('raise', 'coerce',
                'ignore').

        Returns:
            list: Lista de tuplas (texto, {"entities": [...]}).
        """

        def _eh_tupla_exemplo(item: object) -> bool:
            return (
                isinstance(item, tuple)
                and len(item) == _TAM_EXEMPLO_TUPLA
                and isinstance(item[0], str)
                and isinstance(item[1], dict)
            )

        if isinstance(data, tuple) and _eh_tupla_exemplo(data):
            return [data]
        if isinstance(data, list):
            tuplas = []
            restantes = []
            for item in data:
                if _eh_tupla_exemplo(item):
                    tuplas.append(item)
                else:
                    restantes.append(item)
            if restantes:
                tuplas.extend(
                    data_loader.convert_input_data(
                        data=restantes,
                        logger=self.logger,
                        transform_pandas_fn=lambda df: data_loader.transform_data_from_pandas(df, self.logger),
                        load_jsonl_fn=training_io.load_doccano_jsonl,
                        errors=errors,
                    )
                )
            return tuplas
        return data_loader.convert_input_data(
            data=data,
            logger=self.logger,
            transform_pandas_fn=lambda df: data_loader.transform_data_from_pandas(df, self.logger),
            load_jsonl_fn=training_io.load_doccano_jsonl,
            errors=errors,
        )

    def _prepare_data(
        self,
        data: list | dict | pd.DataFrame | str | tuple | None,
        *,
        errors: str = "raise",
        keep_empty_entities: bool = False,
        auto_clean: bool = True,
        strict_clean: bool = True,
        resolve_conflicts: str = "coerce",
        normalize_entities: bool | None = None,
    ) -> list:
        """Converte, normaliza, limpa e valida dados para o formato de treinamento.

        Pipeline compartilhado entre ``add_data`` e ``train_curriculum``:
        conversão para tuplas → normalização de entidades → limpeza automática
        (opcional) → validação (labels, offsets e BILUO).

        Args:
            data: Dados em qualquer formato aceito, incluindo tuplas.
            errors: Política de erro ('raise', 'coerce', 'ignore').
            keep_empty_entities: Se True, mantém textos mesmo sem entidades.
            auto_clean: Se True, aplica limpeza automática nas entidades.
            strict_clean: Se True, descarta exemplos com entidades removidas.
            resolve_conflicts: Como resolver conflitos ('raise', 'ignore',
                'coerce').
            normalize_entities: Se True, normaliza entidades removendo
                prefixos/sufixos. Se None, usa ``self.normalize_entities``.

        Returns:
            list: Dados validados como ``(texto, {"entities": [...]})``.

        Raises:
            ValueError: Quando ``errors="raise"`` e entidades são perdidas
                durante a limpeza.
        """
        new_data = self._convert_to_tuples(data, errors=errors)

        # Normalizar entidades (remover prefixos/sufixos)
        do_normalize = self.normalize_entities if normalize_entities is None else normalize_entities
        if do_normalize:
            normalized_count = 0
            for i, (text, annotations) in enumerate(new_data):
                entities = annotations.get("entities", [])
                new_entities = []
                for start, end, label in entities:
                    entity_text = text[start:end]
                    _cleaned, new_start, new_end, _rule = normalize_entity(entity_text, start, end, label)
                    if new_start != start or new_end != end:
                        normalized_count += 1
                    new_entities.append((new_start, new_end, label))
                new_data[i] = (text, {"entities": new_entities})
            if normalized_count:
                self.logger.debug("Normalizadas %d entidades nos dados de treinamento", normalized_count)

        # Aplicar limpeza automática se habilitada
        if auto_clean:
            # Contar entidades antes da limpeza para detectar perdas
            original_entities_count = sum(len(annotations.get("entities", [])) for _, annotations in new_data)

            new_data = data_loader.apply_auto_clean(
                data=new_data,
                clean_entities_fn=self.clean_entities,
                strict_clean=strict_clean,
                keep_empty_entities=keep_empty_entities,
                resolve_conflicts=resolve_conflicts,
                logger=self.logger,
                errors=errors,
            )

            # Verificar se houve perda de entidades durante a limpeza
            # Apenas levanta erro se errors='raise' E houve perda de entidades
            if errors == "raise" and original_entities_count > 0:
                final_entities_count = sum(len(annotations.get("entities", [])) for _, annotations in new_data)
                if final_entities_count < original_entities_count:
                    msg = (
                        f"Entidades foram removidas durante a limpeza automática: "
                        f"{original_entities_count} -> {final_entities_count}. "
                        "Use auto_clean=False ou errors='coerce' para permitir "
                        "remoção silenciosa de entidades inválidas."
                    )
                    self.logger.exception(msg)
                    raise ValueError(msg)

            # Validar após limpeza - pula BILUO apenas se errors != "raise"
            # Quando errors="raise", queremos validar BILUO para detectar problemas remanescentes
            # Quando errors="ignore", também pulamos BILUO pois já manteve entidades como estão
            skip_biluo = errors in ("coerce", "ignore")
            validated_data = self._validate_data(
                new_data, errors=errors, keep_empty_entities=keep_empty_entities, skip_biluo=skip_biluo
            )
        else:
            # Validar sem limpeza
            validated_data = self._validate_data(new_data, errors=errors, keep_empty_entities=keep_empty_entities)

        return validated_data

    def _validate_data(
        self,
        data_to_validate: list,
        *,
        errors: str = "raise",
        keep_empty_entities: bool = False,
        skip_biluo: bool = False,
    ) -> list:
        """Valida os dados de treinamento fornecidos.

        Args:
            data_to_validate: Lista de tuplas (text, annotations) para validar.
            errors: 'raise' para lançar erro, 'coerce' para corrigir, 'ignore' para ignorar.
            keep_empty_entities: Se True, mantém textos mesmo quando todas as entidades são inválidas.
                            Se False (padrão), descarta textos sem entidades válidas.
            skip_biluo (bool) : se true nao faz validacao de biluo


        Returns:
            list: Lista de dados validados no formato [(text, {"entities": [...]}), ...]

        Raises:
            ValueError: Quando ``errors="raise"`` e é encontrada
                inconsistência.
        """
        return data_validator.validate_data(
            data_to_validate,
            self.supported_labels,
            self._blank_nlp,
            self.logger,
            errors=errors,
            keep_empty_entities=keep_empty_entities,
            skip_biluo=skip_biluo,
        )

    def _validate_biluo_tags(self, text: str, entities: list, errors: str = "raise") -> list:
        """Valida entidades usando o esquema BILUO do spaCy.

        Args:
            text: Texto a ser validado
            entities: Lista de tuplas (start, end, label)
            errors:
                • `"raise"`  - lança exceção na primeira entidade inválida.
                • `"coerce"` - remove somente a entidade problemática.
                • `"ignore"` - mantém todas, mesmo que inconsistentes.

        Returns:
            list: Lista de entidades válidas

        Raises:
            ValueError: Se ``errors="raise"`` e algum problema BILUO é
                encontrado.
        """
        return data_validator.validate_biluo_tags(text, entities, self._blank_nlp, self.logger, errors)

    def _transform_data_from_pandas(self, df: pd.DataFrame) -> list:
        """Converte DataFrame para o formato spaCy.

        Formatos aceitos:

        1. Colunas ``text`` e ``entities``
           - ``entities`` deve ser lista de tuplas.

        2. Formato "linha por entidade"
           - colunas ``texto``, ``start``, ``end`` e ``entidade``.

        Args:
            df: DataFrame a converter.

        Returns:
            list: Lista [(text, {"entities": [...]})].

        Raises:
            ValueError: Se o DataFrame não seguir nenhum dos formatos.
        """
        return data_loader.transform_data_from_pandas(df, self.logger)

    def save_model(self, path: str | None = None) -> None:
        """Persiste o modelo treinado em disco.

        Args:
            path (str | None, optional): Diretório de saída.
                Se ``None``, usa ``self.output_dir``.
        """
        save_path = Path(path) if path else self.output_dir
        save_path.mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(save_path)
        msg = f"Modelo treinado salvo em: {save_path.resolve()}"
        self.logger.info(msg)

    def split_data(self, train_ratio: float = DEFAULT_TRAIN_RATIO) -> None:
        """Divide ``training_data`` em treino e validação.

        Args:
            train_ratio (float, optional): Proporção destinada a treino.
                Default = ``0.8``.

        Raises:
            ValueError: Se houver menos de 2 exemplos.
        """
        self.train_data, self.val_data = data_loader.split_data(self.training_data, train_ratio, self.logger)
        # Sincronizar com o gerenciador de dados centralizado
        self._data_manager.train_data = list(self.train_data)
        self._data_manager.val_data = list(self.val_data)

    def debug_entities(self, text: str, entities: list[tuple[int, int, str]]) -> None:
        """Mostra detalhes de offsets/labels e verifica consistência BILUO.

        A função imprime no *logger*:

        • posição inicial/final de cada entidade
        • texto capturado pelo *offset*
        • possíveis problemas:
            - offset fora dos limites do texto
            - espaços à esquerda/direita
            - label não suportado
            - inconsistência BILUO (offset desalinhado aos tokens spaCy)

        Args:
            text (str): Texto completo a ser inspecionado.
            entities (list[tuple[int, int, str]]): Lista ``(start, end, label)``.
        """
        data_validator.debug_entities(text, entities, self.supported_labels, self.nlp, self.logger)

    def val_data_to_evaluation(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Converte para o formato do Evaluation.

        Converte o conjunto de validação interno (`self.val_data`) para o
        formato requerido pelo ``Evaluation``.

        Estrutura de saída
        A função devolve **duas** tabelas `pandas.DataFrame`:

        1. **df_texts**
        • colunas: ``id``, ``text``
        • cada linha contém o texto bruto que será avaliado.

        2. **df_ground_truth**
        • colunas: ``id``, ``tp_entidade``, ``start_entidade``, ``end_entidade``
        • cada linha representa **uma** entidade de referência
            (ground-truth) pertencente ao texto de mesmo ``id``.

        O par (df_texts, df_ground_truth) pode ser passado diretamente a
        ``Evaluation.load_data()`` sem necessidade de ajustes
        adicionais.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]
            - **df_texts**        - DataFrame com os textos a validar
            - **df_ground_truth** - DataFrame com as entidades de verdade-terreno

        Raises:
            ValueError
            Se ``self.val_data`` ainda não tiver sido gerado (ou seja, o
            método `split_data` não foi chamado ou não há dados
            suficientes para divisão).

        Example:
        >>> trainer.split_data(train_ratio=0.8)           # cria self.val_data
        >>> df_texts, df_gt = trainer.val_data_to_evaluation()
        >>> evaluator = Evaluation()
        >>> evaluator.load_data(df_texts, df_gt)
        >>> evaluator.extract_predictions(anonymizer)
        >>> evaluator.evaluate_model()
        """
        if not getattr(self, "val_data", None):
            msg = "Não existem dados de validação. Execute split_data() antes de chamar val_data_to_evaluation()."
            self.logger.exception(msg)
            raise ValueError(msg)

        return data_loader.val_data_to_evaluation(self.val_data, self.logger)

    def clean_entities(
        self,
        text: str,
        entities: list[tuple[int, int, str]],
        *,
        strict: bool = True,
        resolve_conflicts: str = "coerce",
        errors: str = "coerce",
    ) -> list[tuple[int, int, str]]:
        """Limpa e corrige entidades automaticamente, incluindo resolução de conflitos.

        Args:
            text (str): Texto completo onde as entidades estão localizadas.
            entities (list[tuple[int, int, str]]): Lista de entidades no formato (start, end, label).
            strict (bool): Se True e alguma entidade for removida durante a limpeza,
                retorna uma lista vazia para o documento.
            resolve_conflicts (str): Como lidar com conflitos:
                - "raise": Lança erro quando há conflitos.
                - "ignore": Retorna lista vazia (remove documento).
                - "coerce": Tenta resolver conflitos automaticamente.
            errors (str): Política de erros para validação BILUO:
                - "raise": Lança erro quando há problemas BILUO (offsets desalinhados).
                - "coerce": Remove entidades problemáticas silenciosamente.
                - "ignore": Mantém entidades como estão (sem correção).

        Returns:
            list[tuple[int, int, str]]: Lista de entidades limpas e válidas. Retorna
                uma lista vazia quando ``strict=True`` e alguma entidade é removida.

        Raises:
            ValueError: Se resolve_conflicts="raise" e conflitos forem encontrados,
                ou se errors="raise" e problemas BILUO forem detectados.

        Examples:
            Limpar entidades com correção automática (padrão):

                >>> text = "CPF: 123.456.789-09  "
                >>> entities = [(5, 21, "CPF")]  # Com espaços extras
                >>> trainer.clean_entities(text, entities)
                [(5, 19, "CPF")]  # Espaços removidos

            Manter entidades inválidas (sem correção):

                >>> text = "EMAIL: teste@exemplo.com"
                >>> entities = [(14, 21, "EMAIL")]  # BILUO desalinhado
                >>> trainer.clean_entities(text, entities, errors="ignore")
                [(14, 21, "EMAIL")]  # Mantido como está

            Descartar o documento no modo estrito:

                >>> text = "CPF: 123.456.789-09  "
                >>> entities = [(5, 21, "CPF")]
                >>> trainer.clean_entities(text, entities, strict=True)
                []
        """
        return data_validator.clean_entities(
            text,
            entities,
            self.supported_labels,
            self.nlp,
            self.logger,
            strict=strict,
            resolve_conflicts=resolve_conflicts,
            errors=errors,
        )

    def detect_entity_conflicts(self, entities: list[tuple[int, int, str]]) -> dict:
        """Detecta conflitos entre entidades (duplicatas e sobreposições).

        Args:
            entities: Lista de entidades (start, end, label)

        Returns:
            dict: Informações sobre conflitos encontrados
                - 'duplicates': lista de índices de entidades duplicadas
                - 'overlaps': lista de pares de índices com sobreposição
                - 'has_conflicts': bool indicando se há conflitos
        """
        return data_validator.detect_entity_conflicts(entities)

    def make_folds_by_id(
        self,
        df_entidades: pd.DataFrame,
        n_splits: int = DEFAULT_N_SPLITS,
        *,
        random_state: int = DEFAULT_RANDOM_STATE,
        shuffle: bool = True,
    ) -> list:
        """Cria folds baseados em IDs de documento para evitar vazamento de dados.

        Args:
            df_entidades: DataFrame com coluna 'id' (ou 'id_doc') identificando documentos
            n_splits: Número de folds para cross validation
            random_state: Seed para reprodutibilidade
            shuffle: Se deve embaralhar os dados antes da divisão

        Returns:
            list: Lista de tuplas (ids_train, ids_val) para cada fold
        """
        return cv.make_folds_by_id(
            df_entidades=df_entidades, n_splits=n_splits, random_state=random_state, shuffle=shuffle, logger=self.logger
        )

    def make_stratified_folds_by_id(
        self,
        df_entidades: pd.DataFrame,
        features: list | None = None,
        n_splits: int = DEFAULT_N_SPLITS,
        *,
        random_state: int = DEFAULT_RANDOM_STATE,
        shuffle: bool = True,
    ) -> list:
        """Cria folds estratificados para balancear a distribuição de entidades entre os folds.

        Args:
            df_entidades: DataFrame com entidades anotadas
            features: Lista de tipos de entidades para estratificação
            n_splits: Número de folds
            random_state: Seed para reprodutibilidade
            shuffle: Se deve embaralhar os dados

        Returns:
            list: Lista de tuplas (ids_train, ids_val) para cada fold
        """
        return cv.make_stratified_folds_by_id(
            df_entidades=df_entidades,
            features=features,
            n_splits=n_splits,
            random_state=random_state,
            shuffle=shuffle,
            logger=self.logger,
        )

    def cross_validate(
        self,
        df_entidades: pd.DataFrame | str,
        df_textos: pd.DataFrame | str | None = None,
        n_splits: int = DEFAULT_N_SPLITS,
        features: list | None = None,
        output_dir: str | None = None,
        n_jobs: int = DEFAULT_N_JOBS,
        random_state: int = DEFAULT_RANDOM_STATE,
        train_params: dict | None = None,
        eval_params: dict | None = None,
        add_data_params: dict | None = None,
        *,
        stratified: bool = True,
        replace: bool = False,
        holdout_test_size: float | None = None,
        holdout_stratify: bool = True,
    ) -> tuple[pd.DataFrame, list, list, Any]:
        """Executa cross validation completo do modelo NER.

        Args:
            df_entidades (pd.DataFrame | str): DataFrame com as colunas ``id``,
                ``start``, ``end`` e ``entidade``, ou caminho para arquivo JSONL.
            df_textos (pd.DataFrame | str | None): DataFrame com as colunas ``id``
                e ``text``, ou caminho para arquivo JSONL. Se None e
                ``df_entidades`` for str, assume o mesmo arquivo JSONL.
            n_splits: Número de folds
            stratified: Se deve usar estratificação por tipo de entidade
            features: Lista de tipos de entidades para estratificação
            output_dir: Diretório para salvar resultados
            n_jobs: Número de processos paralelos (1 para sequencial)
            random_state: Seed para reprodutibilidade
            train_params: Parâmetros para treinamento
            eval_params: Parâmetros para avaliação
            add_data_params: Parâmetros para o método add_data (ex: {'errors': 'coerce', 'auto_clean': True})
            replace: Se True, substitui diretório existente
            holdout_test_size: Fração estritamente entre 0 e 0.5 dos dados para separar como test fixo.
                          Ex: 0.2 = 20% para test, 80% para CV.
                          Se None, não separa holdout test.
            holdout_stratify: Se True, estratifica o holdout test por tipo de entidade.

        Returns:
            tuple[pd.DataFrame, list, list, Any]:
                ``all_reports`` consolida os relatórios detalhados dos folds;
                ``summary_metrics`` contém os resumos por fold; ``fold_results``
                contém os resultados retornados por cada execução; e
                ``holdout_results`` contém métricas do holdout ou ``None`` quando
                nenhum holdout é solicitado.

        Raises:
            ValueError: Se os DataFrames/JSONL ou parâmetros de divisão forem inválidos.
            FileExistsError: Se o diretório de saída já existir e ``replace=False``.
        """

        def _create_fold_instance(
            model_name: str | None,
            output_dir: str,
            labels: list[str] | None,
            logger: logging.Logger,
        ) -> "Trainer":
            """Cria instância de Trainer para um fold."""
            return Trainer(
                model_name=model_name,
                output_dir=output_dir,
                labels=labels,
                logger=logger,
            )

        return self._cv_manager.run(
            df_entidades=df_entidades,
            df_textos=df_textos,
            trainer_factory=_create_fold_instance,
            n_splits=n_splits,
            features=features or self.supported_labels,
            output_dir=output_dir,
            n_jobs=n_jobs,
            random_state=random_state,
            train_params=train_params,
            eval_params=eval_params,
            add_data_params=add_data_params,
            supported_labels=self.supported_labels,
            model_name=self.model_name,
            output_path_base=self.output_dir,
            stratified=stratified,
            replace=replace,
            holdout_test_size=holdout_test_size,
            holdout_stratify=holdout_stratify,
        )

    def load_from_doccano_jsonl(self, jsonl_path: str) -> list:
        """Carrega dados de arquivo JSONL exportado do Doccano.

        Suporta dois formatos do Doccano:
        - {"text": "...", "labels": [[start, end, "LABEL"], ...]}
        - {"text": "...", "entities": [{"start_offset": x, "end_offset": y, "label": "L"}]}

        Args:
            jsonl_path: Caminho para arquivo .jsonl

        Returns:
            list: Dados no formato [(text, {"entities": [(start, end, label), ...]}), ...]
        """
        return _load_jsonl(jsonl_path)

    def save_to_doccano_jsonl(self, output_path: str, data: list | None = None, format_type: str = "labels") -> None:
        """Salva dados em formato JSONL compatível com Doccano.

        Args:
            output_path (str): Caminho para salvar o arquivo .jsonl.
            data (list | None): Lista de tuplas no formato ``(text, {"entities":
                [(start, end, label), ...]})``. Se None, usa ``self.training_data``.
            format_type (str): Tipo de formato, ``"labels"`` ou ``"entities"``:
                - 'labels': {"text": "...", "labels": [[start, end, "LABEL"]]}
                - 'entities': {"text": "...", "entities": [{"start_offset": x, ...}]}

        Raises:
            ValueError: Se não houver dados para exportar ou ``format_type`` for inválido.
        """
        if data is None:
            if not self.training_data:
                msg = "Nenhum dado disponível para exportar"
                self.logger.exception(msg)
                raise ValueError(msg)
            data = self.training_data

        _save_jsonl(output_path, data, format_type=format_type)

    def _load_jsonl_to_dataframes(self, jsonl_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Converte arquivo JSONL do Doccano para formato de DataFrames usado no cross_validate.

        Formato JSONL esperado:
            {"text": "...", "labels": [[start, end, "LABEL"], ...]}
            ou
            {"text": "...", "entities": [{"start_offset": x, "end_offset": y, "label": "L"}]}

        Args:
            jsonl_path: Caminho para arquivo .jsonl

        Returns:
            tuple: (df_textos, df_entidades)
                - df_textos: DataFrame com colunas [id, text]
                - df_entidades: DataFrame com colunas [id, start, end, entidade]
        """
        return training_io.load_jsonl_to_dataframes(jsonl_path)
