r"""Módulo anonymizer.

Este módulo fornece funcionalidades para detecção e anonimização de dados pessoais
em textos, utilizando modelos spaCy treinados e padrões regex personalizáveis. É especialmente
útil para anonimizar documentos que contenham informações pessoais como CPF, RG, e-mails,
telefones, dados bancários, endereços, entre outros.

O módulo combina duas abordagens:
1. Detecção por modelo de Machine Learning (spaCy NER)
2. Detecção por padrões regex customizáveis

**Classes:**
    Anonimizar: Classe principal para anonimização de dados pessoais.

**Exemplo básico de uso:**

    >>> from anonimizar import Anonimizar
    >>> # Inicializar o anonimizador
    >>> anonymizer = Anonimizar(model_path="pt_core_news_lg")
    >>>
    >>> # auto_patterns=True aplica os padrões built-in por padrão.
    >>>
    >>> # Processar texto
    >>> texto = "Meu CPF é 123.456.789-09 e meu e-mail é exemplo@dominio.com"
    >>> entidades = anonymizer.extract_entities(text_or_path=texto, return_type="label_detail")
    >>> texto_anonimizado = anonymizer.anonymize_text(texto, entidades)
    >>>
    >>> print(f"Original: {texto}")
    >>> print(f"Anonimizado: {texto_anonimizado}")
    "Anonimizado: Meu CPF é <|CPF|> e meu e-mail é <|EMAIL|>"

**Exemplos avançados de uso:**

    **Exemplo 1: Controle de labels na inicialização**

        >>> # Inicializar com labels específicos e padrões manuais
        >>> from anonimizar import Anonimizar
        >>> anonymizer = Anonimizar("pt_core_news_lg", labels=['RG'], auto_patterns=False)
        >>> text = "Olá meu nome é matheus e meu cpf é 075.834.649-27"
        >>> anonymizer.extract_entities(text, return_type='label_detail')
        []  # Retorna vazio pois não tem nenhum caso de RG
        >>> # Adicionar padrões regex para CPF em configuração manual
        >>> anonymizer.add_apply_patterns(['CPF'])
        >>> anonymizer.extract_entities(text, return_type="label_detail")
        [{'label': 'CPF',
        'start_position': 35,
        'end_position': 49,
        'text': '075.834.649-27',
        'detected_by': re.compile(r'\\d{3}[.]\\d{3}[.]\\d{3}[-.\\/]\\d{2}', re.UNICODE)}]

    **Exemplo 2: Detecção por modelo sem labels específicos**

        >>> # Inicializar sem labels específicos - todos os labels serão ativados
        >>> anonymizer = Anonimizar("pt_core_news_lg")
        >>> text = "Olá meu nome é matheus e meu cpf é 075.834.649-27"
        >>> anonymizer.extract_entities(text, return_type='label_detail')
        [{'label': 'CPF',
        'start_position': 35,
        'end_position': 49,
        'text': '075.834.649-27',
        'detected_by': 'modelo'}]


    **Exemplo 3: Usando use_model_labels em configuração manual**

        >>> # Ativar todos os labels do modelo
        >>> anonymizer = Anonimizar("pt_core_news_lg", auto_patterns=False)
        >>> text = "Olá meu nome é matheus e meu cpf é 075.834.649-27"
        >>> anonymizer.add_apply_patterns(use_model_labels=True)
        >>> anonymizer.extract_entities(text, return_type='label_detail')
        [{'label': 'CPF',
        'start_position': 35,
        'end_position': 49,
        'text': '075.834.649-27',
        'detected_by': 'modelo'}]


    **Exemplo 4: Combinando labels específicos com use_model_labels**

        >>> # Inicializar com label específico e usar apenas esse label
        >>> anonymizer = Anonimizar("pt_core_news_lg", labels=['RG'], auto_patterns=False)
        >>> text = "Olá meu nome é matheus e meu cpf é 075.834.649-27"
        >>> anonymizer.add_apply_patterns(use_model_labels=True)
        >>> anonymizer.extract_entities(text, return_type='label_detail')
        []  # Retorna vazio pois só RG está ativo e não há RG no texto


    **Exemplo 5: Substituição de padrões com replace_patterns**

        >>> # Adicionar padrões e depois substituir completamente
        >>> anonymizer = Anonimizar("pt_core_news_lg", auto_patterns=False)
        >>> anonymizer.add_apply_patterns(['CPF'])
        >>> len(anonymizer.patterns)  # Vários padrões para CPF
        11
        >>> # Adicionar RG sem substituir
        >>> anonymizer.add_apply_patterns(['RG'])
        >>> len(anonymizer.patterns)  # CPF + RG
        15
        >>> # Substituir todos os padrões por apenas EMAIL
        >>> anonymizer.add_apply_patterns(['EMAIL'], replace_patterns=True)
        >>> len(anonymizer.patterns)  # Apenas EMAIL
        1
        >>> anonymizer.patterns
        [{'label': 'EMAIL',
        'pattern': {'REGEX': '[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+'}}]


    **Exemplo 6: Padrões customizados**

        >>> # Adicionar padrão customizado
        >>> custom_patterns = [
        ...     {"label": "CODIGO_PRODUTO", "regex": r"PROD-\\d{4}",
        ...      "description": "Códigos de produto"}
        ... ]
        >>> anonymizer.add_apply_patterns(['CPF'], custom_patterns=custom_patterns)
        >>> text = "Pedido PROD-1234 com CPF 123.456.789-00"
        >>> entidades = anonymizer.extract_entities(text, return_type="label_detail")
        >>> texto_anonimizado = anonymizer.anonymize_text(text, entidades)
        >>> print(texto_anonimizado)
        "Pedido <|CODIGO_PRODUTO|> com CPF <|CPF|>"


    **Exemplo 7: Verificação de configuração ativa**

        >>> # Verificar quais labels e padrões estão ativos
        >>> anonymizer.get_active_labels()
        {
            'model_labels': ['CPF', 'RG', 'EMAIL'],
            'regex_summary': {'CPF': 11, 'EMAIL': 1, 'CODIGO_PRODUTO': 1},
            'regex_patterns': 13,
            'total_patterns': 16
        }

    **Exemplo 8: Validação de CPF com modelo**

        >>> from anonimizar import Anonimizar
        >>> anonymizer = Anonimizar("pt_core_news_lg", labels=['CPF'], use_cpf_validator=True)
        >>> text = "Ola meu nome é matheus e meu cpf é 000.000.000-00"
        >>> anonymizer.extract_entities(text, return_type='label_detail')
        []  # Retorna vazio pois 000.000.000-00 não é válido
        >>> # Desabilitar validação de CPF
        >>> anonymizer.use_cpf_validator = False
        >>> anonymizer.extract_entities(text, return_type='label_detail')
        [{'label': 'CPF',
        'start_position': 35,
        'end_position': 49,
        'text': '000.000.000-00',
        'detected_by': 'modelo'}]


    **Exemplo 9: Validação de CPF com regex**

        >>> anonymizer = Anonimizar("pt_core_news_lg", labels=['RG'], use_cpf_validator=True, auto_patterns=False)
        >>> anonymizer.add_apply_patterns(['CPF'])
        >>> text = "Ola meu nome é matheus e meu cpf é 000.000.000-00"
        >>> anonymizer.extract_entities(text, return_type='label_detail')
        []  # Retorna vazio pois 000.000.000-00 não é válido
        >>> # Desabilitar validação de CPF
        >>> anonymizer.use_cpf_validator = False
        >>> anonymizer.extract_entities(text, return_type='label_detail')
        [{'label': 'CPF',
        'start_position': 35,
        'end_position': 49,
        'text': '000.000.000-00',
        'detected_by': re.compile(r'\\d{3}[.]\\d{3}[.]\\d{3}[-.\\/]\\d{2}', re.UNICODE)}]


**Métodos principais:**
    - **add_custom_pattern:** Adiciona um padrão REGEX customizado para detecção de entidades.
    - **add_pattern_cpf:** Adiciona os padrões de regex para CPF.
    - **add_pattern_rg:** Adiciona os padrões de regex para RG.
    - **add_pattern_rg_estrangeiro:** Adiciona padrões de RNE/CRNM retornando label RG.
    - **add_pattern_titulo_eleitor:** Adiciona os padrões de regex para título de eleitor.
    - **add_pattern_passaporte:** Adiciona os padrões de regex para passaporte.
    - **add_pattern_siape:** Adiciona os padrões de regex para SIAPE.
    - **add_pattern_cnh:** Adiciona os padrões de regex para CNH.
    - **add_pattern_dados_bancarios:** Adiciona os padrões de regex para dados bancários.
    - **add_pattern_email:** Adiciona os padrões de regex para email.
    - **add_pattern_telefone:** Adiciona os padrões de regex para telefone.
    - **add_pattern_data_nascimento:** Adiciona os padrões de regex para data de nascimento.
    - **add_pattern_endereco:** Adiciona os padrões de regex para endereços/CEP.
    - **add_apply_patterns:** Aplica os padrões de regex para as entidades especificadas.
    - **list_patterns:** Retorna os padrões de regex organizados por categoria (built-in e customizados).
    - **verify_entities_unified:** Verifica se as entidades encontradas pelo modelo fazem sentido.
    - **remove_overlap_positions:** Remove entidades que se sobrepõem, priorizando aquelas com maior abrangência.
    - **extract_entities:** Extrai entidades do texto usando o modelo treinado e padrões regex.
    - **extract_entities_regex_re:** Extrai entidades do texto utilizando exclusivamente regex.
    - **anonymize_text:** Anonimiza o texto substituindo entidades detectadas por suas tags.
    - **get_tokens:** Retorna o texto tokenizado pelo spaCy.
    - **get_active_labels:** Retorna informações detalhadas sobre labels e padrões ativos.
    - **valida_cpf:** Valida um CPF completo (com ou sem formatação), incluindo dígitos verificadores.
    - **valida_titulo_eleitor:** Valida Título de Eleitor (12 dígitos) conforme regra do TSE.
    - **valida_cnpj:** Valida um CNPJ completo (com ou sem formatação), incluindo dígitos verificadores.
    - **_create_default_logger:** Cria logger padrão quando não fornecido.

**Configuração de logging:**

    O pacote usa o módulo padrão ``logging`` do Python. Configure o nível conforme o cenário:

    - ``WARNING``: apenas erros e avisos críticos (recomendado para lotes grandes).
    - ``INFO``: inicialização do modelo, início/fim de treinamento, métricas de fold.
    - ``DEBUG``: tudo acima + logs por documento e por entidade.

    Exemplo:

        >>> import logging
        >>> # Uso operacional (lote)
        >>> logging.getLogger("anonimizar").setLevel(logging.WARNING)
        >>>
        >>> # Depuração detalhada
        >>> logging.getLogger("anonimizar").setLevel(logging.DEBUG)
"""

import logging
from collections.abc import Sequence

import spacy

from anonimizar._anonymization.text import anonymize_text as _anonymize_text
from anonimizar._common.logging import create_default_logger
from anonimizar._common.overlap import remove_overlap_positions
from anonimizar._constants.entity_labels import (
    DEFAULT_SUPPORTED_LABELS,
)
from anonimizar._extraction.markdown import (
    extract_entities_from_markdown_tables as _extract_markdown_tables,
)
from anonimizar._extraction.pipeline import extract_entities as _extract_entities_pipeline
from anonimizar._extraction.regex import extract_entities_regex_re as _extract_regex
from anonimizar._patterns.builtin import (
    add_pattern_cid as _bp_add_pattern_cid,
)
from anonimizar._patterns.builtin import (
    add_pattern_cnh as _bp_add_pattern_cnh,
)
from anonimizar._patterns.builtin import (
    add_pattern_cpf as _bp_add_pattern_cpf,
)
from anonimizar._patterns.builtin import (
    add_pattern_dados_bancarios as _bp_add_pattern_dados_bancarios,
)
from anonimizar._patterns.builtin import (
    add_pattern_data_nascimento as _bp_add_pattern_data_nascimento,
)
from anonimizar._patterns.builtin import (
    add_pattern_email as _bp_add_pattern_email,
)
from anonimizar._patterns.builtin import (
    add_pattern_endereco as _bp_add_pattern_endereco,
)
from anonimizar._patterns.builtin import (
    add_pattern_geo_coord as _bp_add_pattern_geo_coord,
)
from anonimizar._patterns.builtin import (
    add_pattern_passaporte as _bp_add_pattern_passaporte,
)
from anonimizar._patterns.builtin import (
    add_pattern_passaporte_est as _bp_add_pattern_passaporte_est,
)
from anonimizar._patterns.builtin import (
    add_pattern_rg as _bp_add_pattern_rg,
)
from anonimizar._patterns.builtin import (
    add_pattern_rg_estrangeiro as _bp_add_pattern_rg_estrangeiro,
)
from anonimizar._patterns.builtin import (
    add_pattern_siape as _bp_add_pattern_siape,
)
from anonimizar._patterns.builtin import (
    add_pattern_telefone as _bp_add_pattern_telefone,
)
from anonimizar._patterns.builtin import (
    add_pattern_titulo_eleitor as _bp_add_pattern_titulo_eleitor,
)
from anonimizar._patterns.custom import (
    add_custom_pattern as _add_custom_pattern,
)
from anonimizar._patterns.registry import (
    add_apply_patterns as _registry_add_apply_patterns,
)
from anonimizar._patterns.registry import (
    get_active_labels as _registry_get_active_labels,
)
from anonimizar._patterns.registry import (
    list_patterns as _registry_list_patterns,
)
from anonimizar._validators.documents import (
    valida_cnh as _valida_cnh,
)
from anonimizar._validators.documents import (
    valida_cnpj as _valida_cnpj,
)
from anonimizar._validators.documents import (
    valida_cpf as _valida_cpf,
)
from anonimizar._validators.documents import (
    valida_titulo_eleitor as _valida_titulo_eleitor,
)
from anonimizar._validators.unified import (
    verify_entities_unified as _verify_unified,
)


class Anonimizar:
    """Classe para anonimização."""

    def __init__(  # noqa: PLR0915
        self,
        model_path: str = "",
        *,
        labels: list | None = None,
        use_cpf_validator: bool = True,
        use_titulo_validator: bool = True,
        use_cnh_validator: bool = True,
        auto_patterns: bool = True,
        normalize_entities: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        """Inicializa o anonimizador com um modelo spaCy.

        Carrega o modelo spaCy especificado e configura as definições padrão para
        processamento de texto e detecção de entidades. O modelo deve ser um modelo
        spaCy treinado para reconhecimento de entidades nomeadas em português.

        Este construtor carrega um modelo spaCy (público ou treinado pelo
        ``Trainer``), configura labels permitidos e define
        parâmetros de validação.

        Args:
            model_path (str): Caminho para o modelo spaCy pré-treinado.
                Exemplos: "pt_core_news_sm", "./meu_modelo_ner"
                ou caminho para modelo customizado.
            labels (list, optional): Lista de labels específicos para detecção.
            use_cpf_validator (bool): Se True, aplica validação algorítmica de CPF nas entidades
                detectadas.
            use_titulo_validator (bool): Se True, aplica validação algorítmica de Titulo de eleitor
                nas entidades detectadas.
            use_cnh_validator (bool): Se True, aplica validação algorítmica de CNH
                nas entidades detectadas.
            auto_patterns (bool): Se True (padrão), aplica automaticamente os padrões de regex
                para todas as entidades suportadas ao instanciar, incluindo RNE/CRNM como RG.
                Se False, o usuário deve chamar add_apply_patterns() manualmente.
            normalize_entities (bool): Se True (padrão), normaliza entidades removendo prefixos
                e sufixos textuais (ex: "RG: 1234567 SSP/DF" vira "1234567").
            logger (logging.Logger, optional): Logger personalizado para usar.

        Raises:
            ValueError: Se model_path estiver vazio ou não especificado.
            OSError: Se o modelo especificado não puder ser carregado.

        Note:
        - O limite máximo de caracteres por texto é definido como 3.000.000
        - Labels ignoradas por padrão: ["LOC", "MISC", "ORG", "PER"]
        - É necessário ter o modelo spaCy instalado: `python -m spacy download pt_core_news_lg`
        - Para desativar RNE/CRNM por padrão, use auto_patterns=False e registre RG com foreign_rg=False.

        Examples:
        >>> anonymizer = Anonimizar("./meu_modelo_ner", labels=["CPF", "EMAIL"])
        >>> anonymizer.extract_entities("Meu CPF é 123.456.789-09.")
        [{'label': 'CPF', 'start_position': 10, 'end_position': 24, ...}]
        """
        self.logger = logger or create_default_logger(__name__)
        msg = f"Inicializando Anonimizar com model_path: {model_path}"
        self.logger.info(msg)
        self.model_path = model_path
        if self.model_path == "":
            msg = "É necessario ter o model_path preenchido."
            self.logger.exception(msg)
            raise ValueError(msg)
        try:
            msg = f"Carregando modelo spaCy: {model_path}"
            self.logger.debug(msg)
            self.nlp_trained = spacy.load(self.model_path)
            self.nlp_trained.max_length = 3000000
            self.patterns = []
            self.labels = set(labels) if labels else set()
            self.use_cpf_validator = use_cpf_validator
            self.use_titulo_validator = use_titulo_validator
            self.use_cnh_validator = use_cnh_validator
            self.normalize_entities = normalize_entities
            if labels is None or len(labels) == 0:
                if "ner" in self.nlp_trained.pipe_names:
                    model_labels = list(self.nlp_trained.get_pipe("ner").labels)
                    self.labels = set(model_labels)
                    msg = f"Labels extraídos do modelo: {model_labels}"
                    self.logger.debug(msg)
                else:
                    msg = "Modelo não possui componente NER, usando labels padrão"
                    self.logger.error(msg)
                    raise TypeError(msg)  # noqa: TRY301
            else:
                self.labels = set(labels)
            msg = "Modelo carregado com sucesso."
            self.logger.info(msg)
            msg = f"\tMax length: {self.nlp_trained.max_length}"
            self.logger.debug(msg)
            msg = f"\tLabels configurados: {self.labels}"
            self.logger.debug(msg)
            msg = f"\tValidação de CPF ativa: {self.use_cpf_validator}"
            self.logger.debug(msg)
            msg = f"\tValidação de TITULO_ELEITOR ativa: {self.use_titulo_validator}"
            self.logger.debug(msg)
            msg = f"\tValidação de CNH ativa: {self.use_cnh_validator}"
            self.logger.debug(msg)

            if auto_patterns:
                self.add_apply_patterns(list(DEFAULT_SUPPORTED_LABELS), foreign_rg=True)
                self.logger.debug("Patterns aplicados automaticamente para: %s", DEFAULT_SUPPORTED_LABELS)

        except OSError as e:
            msg = f"Erro ao carregar modelo spaCy '{model_path}': {e!s}"
            self.logger.exception(msg)
            raise
        except Exception as e:
            msg = f"Erro inesperado durante inicialização do Anonimizar. {e!s}"
            self.logger.exception(msg)
            raise

    def add_custom_pattern(self, label: str, regex_pattern: str, description: str = "") -> None:
        """Adiciona padrão REGEX customizado. Wrapper para _patterns.custom.add_custom_pattern."""
        _add_custom_pattern(self.patterns, label, regex_pattern, description, self.logger)

    def add_pattern_cid(self) -> None:
        """Registra padrões regex para CID. Wrapper para _patterns.builtin.add_pattern_cid."""
        _bp_add_pattern_cid(self.patterns, self.logger)

    def add_pattern_cpf(self) -> None:
        """Registra padrões regex para CPF. Wrapper para _patterns.builtin.add_pattern_cpf."""
        _bp_add_pattern_cpf(self.patterns, self.logger)

    def add_pattern_endereco(self) -> None:
        """Registra padrões regex para CEPs. Wrapper para _patterns.builtin.add_pattern_endereco."""
        _bp_add_pattern_endereco(self.patterns, self.logger)

    def add_pattern_geo_coord(self) -> None:
        """Registra padrões regex para coordenadas. Wrapper para _patterns.builtin.add_pattern_geo_coord."""
        _bp_add_pattern_geo_coord(self.patterns, self.logger)

    def add_pattern_rg(self) -> None:
        """Registra padrões regex para RG. Wrapper para _patterns.builtin.add_pattern_rg."""
        _bp_add_pattern_rg(self.patterns, self.logger)

    def add_pattern_rg_estrangeiro(self) -> None:
        """Registra padrões regex para RG estrangeiro (RNE/CRNM). Wrapper."""
        _bp_add_pattern_rg_estrangeiro(self.patterns, self.logger)

    def add_pattern_titulo_eleitor(self) -> None:
        """Registra padrões regex para Título de Eleitor. Wrapper."""
        _bp_add_pattern_titulo_eleitor(self.patterns, self.logger)

    def add_pattern_passaporte(self) -> None:
        """Registra padrões regex para passaportes brasileiros. Wrapper."""
        _bp_add_pattern_passaporte(self.patterns, self.logger)

    def add_pattern_passaporte_est(self) -> None:
        """Registra padrões regex para passaportes estrangeiros. Wrapper."""
        _bp_add_pattern_passaporte_est(self.patterns, self.logger)

    def add_pattern_siape(self) -> None:
        """Registra padrões regex para SIAPE. Wrapper."""
        _bp_add_pattern_siape(self.patterns, self.logger)

    def add_pattern_cnh(self) -> None:
        """Registra padrões regex para CNH. Wrapper."""
        _bp_add_pattern_cnh(self.patterns, self.logger)

    def add_pattern_dados_bancarios(self) -> None:
        """Registra padrões regex para dados bancários. Wrapper."""
        _bp_add_pattern_dados_bancarios(self.patterns, self.logger)

    def add_pattern_email(self) -> None:
        """Registra padrões regex para e-mail. Wrapper."""
        _bp_add_pattern_email(self.patterns, self.logger)

    def add_pattern_telefone(self) -> None:
        """Registra padrões regex para telefones. Wrapper."""
        _bp_add_pattern_telefone(self.patterns, self.logger)

    def add_pattern_data_nascimento(self) -> None:
        """Registra padrões regex para datas de nascimento. Wrapper."""
        _bp_add_pattern_data_nascimento(self.patterns, self.logger)

    def add_apply_patterns(
        self,
        labels: list | None = None,
        custom_patterns: list[dict] | None = None,
        *,
        replace_patterns: bool = False,
        use_model_labels: bool = False,
        foreign_passport: bool = False,
        foreign_rg: bool = False,
    ) -> None:
        r"""Adiciona e aplica padrões de regex para detecção de entidades, incluindo padrões customizados.

        Este método permite configurar quais tipos de entidades serão detectadas pelo anonimizador,
        combinando padrões pré-definidos (como CPF, RG, etc.), padrões personalizados e, opcionalmente,
        entidades reconhecidas por um modelo spaCy.

        Args:
            labels (list, optional): Lista de rótulos dos padrões pré-definidos a serem aplicados.
                Labels com padrões regex: 'CPF', 'RG', 'TITULO_ELEITOR',
                'PASSAPORTE', 'SIAPE', 'DADOS_BANCARIOS', 'EMAIL', 'TELEFONE',
                'DATA_NASCIMENTO', 'CNH', 'ENDEREÇO', 'GEO_COORD', 'CID',
                'PIS', 'CNS' e 'RESERVISTA'.
                'FISTEL' é uma compatibilidade que ativa os padrões de 'CPF'.

                - Quando `use_model_labels` for False, este argumento é obrigatório e não pode ser vazio.
                - Quando `use_model_labels` for True, este argumento é opcional e será combinado aos
                  rótulos fornecidos pelo modelo (`self.labels`), se presentes.

            custom_patterns (list[dict], optional): Lista de padrões customizados no formato:
                [{"label": "MEU_PADRAO", "regex": "meu-regex", "description": "Descrição opcional"}].
                O campo `"label"` e `"regex"` são obrigatórios para cada padrão.
                Defaults to None.

            replace_patterns (bool, optional): Se True, remove todos os padrões previamente
                adicionados antes de aplicar os novos padrões especificados. Se False,
                os novos padrões são adicionados aos já existentes.
                Útil para reconfigurar completamente o anonimizador ou para evitar duplicação de padrões.
                Defaults to False.

            use_model_labels (bool, optional): Se True, ativa a detecção por modelo spaCy
                para os tipos de entidade definidos em `self.labels`. Caso o argumento `labels` também
                seja fornecido, os dois serão combinados.
                Defaults to False.

            foreign_passport (bool, optional): Se True e a label `PASSAPORTE` ativada irá adicionar
                os regex de passaporte estrangeiro.
                Defaults to False.

            foreign_rg (bool, optional): Se True e a label `RG` ativada irá adicionar
                os regex de RG estrangeiro (RNE/CRNM), mantendo a label de retorno como `RG`.
                Defaults to False.

        Raises:
            ValueError:
                - Se `labels` estiver vazio ou None quando `use_model_labels` for False.
                - Se algum rótulo em `labels` não for suportado.
                - Se algum padrão customizado não possuir os campos obrigatórios (`label`, `regex`).

        Examples:
            Aplicar padrões básicos:
            >>> anonymizer = Anonimizar(model_path="pt_core_news_lg", auto_patterns=False)
            >>> anonymizer.add_apply_patterns(['CPF', 'EMAIL'])

            Incluir RG estrangeiro manualmente:
            >>> anonymizer.add_apply_patterns(['RG'], foreign_rg=True)

            Substituir padrões existentes:
            >>> anonymizer.add_apply_patterns(['RG', 'TELEFONE'], replace_patterns=True)

            Adicionar padrões customizados:
            >>> custom = [{"label": "CODIGO_PRODUTO", "regex": r"PROD-\d{4}",
            ...            "description": "Códigos de produto"}]
            >>> anonymizer.add_apply_patterns(['CPF'], custom_patterns=custom)

            Reconfigurar completamente com novos padrões:
            >>> anonymizer.add_apply_patterns(['EMAIL', 'TELEFONE'],
            ...                              custom_patterns=custom,
            ...                              replace_patterns=True)

            Usar labels do modelo e combinar com outros padrões:
            >>> anonymizer.add_apply_patterns(['EMAIL'], use_model_labels=True)
        """
        _registry_add_apply_patterns(
            self.patterns,
            labels,
            list(self.labels),
            self.logger,
            replace_patterns=replace_patterns,
            use_model_labels=use_model_labels,
            foreign_passport=foreign_passport,
            foreign_rg=foreign_rg,
            custom_patterns=custom_patterns,
        )

    def list_patterns(self) -> dict:
        """Retorna todos os padrões regex registrados, organizados por categoria.

        A classificação usa a presença do label em ``self.labels``: padrões com
        labels ativos no modelo ficam em ``builtin``; os demais ficam em ``custom``.
        Ela não representa necessariamente a origem do padrão.

        Returns:
            dict: Dicionário com duas chaves:
                - "builtin": Padrões cujo label está ativo no modelo
                - "custom": Padrões cujo label não está ativo no modelo

                Cada padrão contém:
                - label: Rótulo da entidade
                - pattern: Dicionário com a regex
                - description: Descrição do padrão (se disponível)

        Note:
            - Útil para debug e auditoria dos padrões ativos
        """
        return _registry_list_patterns(self.patterns, list(self.labels))

    def get_entity_attribute(self, entity: dict | object, attr_name: str) -> object:
        """Obtém um atributo de uma entidade em formato de dicionário ou objeto.

        Args:
            entity (dict | object): Dicionário indexável ou objeto que expõe o atributo.
            attr_name (str): Nome da chave ou atributo a obter.

        Returns:
            object: O valor associado a ``attr_name``.

        Raises:
            KeyError: Se ``entity`` for um dicionário sem a chave solicitada.
            AttributeError: Se o objeto não expuser o atributo solicitado.
        """
        try:
            return getattr(entity, attr_name)
        except AttributeError:
            return entity[attr_name]

    def verify_entities_unified(self, entity: dict, text: str) -> bool | str:
        """Valida entidades detectadas pelo modelo spaCy ou regex com base no contexto textual.

        A heurística verifica palavras-chave próximas, tamanhos máximos,
        dígitos verificadores (para CPF) e outros critérios específicos
        descritos abaixo.

        Args:
            entity (dict): Objeto de entidade spaCy contendo:
                - label_: Rótulo da entidade detectada
                - start_char: Posição inicial no texto
                - end_char: Posição final no texto
            text (str): Texto completo onde a entidade foi encontrada.

        Returns:
            bool | str:
                - True: Se a entidade foi validada com o rótulo original
                - False: Se a entidade deve ser desconsiderada
                - str: Novo rótulo da entidade (reclassificação)

        Note:
            Validações específicas por tipo:
            - CPF: Verifica contexto "cpf", "fistel", "cnh" e limites de tamanho
            - CNH: Busca palavra "cnh" no contexto
            - SIAPE: Procura "siape" vs "sei" no contexto
            - DATA_NASCIMENTO: Identifica palavras como "nascimento", "emissor"
            - PASSAPORTE: Valida comprimento máximo
        """
        return _verify_unified(
            entity,
            text,
            self.logger,
            use_cpf_validator=self.use_cpf_validator,
            use_cnh_validator=self.use_cnh_validator,
            use_titulo_validator=self.use_titulo_validator,
        )

    def remove_overlap_positions(self, entities: list[dict]) -> list[dict]:
        """Remove as entidades com overlap deixando apenas a entidade com maior abrangência.

        Este método é um wrapper para a função standalone em _common.overlap.
        Mantido para compatibilidade com código existente.

        Args:
            entities: Lista com as entidades detectadas.

        Returns:
            Lista de entidades sem overlaps.
        """
        return remove_overlap_positions(entities, logger=self.logger)

    def extract_entities_from_markdown_tables(self, text: str, return_type: str = "label_position") -> list[dict]:
        """Extrai entidades de colunas com dados pessoais em tabelas markdown.

        Detecta automaticamente tabelas markdown no texto e extrai entidades de colunas
        que contenham nomes relacionados a dados pessoais (CPF, RG, Titulo, Documento).

        Args:
            text (str): Texto contendo tabelas markdown
            return_type (str): Formato de retorno das entidades

        Returns:
            list[dict]: Lista de entidades encontradas nas tabelas
        """
        return _extract_markdown_tables(text, return_type, self.logger)

    def extract_entities(self, text_or_path: str, return_type: str = "label_position") -> list[dict]:
        r"""Extrai dados pessoais de texto usando modelo spaCy e padrões regex.

        Combina detecção por modelo de Machine Learning com padrões regex personalizados
        para identificar entidades como CPF, RG, e-mail, telefone, etc. Remove sobreposições
        automáticamente e valida entidades com base no contexto.

        Fluxo simplificado:
            1. Carrega texto (string ou arquivo `.md`).
            2. Executa o modelo spaCy filtrando pelos labels ativos.
            3. Aplica todos os REGEX cadastrados.
            4. Remove sobreposições e converte no formato solicitado.

        Args:
            text_or_path (str): Texto a ser processado ou caminho para arquivo .md
                contendo o texto. Se for um caminho de arquivo, deve ter extensão .md.
            return_type (str, optional): Formato do retorno das entidades encontradas.
                Opções disponíveis:
                - "label_position": {'label': str, 'start_position': int, 'end_position': int}
                - "label_text": {'label': str, 'text': str}
                - "label_detail": Inclui todos os campos + 'detected_by'
                Defaults to "label_position".

        Returns:
            list[dict]: Lista de entidades encontradas no formato especificado:
                - label_position: Posições das entidades no texto
                - label_text: Textos das entidades sem posições
                - label_detail: Informações completas incluindo método de detecção

        Raises:
            ValueError: Se o arquivo não for .md ou return_type for inválido.

        Note:
            - Entidades sobrepostas são automaticamente removidas
            - Validação contextual é aplicada a todas as entidades
            - Combina resultados de modelo spaCy e regex
            - Labels ignoradas: ["LOC", "MISC", "ORG", "PER"]
            - A normalização de entidades (remoção de prefixos/sufixos) é controlada
              pelo atributo ``self.normalize_entities``.

        """
        return _extract_entities_pipeline(
            self.nlp_trained,
            text_or_path,
            self.labels,
            self.patterns,
            return_type,
            self.verify_entities_unified,
            self.logger,
            normalize=self.normalize_entities,
        )

    def extract_entities_regex_re(self, text: str, return_type: str = "label_position") -> list[dict]:
        """Extrai entidades do texto usando exclusivamente padrões regex configurados.

        Aplica todos os padrões regex registrados ao texto e valida cada entidade encontrada
        através de análise contextual. Pode ser chamado independentemente para detecção
        apenas por regex.

        Args:
            text (str): Texto a ser processado para extração de entidades.
            return_type (str, optional): Formato de retorno das entidades:
                - "label_text": {'label': str, 'text': str}
                - "label_position": {'label': str, 'start_position': int, 'end_position': int}
                - "label_detail": Informações completas + objeto regex usado
                Defaults to "label_position".

        Returns:
            list[dict]: Lista de entidades encontradas pelos padrões regex.
                O formato específico depende do return_type:
                - label_text: Apenas rótulo e texto da entidade
                - label_position: Rótulo e posições no texto
                - label_detail: Todos os campos + referência ao regex usado

        Note:
            - Cada padrão regex é aplicado independentemente
            - Validação contextual é aplicada via verify_entities_unified()
            - Entidades inválidas são automaticamente filtradas
            - Posições são ajustadas para remover espaços extras
        """
        return _extract_regex(text, self.patterns, return_type, self.verify_entities_unified, self.logger)

    def anonymize_text(self, text: str, entities_list: list) -> str:
        """Substitui entidades detectadas no texto por tags de anonimização.

        Processa uma lista de entidades extraídas e substitui cada ocorrência no texto
        original por uma tag formatada. As substituições são feitas em ordem reversa
        para preservar as posições corretas dos caracteres.

        Args:
            text (str): Texto original a ser anonimizado.
            entities_list (list[dict]): Lista de entidades extraídas usando return_type
                "label_position" ou "label_detail". Cada entidade deve conter:
                - 'start_position': posição inicial no texto
                - 'end_position': posição final no texto
                - 'label': tipo da entidade

        Returns:
            str: Texto anonimizado onde entidades são substituídas por tags no formato
                <|TIPO_ENTIDADE|>. Exemplo: <|CPF|>, <|EMAIL|>, <|TELEFONE|>.

        Note:
            - Substituições são feitas em ordem reversa (do fim para o início)
            - Entidades com labels em ignore_labels são puladas
            - Preserva formatação e espaçamento do texto original
            - Tags seguem o padrão: <|LABEL|>
            - Prefixos fora dos spans, como ``CPF `` e ``RG: ``, são preservados.

        Examples:
            >>> text = "João, CPF 123.456.789-09, email: joao@email.com"
            >>> entities = [
            ...     {'label': 'CPF', 'start_position': 10, 'end_position': 24},
            ...     {'label': 'EMAIL', 'start_position': 33, 'end_position': 47}
            ... ]
            >>> result = anonymizer.anonymize_text(text, entities)
            >>> print(result)
            "João, CPF <|CPF|>, email: <|EMAIL|>"

            >>> # Com múltiplas entidades
            >>> text = "Contato: (11) 99999-9999, RG: 12.345.678-9"
            >>> entities = [
            ...     {'label': 'TELEFONE', 'start_position': 9, 'end_position': 24},
            ...     {'label': 'RG', 'start_position': 31, 'end_position': 43}
            ... ]
            >>> result = anonymizer.anonymize_text(text, entities)
            >>> print(result)
            "Contato: <|TELEFONE|>, RG: <|RG|>"
        """
        return _anonymize_text(text, entities_list, self.logger)

    def get_tokens(self, text: str) -> list:
        r"""Retorna o texto tokenizado usando o modelo spaCy carregado.

        Processa o texto através do pipeline spaCy e retorna uma lista de tokens
        mantendo espaços e formatação original. Útil para análise detalhada do
        texto ou processamento posterior.

        Args:
            text (str): Texto a ser tokenizado.

        Returns:
            list: Lista de strings contendo cada token com seus espaços preservados.
                Cada elemento mantém a formatação original (espaços, quebras de linha).

        Note:
            - Utiliza o modelo spaCy carregado na inicialização
            - Preserva espaços em branco e formatação
            - Tokens incluem pontuação como elementos separados

        Examples:
            >>> tokens = anonymizer.get_tokens("Olá, mundo! Como vai?")
            >>> print(tokens)
            ['Olá', ', ', 'mundo', '! ', 'Como ', 'vai', '?']

            >>> # Com quebras de linha
            >>> tokens = anonymizer.get_tokens("Primeira linha\\nSegunda linha")
            >>> print(tokens)
            ['Primeira ', 'linha\\n', 'Segunda ', 'linha']
        """
        try:
            self.logger.debug("Tokenizando texto (len=%d)", len(text))
            doc = self.nlp_trained(text)
            tokens = [token.text_with_ws for token in doc]
            self.logger.debug("Total de tokens gerados: %d", len(tokens))
        except Exception as e:
            self.logger.exception("Erro ao tokenizar texto: %s", e)  # noqa: TRY401
            raise e from e
        return tokens

    def get_active_labels(self) -> dict:
        """Retorna informações detalhadas sobre labels ativos no anonimizador.

        Returns:
            dict: Dicionário contendo:
                - "model_labels": Lista de labels ativos para detecção por modelo spaCy
                - "regex_summary": Dicionário com labels e quantidade de padrões regex por label
                - "regex_patterns": Número total de padrões regex registrados
                - "total_patterns": Soma de labels do modelo com total de padrões regex

        Examples:
            >>> anonymizer = Anonimizar(model_path="pt_core_news_lg")
            >>> anonymizer.add_apply_patterns(['CPF', 'EMAIL'], use_model_labels=True)
            >>> info = anonymizer.get_active_labels()
            >>> print(info)
            {
                'model_labels': ['CPF', 'EMAIL'],
                'regex_summary': {'CPF': 3, 'EMAIL': 2},
                'regex_patterns': 5,
                'total_patterns': 7
            }
        """
        return _registry_get_active_labels(self.patterns, list(self.labels))

    @staticmethod
    def valida_cpf(cpf: str) -> bool:
        """Valida CPF. Wrapper para _validators.documents.valida_cpf."""
        return _valida_cpf(cpf)

    @staticmethod
    def valida_titulo_eleitor(numero: str) -> bool:
        """Valida Título de Eleitor. Wrapper para _validators.documents.valida_titulo_eleitor."""
        return _valida_titulo_eleitor(numero)

    @staticmethod
    def valida_cnh(numero: str) -> bool:
        """Valida CNH. Wrapper para _validators.documents.valida_cnh."""
        return _valida_cnh(numero)

    @staticmethod
    def valida_cnpj(cnpj: str) -> bool:
        """Valida CNPJ. Wrapper para _validators.documents.valida_cnpj."""
        return _valida_cnpj(cnpj)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    import os

    from dotenv import load_dotenv

    load_dotenv()

    model_path_env = os.getenv("SPACY_MODEL_PATH")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(description="Anonimizador de textos")
    parser.add_argument(
        "--text_or_path",
        type=str,
        required=True,
        help=("O texto a ser processado ou path do arquivo com o texto a ser processado."),
    )
    parser.add_argument(
        "--return_type",
        type=str,
        default="label_position",
        help=("label_text, label_position ou label_detail, muda a forma de retorno das entidades"),
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help=("Caminho para o modelo spacy treinado para a identificacao das entidades."),
    )

    args = parser.parse_args(argv)

    if (args.model_path == "") and (model_path_env is None):
        msg = "A variavel de ambiente  SPACY_MODEL_PATH ou o argumento -- model_path deve estar preenchido"
        raise ValueError(msg)
    if (args.model_path != "") and (model_path_env is None):
        model_path_env = args.model_path

    anonymizer = Anonimizar(model_path=model_path_env)

    anonymizer.add_apply_patterns(
        [
            "CPF",
            "RG",
            "TITULO_ELEITOR",
            "PASSAPORTE",
            "SIAPE",
            "TELEFONE",
            "DATA_NASCIMENTO",
            "ENDEREÇO",
            "DADOS_BANCARIOS",
            "EMAIL",
            "CNH",
            "GEO_COORD",
            "CID",
        ]
    )

    response = anonymizer.extract_entities(text_or_path=args.text_or_path, return_type=args.return_type)
    msg = f"Entidades extraídas do texto '{args.text_or_path}' com return_type '{args.return_type}':"
    logging.info(msg)
    msg = f"Total de entidades encontradas: {len(response)}"
    logging.info(msg)
    msg = f"\n{json.dumps(response, indent=4, ensure_ascii=False)}\n"
    logging.info(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
