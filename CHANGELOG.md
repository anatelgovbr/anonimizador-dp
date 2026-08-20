# Changelog - Biblioteca Anonimizar

## v1.1.0

### Novidades

- **Curriculum learning no `Trainer`**: novo método `Trainer.train_curriculum()`
  treina o modelo por fases sequenciais com janelas de dificuldade
  (`w0`/`w1`/`w2`/`full`/`w00`), reproduzindo o padrão validado nos
  experimentos da estória 942. Dois fluxos de entrada, misturáveis entre
  fases, ambos com a chave `dataset`: end-to-end (`df_textos` + `df_entidades`
  ou `.jsonl`, fases pelo nome da janela em `dataset`/`conjunto`) e separado
  (dados prontos em `dataset`, incluindo tuplas `(texto, {"entities": [...]})`
  dos joblibs da 942). String que não termina em `.jsonl` é tratada como nome
  de janela; qualquer outro valor é dado pronto.
- **Novo módulo `anonimizar._training.curriculum_data`**: geração de datasets
  por janela de contexto (`build_context_window_dataset` com `window` int ≥ 0
  ou `"full"`), documento completo (`build_full_text_dataset`), entidade pura
  isolada com oversampling (`build_pure_entity_dataset`) e o conjunto completo
  (`build_curriculum_datasets`); persistência joblib compatível com os
  datasets da 942 (`save_curriculum_datasets`/`load_curriculum_datasets`).
- **Filtros herdados da 942**: documentos com `TEM_ERRO == True` removidos por
  completo, entidades com label `_remover` descartadas, deduplicação e spans
  inválidos (`start >= end`) removidos.
- **`train_ner_model_curriculum`** em `anonimizar._training.trainer`:
  núcleo de treinamento em fases com `batch_compounding` opcional
  (minilotes crescentes) e `nlp.initialize()` na 1ª fase.
- **Labels novos registrados automaticamente**: labels encontradas nas fases
  fora de `supported_labels` (ex.: `PIS`, `CNS`) são adicionadas ao modelo e
  ao `supported_labels` antes do treino.
- Nova constante `DEFAULT_CURRICULUM_WINDOWS` em `_constants/thresholds.py` e
  novos exports públicos em `anonimizar._training`.

### Testes e Qualidade

- Novo `tests/test_training_curriculum_data.py` (18 testes): janelas
  w0/w1/w2/full, entidade pura, oversampling, filtros da 942, persistência
  joblib e conversão do formato 942.
- `tests/test_training_trainer.py`: suíte `TestTrainNERModelCurriculum` para
  `train_ner_model_curriculum` (ordem das fases, validações, compounding).
- `tests/test_sei_anonimizar_training.py`: suíte `TestTrainerCurriculum` para
  os fluxos e2e/separado do `train_curriculum`, erros de validação e registro
  de labels novos.
- README, `docs/README.md` e pdoc3 atualizados com o curriculum learning.

## v1.0.6

### Correções

- **Dependência `click` declarada explicitamente**: o `click` (8.1.8) passa a
  ser dependência direta do pacote, garantindo sua instalação mesmo quando o
  `spacy`/`typer` não o resolvem no ambiente do usuário.
- **Versão interna corrigida**: `__version__` em `anonimizar/__init__.py`
  alinhado à versão do pacote (a 1.0.5 publicada reportava 1.0.4 e foi
  removida do PyPI).

## v1.0.4

### Mudanças Estruturais

- **Renomeação de classes públicas**: `SeiAnonimizar` → `Anonimizar`,
  `SeiAnonimizarEvaluation` → `Evaluation`,
  `SeiAnonimizarNERTrainer` → `Trainer`.
- **Refatoração de código-fonte**: classes movidas para módulos dedicados
  (`_anonymization/anonymizer.py`, `_evaluation/evaluation.py`,
  `_training/trainer_facade.py`).
- **Compatibilidade reversa**: módulos-ponte `sei_anonimizar.py`,
  `sei_anonimizar_evaluation.py` e `sei_anonimizar_treino.py` emitem
  `DeprecationWarning` e delegam para as classes renomeadas; aliases
  deprecados em `__init__.__all__` e `__getattr__`.
- **CLI extraída**: `main(argv)` em `_anonymization/anonymizer.py` com
  caminho de execução via `python -m anonimizar --text_or_path=...`;
  `python -m anonimizar.sei_anonimizar` continua funcionando com
  `DeprecationWarning`.

### Testes e Qualidade

- Novo `test_legacy_imports.py` com testes de identidade, warnings e CLI.
- `test_init.py` ampliado para verificar aliases deprecados e identidade.
- Ajustes de patch targets e imports em ~14 arquivos de teste.
- `ruff check .` e `ruff format .` limpos (apenas warns pré-existentes
  em notebooks/scratch não pertencentes ao pacote).
- Cobertura mínima de 85% mantida.

## v1.0.3

### Novas Funcionalidades

- **Normalização pública de entidades**: disponibilizado o subpacote
  `anonimizar._normalization`, com funções offset-safe para remover prefixos,
  sufixos e espaços de entidades sem perder a correspondência com o texto
  original.

### Melhorias

- **Avaliação de spans**: o cálculo de overlap foi alinhado à razão de
  interseção sobre união (IoU), tornando a comparação entre ground truth e
  predições consistente em todo o fluxo de avaliação.
- **Extração e validação**: corrigido o tratamento de prefixos de e-mail e
  ampliada a normalização de entidades no pipeline de extração e nos dados de
  treinamento.
- **Contratos de API e pdoc3**: revisadas docstrings, exemplos, retornos e
  exceções das APIs públicas e internas de extração, anonimização, treinamento
  e avaliação. A documentação HTML agora descreve com maior precisão schemas,
  callbacks, formatos de exportação e casos de retorno vazio.
- **Dados de demonstração**: a documentação pública passou a declarar que os
  dados de treino e exemplos distribuídos são fictícios e preservam somente os
  padrões estruturais de documentos reais.

### Testes e Qualidade

- Adicionados testes de regressão para exemplos distribuídos, contratos de
  predições vazias, normalização e políticas de limpeza de dados de treino.
- Atualizados exemplos de uso, notebooks de demonstração e script de políticas
  de erro para usar imports, offsets e resultados compatíveis com a API atual.

## v1.0.2

### Novas Funcionalidades

- **Novas entidades PIS/PASEP/NIT (`PIS`), Cartão Nacional de Saúde (`CNS`) e Certificado de Reservista (`RESERVISTA`)**:
    - Adicionadas às listas de `DEFAULT_SUPPORTED_LABELS`, `ALL_MODEL_LABELS`, `ENTITY_PRIORITY_LIST` e `DEFAULT_ENTITY_MAPPING` (com `PASEP` e `NIT` mapeados para `PIS`).
    - Novos padrões regex em `add_pattern_pis()`, `add_pattern_cns()` e `add_pattern_reservista()`, registrados em `PATTERN_ADDERS`.
    - Novos validadores algorítmicos `valida_pis()` (módulo 11, pesos `[3,2,9,8,7,6,5,4,3,2]`) e `valida_cns()` (15 dígitos, primeiro dígito em `1/2/7/8/9`, módulo 11) em `_validators/documents.py`.
    - Novos validadores contextuais `validate_pis_context`, `validate_cns_context` e `validate_reservista_context`, exigindo palavra-chave de contexto além do dígito verificador para aceitar a entidade (evita falsos positivos de números de 11/15 dígitos soltos no texto).
    - Novas listas de palavras-chave `KEYWORDS_PIS`, `KEYWORDS_CNS` e `KEYWORDS_RESERVISTA`, e janelas de contexto próprias em `CONTEXT_WINDOWS`.
    - Novos markers de teste `pis`, `cns` e `reservista` em `pytest.ini`.
- **RG de conselhos profissionais (CRM, CRO, CRP, CREA, CRF, CRESS, CRECI, COREN, CRMV)**:
    - Novos padrões regex em `add_pattern_rg()` para capturar número + UF do conselho em diferentes ordens (`CRM/UF 12345`, `12345-CRM/UF`, `CRM: 12345`).
    - `_validate_rg_issuer()` passa a aceitar automaticamente RG cujo texto contenha sigla de conselho profissional, sem exigir a mesma verificação estrita de UF usada para órgãos emissores tradicionais.

### Melhorias

- **`add_apply_patterns`**: labels reconhecidas em `ALL_MODEL_LABELS` que não possuem padrões regex próprios (ex.: labels detectadas apenas pelo modelo spaCy) deixam de gerar erro de "rótulo não suportado" e passam a ser apenas logadas em `DEBUG`.
- **`_validate_rg_issuer`**: além de UF ao final ou após `/`, passa a aceitar UF seguida de espaço e dígitos (`SSP MG 123456`), ampliando a detecção de órgãos emissores tradicionais.
- **Fixtures de teste**: nova fixture `anonymizer_rne` (RG com `foreign_rg=True`) e `geo_anonymizer` (somente padrões `GEO_COORD`) em `tests/conftest.py`; fixture `anonymizer` passa a incluir `PIS`, `CNS` e `RESERVISTA` por padrão.

### Testes e Qualidade

- **Reorganização da suíte de testes**: `tests/test_sei_anonimizar.py` (1575 linhas) foi dividido em arquivos temáticos menores — `test_extract_entities.py`, `test_extract_markdown_tables.py`, `test_patterns.py`, `test_remove_overlap.py`, `test_valida_documentos.py`, `test_verify_entities.py`, `test_anonymize_text.py` — e `test_auto_patterns.py` foi renomeado para `test_construtor.py`, absorvendo também os testes de inicialização da classe.
- `tests/test_email_validation.py`, `tests/test_rg_estrangeiro.py`, `tests/test_rg_regression.py` e `tests/test_validators_documents.py` foram incorporados aos novos arquivos temáticos.
- Novo módulo `tests/_helpers.py` com utilitários compartilhados (`_make_logger`, `make_ent`, `_find_span`) para reduzir duplicação entre arquivos de teste.
- Novos testes de regressão para PIS, CNS, RESERVISTA e RG de conselhos profissionais.
- Correções pontuais de lint (`ruff`) em `_training/data_validator.py` (uso de f-string em log e `if`/`else` aninhado desnecessário) e em testes (ordenação de imports, `open()` sem modo redundante).

## v1.0.1

### Novas Funcionalidades

- **Padrões automáticos por padrão**: `SeiAnonimizar(..., auto_patterns=True)` aplica os padrões built-in na inicialização, sem exigir chamada manual a `add_apply_patterns()` para o uso comum.
- **RG estrangeiro integrado**: RNE/CRNM passam a ser detectados automaticamente como entidades com label `RG`; o comportamento pode ser desativado com `auto_patterns=False` e registro manual de `RG` com `foreign_rg=False`.

### Melhorias

- **EMAIL**: validações e exemplos revisados para reduzir falsos positivos em URLs e formatos inválidos.
- **RG**: padrões e validação contextual aprimorados para casos com UF, RNE/CRNM e prefixos como `RG:` antes do valor.
- **SIAPE**: remoção de padrão excessivamente amplo e ajuste de validação contextual para reduzir falsos positivos.
- **TITULO_ELEITOR**: suporte ampliado para formatos contextuais com 10 a 12 dígitos.
- **Logging**: documentação do uso operacional com `WARNING`/`INFO` e depuração detalhada com `DEBUG`.

### Testes e Qualidade

- Novos testes de regressão para `auto_patterns`, RNE/CRNM como `RG`, validação de EMAIL, logging e casos de RG com prefixo/contexto.
- Suite principal validada com `pytest tests/` e cobertura acima do mínimo configurado para o pacote principal.

## v1.0.0

### Refatoração Arquitetural

**Nota:** Compatibilidade de APIs públicas mantida — nenhuma breaking change.
As classes `SeiAnonimizar`, `SeiAnonimizarEvaluation` e `SeiAnonimizarNERTrainer`
permanecem com a mesma interface.

- **Separação de `_constants.py`**: Dividido em 5 submódulos
    - `entity_labels.py`, `formats.py`, `keywords.py`, `thresholds.py`, `validators.py`
    - Seguindo o princípio de responsabilidade única (SRP)

- **Novo módulo `_training/`**: 8 arquivos para treinamento NER
    - `trainer.py`, `cross_validation.py`, `cv_manager.py`
    - `data_loader.py`, `data_manager.py`, `data_validator.py`
    - `io.py`, `io_handler.py`

- **Novo módulo `_evaluation/`**: 5 arquivos para avaliação de modelos
    - `metrics.py`, `reporter.py`, `comparison.py`
    - `data_loader.py`, `predictor.py`

- **Novo módulo `_extraction/`**: 4 arquivos para extração de entidades
    - `pipeline.py`, `model.py`, `regex.py`, `markdown.py`

- **Novo módulo `_patterns/`**: 3 arquivos para gerenciamento de padrões regex
    - `builtin.py`, `custom.py`, `registry.py`

- **Novo módulo `_validators/`**: 4 arquivos para validação de entidades
    - `documents.py`, `context.py`, `registry.py`, `unified.py`

- **Novo módulo `_common/`**: Utilitários compartilhados
    - `logging.py`, `overlap.py`

- **Novo módulo `_anonymization/`**: Lógica de anonimização de texto
    - `text.py`

### Testes

- Expansão de ~6 arquivos de teste para **19 arquivos**
- Cobertura expandida para todos os novos módulos estruturados

### Qualidade de Código

- Type hints obrigatórios em todas as funções (regra ANN do Ruff)
- Docstrings Google-style em português em todas as funções públicas
- Limpeza de código automatizada com `ruff check` e `ruff format`


## v0.0.10

### Novas Funcionalidades

- **Detecção de Entidades em Tabelas Markdown**: Suporte completo para extração de dados pessoais em tabelas
    - Novo método `extract_entities_from_markdown_tables()` para processar tabelas markdown
    - Detecção automática de colunas com dados pessoais por palavras-chave (CPF, RG, Título, Documento, Passaporte, CNH, SIAPE)
    - Identificação automática de tabelas markdown no texto (linhas com `|` e separador `---`)
    - Extração de entidades de células em colunas identificadas como contendo dados pessoais
    - Tag especial `detectedby: "tabela-markdown"` para rastreabilidade da origem
    - Suporte aos formatos de retorno `labelposition`, `labeltext` e `labeldetail`

### Melhorias Importantes

- **Integração Automática com Pipeline Principal**: `extract_entities()` agora processa tabelas automaticamente
    - Chamada automática de `extract_entities_from_markdown_tables()` durante extração
    - Combinação de entidades de tabelas com resultados do modelo spaCy e regex
    - Remoção de sobreposições unificada incluindo entidades de tabelas
    - Workflow completo: texto → modelo spaCy → regex → tabelas → remoção de overlaps
- **Validação de Estrutura de Tabelas**: Verificação de formato markdown válido
    - Validação de cabeçalho com separadores `|`
    - Verificação de linha divisória com padrão `---`
    - Mínimo de 3 linhas necessárias (header, separator, data)
    - Tratamento robusto de erros com logging detalhado

### Melhorias de Usabilidade

- **Logging Detalhado**: Mensagens de debug específicas para processamento de tabelas
    - Log de quantidade de entidades extraídas de tabelas markdown
    - Logging de erros durante processamento de tabelas individuais
    - Rastreamento de linha onde ocorreram erros
- **Flexibilidade de Dados**: Detecção baseada em contexto de coluna
    - Case-insensitive na identificação de colunas com dados pessoais
    - Suporte a múltiplas colunas com dados pessoais na mesma tabela
    - Rótulo da coluna usado como label da entidade extraída

### Exemplos de Uso

```python
# Texto com tabela markdown
texto = """
| Nome | CPF | RG |
|------|-----|-----|
| João | 123.456.789-00 | 12.345.678-9 |
| Maria | 987.654.321-00 | 98.765.432-1 |
"""

anonymizer = SeiAnonimizar(PATH)
anonymizer.add_apply_patterns(["CPF", "RG"])

# Extração automática inclui tabelas
entidades = anonymizer.extract_entities(texto, return_type="label_detail")
# Retorna entidades com detectedby: "tabela-markdown"
```

### Notas Técnicas

- Tabelas são processadas após modelo e regex, mas antes da remoção de overlaps
- Colunas com dados pessoais identificadas via keywords: `["cpf", "rg", "titulo", "documento", "passaporte", "cnh", "siape"]`
- Posições (`startposition`/`endposition`) são calculadas no texto original completo
- Compatível com todos os `return_type` existentes: `labelposition`, `labeltext`, `labeldetail`


## v0.0.9

### Novas Funcionalidades

- **Holdout Test Set em Cross-Validation**: Adicionado suporte a conjunto de teste fixo no `cross_validate()`
    - Novo parâmetro `holdout_test_size` para separar dados de teste antes do CV (ex: 0.2 = 20% teste fixo)
    - Novo parâmetro `holdout_stratify` para estratificação do conjunto holdout
    - Avaliação automática de todos os folds no mesmo conjunto holdout para comparabilidade de métricas
    - Salvamento de estatísticas agregadas do holdout test (`holdout_test_stats.json`)
    - Arquivos gerados: `holdout_test_ids.json/csv`, `holdout_test_summary.parquet/csv`, `holdout_test_detailed.parquet` por fold
- **Suporte a Formato JSONL Doccano**: Integração completa com formato de exportação do Doccano
    - Método `load_from_doccano_jsonl()` para carregar anotações
    - Método `save_to_doccano_jsonl()` para exportar dados de treinamento
    - Método `load_jsonl_to_dataframes()` para converter JSONL em DataFrames
    - Suporte a formatos `labels` e `entities` do Doccano
    - Parâmetro `data` em `add_data()` aceita caminho para arquivo `.jsonl`
    - Parâmetros `df_entidades` e `df_textos` em `cross_validate()` aceitam caminhos para `.jsonl`
- **Rastreabilidade de Folds**: Salvamento automático de IDs usados em cada fold
    - Arquivo `fold_ids.json` com metadados completos (train_ids, val_ids, contagens)
    - Arquivos `train_ids.csv` e `val_ids.csv` para cada fold
    - Facilita reprodutibilidade e análise de distribuição entre folds


### Melhorias Importantes

- **Validação de Entidades Aprimorada**: Método `validate_entities()` mais robusto
    - Adicionada validação `start >= end` para detectar offsets inválidos
    - Adicionada validação `start < 0` para posições negativas
    - Ordem otimizada de validações para "fail fast"
    - Logs de debug mais informativos com detalhes específicos de cada erro
- **Compatibilidade com NumPy/Pandas**: Conversão automática de tipos para JSON
    - Resolução de erros `TypeError: Object of type int64 is not JSON serializable`
    - Conversão explícita de `np.int64`, `np.float64` para tipos nativos Python
    - Aplicado em salvamento de fold IDs e holdout test IDs
- **Retorno Expandido de `cross_validate()`**: Agora retorna 4 valores
    - `(all_reports, summary_metrics, results, holdout_results)`
    - `holdout_results` contém métricas de todos os folds no conjunto holdout (None se não usado)


### Correções

- **Serialização JSON**: Correção de erros ao salvar IDs de folds e holdout test
- **Tipos de Dados**: Garantia de compatibilidade entre NumPy/Pandas e JSON padrão


### Testes

- **Cobertura de JSONL**: Novos testes para importação/exportação formato Doccano
- **Testes de Holdout**: Validação de separação e avaliação em conjunto holdout
- **Correção de Testes**: Ajustes para compatibilidade com novo retorno de 4 valores em `cross_validate()`

## v0.0.8

### Novas Funcionalidades

- **Suporte a CID (CID)**
    - Regex para localizar `CID`.
- **Cross-Validation no Treinamento**: Adicionado suporte completo a cross-validation no módulo `SeiAnonimizarNERTrainer`
    - Método `cross_validate()` para executar K-folds com estratificação opcional por tipo de entidade
    - Geração automática de relatórios por fold com métricas de avaliação
    - Suporte a paralelismo via `joblib` (parâmetro `n_jobs`)
    - Estratégias de folding: simples (KFold) ou estratificado por features/entidades
    - Salvamento automático de modelos e métricas por fold
- **Métodos Auxiliares para CV**: Novos métodos como `make_folds_by_id()`, `make_stratified_folds_by_id()` e `_prepare_fold_data()` para gerenciamento de folds baseados em IDs de documentos


### Melhorias Importantes

- **Integração com Avaliação**: Cross-validation agora integra diretamente com `SeiAnonimizarEvaluation` para métricas automáticas por fold
- **Pré-processamento de Dados**: Adicionado pré-processamento global de dados antes do CV para limpeza consistente
- **Parâmetros Flexíveis**: Novos parâmetros em `cross_validate()` para configuração de treinamento (`train_params`), avaliação (`eval_params`) e adição de dados (`add_data_params`)
- **Documentação**: Exemplos expandidos no código-fonte demonstrando workflow completo de CV


### Correções

- **Tratamento de IDs**: Melhorias no mapeamento de IDs entre textos e entidades durante o folding
- **Estratificação**: Correções para garantir distribuição equilibrada de entidades raras


### Testes

- **Cobertura Ampliada**: Novos testes para cenários de cross-validation, incluindo estratificação e paralelismo

## v0.0.7

### Novas Funcionalidades

- **Suporte a Coordenadas Geográficas (GEO_COORD) Expandido**
    - Regex aprimorados para diferentes formatos: graus, DMS, decimais com vírgula/ponto, UTM.
    - Contexto adicional para reduzir falsos positivos (palavras-chave como "GPS", "localização", "latitude", "longitude").
- **Novos Padrões de Passaporte**
    - Adicionado `add_pattern_passaporte_est()` para **passaportes estrangeiros**.
    - Suporte a formatos EUA, Reino Unido, Alemanha, França, Austrália, ICAO.
    - Novo parâmetro `foreign_passport=True` em `add_apply_patterns()`.
- **Suporte Ampliado a Dados Bancários**
    - Regex adicionais para IBAN e contas correntes com múltiplos formatos.
    - Melhoria em padrões de agência e conta (suporte mais robusto a hífens e prefixos).
- **Suporte a Telefone Expandido**
    - Regex adicionais para formatos nacionais/internacionais.
    - Inclusão de variantes com espaços, hífens, DDD de 2 ou 3 dígitos.
    - Melhor cobertura para WhatsApp e contatos internacionais.
- **Análises de Erro e Classificação (Evaluation)**
    - Novo método `get_classification_cases()` para listar casos **TP/FP/FN/TN individualmente**.
    - Novo método `get_error_analysis()` para estatísticas por entidade e exemplos de erros.
    - Novo método `save_classification_cases()` para exportar casos em Parquet/CSV/JSON.


### Melhorias Importantes

- **Validação de Entidades (verify_entities_unified)**:
    - Mais robusta para **RG** (palavras-chave expandidas, incluindo "identificação civil" e "RG digital").
    - Melhor distinção entre **SIAPE** e **SEI** (prevenção de falsos positivos).
    - Telefones agora exigem contexto ("telefone", "whatsapp"), reduzindo ruído.
    - Passaporte agora valida **comprimento** (6-9 chars) e busca contexto multilíngue.
- **Treinamento (SeiAnonimizarNERTrainer)**:
    - Novo método `debug_entities()` para validar offsets manualmente no log.
    - `clean_entities()` aprimorado: agora descarta entidades desalinhadas e não apenas as corrige.
    - Melhor alinhamento com esquema **BILUO** → menos erros silenciosos.
- **Avaliação (SeiAnonimizarEvaluation)**
    - Melhor consistência no `entity_mapping` (normalização de rótulos heterogêneos).
    - Comparação entre relatórios (`compare_reports`) trazendo percentuais de variação.
    - `evaluate_multiple_thresholds()` agora avalia múltiplos betas/overlaps num único DataFrame.
- **Testes automatizados**
    - Maior cobertura de testes usando o `pytest`


### Refatorações e Qualidade

- **Code Quality**
    - Uso consistente de `self.logger.debug/info/warning/exception`.
    - Melhor separação entre logs de debug e logs informativos.
    - Organização mais clara dos padrões regex em métodos isolados.
- **Documentação**
    - Exemplos de uso atualizados para novos contextos (Geo, Passaporte estrangeiro, CNH validada algorítmica).
    - Mais detalhes sobre casos de sobreposição e como funciona o `remove_overlap_positions()`.
- **CLI**
    - Agora aceita `--model_path` ou variável de ambiente `SPACY_MODEL_PATH`.
    - Impressão em JSON das entidades detectadas (indentação + UTF-8).


### Correções

- **CNPJs** corretamente tratados como falsos positivos em spans errados.
- **RG de estados específicos** (como AM/RR e MG) agora têm regex dedicados.
- Correção de bugs menores em `valida_cnh()` (cálculo DV).
- Correção em exportação JSON (`ensure_ascii=False`).

## v0.0.6-beta

### Novas Funcionalidades de Validação

- **Validadores Adicionais**: Novos validadores algorítmicos implementados
    - `valida_titulo_eleitor()`: Validação completa de Título de Eleitor conforme regras do TSE
    - `valida_cnh()`: Validação de CNH (Registro Nacional 11 dígitos e Espelho 10 dígitos)
    - `valida_cnpj()`: Validação completa de CNPJ com dígitos verificadores
- **Controles de Validação**: Novos parâmetros no construtor `SeiAnonimizar`
    - `use_titulo_validator`: Ativa/desativa validação de Título de Eleitor
    - `use_cnh_validator`: Ativa/desativa validação de CNH
- **Nova Entidade**: Suporte a coordenadas geográficas (`GEO_COORD`)
    - Detecção de coordenadas em múltiplos formatos (graus, decimais, DMS)
    - Validação contextual para evitar falsos positivos


### Melhorias no Módulo de Treinamento

- **Limpeza Automática de Dados**: Nova funcionalidade `clean_entities()`
    - Remoção automática de espaços nas extremidades de entidades
    - Validação de alinhamento com tokens spaCy
    - Prevenção de erro E024 durante treinamento
- **Validação de Entidades**: Novo método `validate_entities()`
    - Verificação completa de integridade das anotações
    - Validação de offsets, labels e esquema BILUO
- **Controles de Qualidade**: Novos parâmetros em `add_data()`
    - `auto_clean`: Ativa limpeza automática das entidades
    - `strict_clean`: Define se descarta exemplos com entidades removidas
- **Supressão de Warnings**: Filtro automático para warnings spaCy desnecessários


### Expansão de Padrões Regex

- **Coordenadas Geográficas**: Padrão `add_pattern_geo_coord()`
    - Suporte a formatos: 22°31.26' S, -20.308805°, Lat.22S233357
    - Validação contextual com palavras-chave relacionadas
- **Passaportes Estrangeiros**: Padrão `add_pattern_passaporte_est()`
    - Suporte a formatos internacionais (EUA, França, Reino Unido, etc.)
    - Parâmetro `foreign_passport` em `add_apply_patterns()`
- **Melhorias Contextuais**: Validação aprimorada para múltiplas entidades
    - RG: Lista expandida de palavras-chave de contexto
    - Passaporte: Detecção multilíngue (português/inglês)
    - CNPJ: Prevenção de conflitos com outras entidades


### Aprimoramentos no Módulo de Avaliação

- **Mapeamento Unificado**: Atualização do `entity_mapping`
    - CPF e FISTEL agora mapeados para "CPF/FISTEL"
    - Melhor consistência na avaliação de entidades relacionadas
- **Qualidade do Código**: Implementação de `strict=True` em operações zip()
    - Maior segurança na comparação de listas
    - Prevenção de bugs silenciosos em avaliações


### Melhorias Gerais de Qualidade

- **Code Quality**: Correções extensivas de linting
    - Remoção de imports não utilizados
    - Padronização de nomenclatura de variáveis
    - Melhor tratamento de exceções
- **Documentação Expandida**:
    - Docstrings aprimoradas com exemplos práticos
    - Melhor descrição de algoritmos de validação
    - Guias de uso para novas funcionalidades
- **Logging Aprimorado**: Mensagens de debug mais informativas
    - Melhor rastreamento de validações de entidades
    - Logs mais detalhados no processo de limpeza


### Correções e Otimizações

- **Validação de CNH**: Lógica aprimorada na detecção contextual
    - Melhor distinção entre CNH e CPF
    - Validação algorítmica integrada com detecção
- **Detecção de CNPJ**: Prevenção de falsos positivos
    - Filtro automático quando entidade não é rotulada como CNPJ
- **Performance**: Otimizações em validações contextuais
    - Busca mais eficiente por palavras-chave
    - Redução de processamento desnecessário


### Versionamento

- **Versão Atualizada**: Incremento para `__version__ = "0.6.0"`


## v0.0.5-beta

### Nova Arquitetura - Pipeline Completo de ML

- **Pacote Completo**: Transformação em pacote Python completo com `__init__.py` e exportação de classes principais
- **Pipeline de Machine Learning**: Introdução de workflow completo: treinamento → avaliação → uso em produção


### Novas Funcionalidades

- **Módulo de Treinamento**: Nova classe `SeiAnonimizarNERTrainer` para treinamento de modelos NER personalizados
    - Suporte a modelos base do spaCy ou criação de modelos em branco
    - Validação automática de dados de treinamento com esquema BILUO
    - Divisão automática de dados treino/validação
    - Tratamento flexível de erros (`raise`, `coerce`, `ignore`)
- **Módulo de Avaliação**: Nova classe `SeiAnonimizarEvaluation` para avaliação completa de modelos
    - Métricas detalhadas: F-beta, precisão, recall por tipo de entidade
    - Análise de sobreposição (overlap) configurável
    - Relatórios detalhados e sumarizados
    - Exportação para múltiplos formatos (Parquet, CSV, JSON)
    - Comparação entre versões de modelos
- **Sistema de Logging Avançado**: Logger personalizado opcional em todas as classes
    - Logs detalhados de inicialização, processamento e erros
    - Configuração flexível de níveis de log
    - Rastreamento de performance e debug


### Melhorias Significativas

- **Padrões Regex Aprimorados**:
    - Uso de lookahead/lookbehind negativos para maior precisão
    - Redução de falsos positivos com `(?<!\\d)` e `(?!\\d)`
    - Padrões mais robustos para telefone, dados bancários e CPF
- **Validação Unificada**: Método `verify_entities_unified()` consolida validação de entidades
    - Suporte tanto para objetos spaCy quanto dicionários
    - Lógica de validação mais consistente entre modelo e regex
- **Documentação Extensiva**:
    - Docstrings completas com exemplos práticos
    - Guia de uso do pipeline completo no `__init__.py`
    - Exemplos de integração entre treinamento, avaliação e uso
- **Tratamento de Erros Robusto**:
    - Try-catch abrangente com logging de exceções
    - Validação de entradas com mensagens de erro claras
    - Recuperação graceful de erros não-críticos


### Mudanças Arquiteturais

- **Estrutura de Pacote**: Migração de módulos individuais para pacote Python estruturado
- **Exportação de APIs**: `__all__` definido para controle de importações públicas
- **Versionamento**: Sistema de versionamento implementado (`__version__ = "0.4.0"`)


### Workflow de ML Integrado

- **Treinamento**:

```python
trainer = SeiAnonimizarNERTrainer(labels=["CPF", "EMAIL"])
trainer.add_data(dados)
trainer.train()
trainer.save_model("./modelo_treinado")
```

- **Avaliação**:

```python
evaluator = SeiAnonimizarEvaluation(texts_path="texts.parquet", ground_truth_path="gt.parquet")
evaluator.extract_predictions(anonymizer)
print(evaluator.evaluate_model())
```

- **Uso em Produção**:

```python
anonymizer = SeiAnonimizar("./modelo_treinado")
anonymizer.add_apply_patterns(use_model_labels=True)
```


### Correções e Otimizações

- **Detecção de Sobreposições**: Algoritmo otimizado para remoção de entidades sobrepostas
- **Validação de CPF**: Integração melhor com padrões regex
- **Performance**: Otimizações no processamento de grandes volumes de texto
- **Compatibilidade**: Melhor suporte para diferentes versões do spaCy


## v0.0.4-beta

### Novas Funcionalidades

- **Validação de CPF**: Adicionado método `valida_cpf()` para validação algorítmica de CPF com dígitos verificadores
- **Controle de Labels**: Novo parâmetro `labels` no construtor para especificar quais entidades o modelo deve detectar
- **Informações de Configuração**: Novo método `get_active_labels()` para verificar labels e padrões ativos
- **Validação Opcional de CPF**: Parâmetro `use_cpf_validator` para ativar/desativar validação de CPF nas entidades detectadas


### Melhorias

- **Documentação Extensiva**: Adicionada documentação completa com docstrings detalhadas e exemplos práticos para todos os métodos
- **Flexibilidade na Configuração**: Parâmetro `use_model_labels` em `add_apply_patterns()` para ativar detecção por modelo spaCy
- **Substituição de Padrões**: Parâmetro `replace_patterns` para limpar padrões existentes antes de adicionar novos
- **Método de Anonimização Atualizado**: `anonymize_text()` agora aceita lista de entidades como parâmetro
- **Detecção de Data Melhorada**: Padrões de data de nascimento mais robustos incluindo formatos com 2 ou 4 dígitos para ano


### Correções

- **Lógica de Sobreposição**: Correção na função `remove_overlap_positions()` para melhor tratamento de entidades sobrepostas
- **Validação de Entidades**: Melhorias na validação contextual para CPF, RG e outras entidades
- **Tratamento de Labels Customizados**: Melhor handling de labels não incluídos na lista de prioridades


### Documentação

- Exemplos detalhados de uso para todos os cenários principais
- Documentação de todos os formatos de retorno suportados
- Guias de configuração avançada
- Exemplos de padrões customizados


## v0.0.3-beta

### Novas Funcionalidades

- **Padrões Customizados**: Método `add_custom_pattern()` para adicionar padrões regex personalizados
- **Listagem de Padrões**: Método `list_patterns()` para visualizar padrões registrados (built-in vs custom)
- **Suporte a Padrões Customizados**: Parâmetro `custom_patterns` em `add_apply_patterns()`


### Melhorias

- **Flexibilidade em `add_apply_patterns()`**: Agora suporta padrões customizados além dos pré-definidos
- **Tratamento de Prioridades**: Melhor handling de labels customizados na função de remoção de sobreposições
- **Documentação**: Adicionados exemplos de uso básico na documentação do módulo


### Correções

- **Validação de Passaporte**: Correção na lógica de validação contextual para passaportes
- **Lógica de Sobreposição**: Ajustes na priorização de entidades sobrepostas


## v0.0.2-beta

### Remoções

- **Módulo LLM**: Remoção completa do arquivo `sei_anonimizar_llm.py` e funcionalidades relacionadas


### Refatoração Major

- **Arquitetura Simplificada**: Foco exclusivo em detecção via spaCy + regex, removendo dependência de LLM
- **Inicialização Simplificada**: Construtor agora requer apenas `model_path`
- **Método de Extração Unificado**: `extract_entities()` combina modelo spaCy e regex automaticamente


### Novas Funcionalidades

- **Novos Tipos de Retorno**: Adicionado `return_type="label_detail"` com informações de método de detecção
- **Detecção Híbrida**: Combinação automática de resultados do modelo spaCy e padrões regex
- **Validação Contextual**: Métodos `verify_entities()` e `verify_entities_regex()` para validação baseada em contexto
- **Remoção de Sobreposições**: Método `remove_overlap_positions()` para eliminar entidades duplicadas
- **Novos Padrões**: Adicionados padrões para endereços/CEP via `add_pattern_endereco()`


### Melhorias

- **Padrões Regex Aprimorados**: Padrões mais precisos e abrangentes para todas as entidades
- **Performance**: Processamento mais eficiente sem dependências de LLM
- **Flexibilidade**: Melhor controle sobre quais entidades detectar
- **Documentação**: Docstrings melhoradas com exemplos


### Correções

- **Detecção de CPF**: Lógica melhorada para distinguir entre CPF, FISTEL e CNH
- **Validação de Contexto**: Verificações mais rigorosas para evitar falsos positivos
- **Tratamento de Arquivos**: Melhor handling de arquivos .md


## v0.0.1-beta

### Release Inicial

- **Classe Base**: `SeiAnonimizar` para detecção de entidades via spaCy
- **Classe LLM**: `SeiAnonimizarLlm` para detecção via modelos de linguagem
- **Padrões Regex**: Suporte inicial para CPF, RG, CNH, SIAPE, dados bancários, email, telefone, etc.
- **Múltiplos Formatos**: Suporte para diferentes formatos de retorno
- **Anonimização**: Funcionalidade básica de substituição de entidades por tags
- **CLI**: Interface de linha de comando para ambas as classes


## Resumo de Evolução

| Versão | Principais Mudanças |
| :-- | :-- |
| **v0.0.1** | Release inicial com duas abordagens (spaCy + LLM) |
| **v0.0.2** | Remoção de LLM, foco em spaCy + regex, detecção híbrida |
| **v0.0.3** | Adição de padrões customizados e melhor organização |
| **v0.0.4** | Validação de CPF, controle granular de labels, documentação extensiva |
| **v0.0.5** | **Pipeline completo de ML**: treinamento + avaliação + produção, logging avançado |
| **v0.0.6** | **Validadores avançados**: Título de Eleitor, CNH, CNPJ + coordenadas geográficas, limpeza automática de dados |
| **v0.0.7** | Geo-coord, passaporte estrangeiro, novos regex bancários/telefone, error analysis avançado, aprimoramento de testes automáticos, novos exemplos |
| **v0.0.8** | CID, Cross-Validation: Suporte completo a K-folds com estratificação, paralelismo e integração com avaliação |
| **v0.0.9** | **Holdout test set** para métricas comparáveis, suporte completo a formato **JSONL/Doccano**, rastreabilidade de folds, validação `start >= end`, correção de serialização NumPy/JSON |
| **v1.0.0** | Arquitetura modular: Separação de constants em 5 submódulos, novos módulos (_training, _evaluation, _extraction, _patterns, _validators, _common, _anonymization), type hints obrigatórios, docstrings Google-style e 19 arquivos de teste |
| **v1.0.1** | Sprint 75: padrões automáticos por padrão, RNE/CRNM como RG, melhorias em EMAIL/RG/SIAPE/TITULO_ELEITOR, logging documentado e documentação atualizada |
| **v1.0.2** | Sprint 80: novas entidades PIS/PASEP/NIT, CNS e RESERVISTA com validação algorítmica e contextual, RG de conselhos profissionais (CRM/CRO/CREA/etc.), reorganização da suíte de testes em arquivos temáticos |
| **v1.0.3** | Normalização offset-safe, alinhamento do overlap por IoU, revisão de contratos e exemplos da API, documentação pdoc3 e declaração de dados fictícios |
| **v1.0.4** | Renomeação `SeiAnonimizar*` → `Anonimizar`/`Evaluation`/`Trainer`, módulos-ponte com `DeprecationWarning`, CLI extraída para `python -m anonimizar`, bridges legadas preservadas |

