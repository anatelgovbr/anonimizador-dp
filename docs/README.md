# Anonimizador-dp

O Anonimizador-dp é uma ferramenta completa para detecção e anonimização de dados pessoais em textos, com suporte a modelos spaCy e treinamento personalizado. O sistema combina técnicas de Machine Learning (NER) com padrões regex para identificar e substituir dados pessoais, como números de documentos pessoais, telefone, e-mail, data de nascimento, dados bancários, coordenadas geográficas e endereço.

> **Dados Fictícios:** todos os dados usados nos exemplos e no treinamento dos
> modelos NER distribuídos no projeto são fictícios. Eles preservam somente
> padrões estruturais existentes em documentos reais, sem reproduzir dados
> pessoais ou conteúdos reais. **Apenas parecem dados reais**, mas são
> totalmente fictícios e fora do contexto do conteúdo do texto de origem
> utilizado no treinamento dessa biblioteca.

## Índice

- [Anonimizador-dp](#anonimizador-dp)
  - [Entidades Suportadas](#entidades-suportadas)
  - [Como Funciona (Pipeline)](#como-funciona-pipeline)
  - [Arquitetura do Pacote](#arquitetura-do-pacote)
  - [Instalação](#instalação)
  - [Uso Básico](#uso-básico)
    - [Como Biblioteca Python](#como-biblioteca-python)
    - [Tipos de Retorno — Quando Usar Cada Um](#tipos-de-retorno---quando-usar-cada-um)
  - [Documentação dos Módulos](#documentação-dos-módulos)
    - [1. Módulo Anonimizar](#1-módulo-anonimizar)
    - [2. Módulo Trainer](#2-módulo-trainer)
    - [3. Módulo Evaluation](#3-módulo-evaluation)
  - [Workflow Completo](#workflow-completo)
    - [1. **Anonimização**](#1-anonimização)
    - [2. **Treinamento**](#2-treinamento)
    - [3. **Avaliação**](#3-avaliação)
  - [Exemplo Completo de Uso](#exemplo-completo-de-uso)
  - [Testes](#testes)
  - [Descrição Detalhada dos Padrões Suportados](#descrição-detalhada-dos-padrões-suportados)
  - [Configuração de Logging](#configuração-de-logging)
  - [Dicas e Resolução de Problemas](#dicas-e-resolução-de-problemas)

## Entidades Suportadas

| Rótulo | Descrição | Método de Detecção |
| :-- | :-- | :-- |
| CPF | Cadastro de Pessoa Física | REGEX + NER + **validação alg. (DV)** |
| RG | Registro Geral (inclui RNE/CRNM) | REGEX + NER + **validação estrutural** |
| CNH | Carteira Nacional de Habilitação | REGEX + NER + **validação alg. (DV)** |
| TITULO_ELEITOR | Título de Eleitor | REGEX + NER + **validação alg. (DV+UF)** |
| PASSAPORTE | Passaporte | REGEX + NER + **validação contextual** |
| SIAPE | Sistema Integrado de Administração de Pessoal | REGEX + NER + **validação contextual** |
| DATA_NASCIMENTO | Data de Nascimento | REGEX + NER + **validação contextual** |
| DADOS_BANCARIOS | Dados Bancários | REGEX + NER + **validação contextual** |
| EMAIL | Endereço de E-mail | REGEX + NER + **validação estrutural** |
| TELEFONE | Número de Telefone | REGEX + NER + **validação contextual** |
| ENDEREÇO | Endereço Postal / CEP | REGEX + NER + **validação contextual** |
| CID | Classificação Internacional de Doenças | REGEX + NER + **validação estrutural** |
| GEO_COORD | Coordenadas Geográficas | REGEX + NER + **validação estrutural** |
| PIS | PIS/PASEP/NIT | REGEX + NER + **validação alg. (DV)** |
| CNS | Cartão Nacional de Saúde | REGEX + NER + **validação alg. (DV+prefixo)** |
| RESERVISTA | Certificado de Reservista | REGEX + NER + **validação contextual** |

Legenda:
- **validação alg. (DV)** = algoritmo de dígitos verificadores
- **validação estrutural** = verificação de formato sem DV
- **validação contextual** = palavras-chave no entorno do texto

## Como Funciona (Pipeline)

O pipeline de detecção combina três fontes de extração em etapas sequenciais:

```
               Texto de entrada
                      │
                      ▼
    ┌─────────────────────────────┐
    │  1. Modelo spaCy NER        │  ← entidades via ML
    └──────────┬──────────────────┘
                      │
                      ▼
    ┌─────────────────────────────┐
    │  2. Padrões REGEX           │  ← padrões conhecidos
    └──────────┬──────────────────┘
                      │
                      ▼
    ┌─────────────────────────────┐
    │  3. Tabelas Markdown        │  ← colunas com dados pessoais
    └──────────┬──────────────────┘
                      │
                      ▼
    ┌─────────────────────────────┐
    │  4. União + Remoção de      │
    │     sobreposições           │  ← mescla e resolve conflitos
    └──────────┬──────────────────┘
                      │
                      ▼
    ┌─────────────────────────────┐
    │  5. Validadores             │  ← contexto + algoritmo (DV)
    └──────────┬──────────────────┘
                      │
                      ▼
               Lista final de entidades
```

Cada entidade detectada passa por um validador específico:
- **Validação algorítmica** — confere dígitos verificadores (CPF, CNH, TITULO_ELEITOR, PIS, CNS)
- **Validação estrutural** — verifica formato esperado (EMAIL, CID, GEO_COORD, RG)
- **Validação contextual** — busca palavras-chave no entorno (SIAPE, TELEFONE, PASSAPORTE, etc.)

## Arquitetura do Pacote

```
anonimizar/
├── __init__.py              # API pública (Anonimizar, Trainer, Evaluation)
├── _anonymization/
│   └── anonymizer.py        #   → classe Anonimizar + CLI main()
├── _extraction/
│   ├── pipeline.py          #   → orquestração modelo → regex → tabelas
│   ├── model.py             #   → inferência spaCy NER
│   ├── regex.py             #   → execução de padrões regex
│   └── markdown.py          #   → extração de tabelas markdown
├── _patterns/
│   ├── builtin.py           #   → 16 padrões regex pré-definidos
│   ├── custom.py            #   → padrões do usuário
│   └── registry.py          #   → gerenciamento de adders
├── _validators/
│   ├── context.py           #   → validadores contextuais (16 labels)
│   ├── documents.py         #   → validadores algorítmicos (CPF, CNH, etc.)
│   └── unified.py           #   → dispatcher único de validação
├── _training/
│   └── trainer_facade.py    #   → classe Trainer
├── _evaluation/
│   └── evaluation.py        #   → classe Evaluation
├── _normalization/          # normalização offset-safe de entidades
├── _constants/              # labels, prioridades, thresholds
├── _common/                 # logging, overlap
├── sei_anonimizar.py        # ponte legado (DeprecationWarning)
├── sei_anonimizar_evaluation.py
└── sei_anonimizar_treino.py
```

## Instalação

```bash
pip install anonimizador-dp
```

O nome do pacote no PyPI (`anonimizador-dp`) é diferente do nome do módulo de
import (`anonimizar`), como em `scikit-learn`/`sklearn`.


## Uso Básico

### Como Biblioteca Python

Fazer o import das bibliotecas necessárias:

```python
from anonimizar import Anonimizar
```

Instanciar o objeto do anonimizador:

```python
anonymizer = Anonimizar(model_path)
```

- O parâmetro `model_path` é **obrigatório** e deve apontar para um modelo spaCy treinado.
- Todos os demais parâmetros são opcionais.

Quando `auto_patterns=True` (padrão do construtor), os padrões built-in são aplicados automaticamente, incluindo RG estrangeiro (`RNE`/`CRNM`) como entidades com label `RG`. Para configurar padrões manualmente ou desativar esse comportamento, inicialize com `auto_patterns=False` e registre os rótulos desejados:

```python
anonymizer = Anonimizar(model_path, auto_patterns=False)
anonymizer.add_apply_patterns(['CPF', 'RG', 'EMAIL'], foreign_rg=False)
```

Caso deseje, pode adicionar padrões customizados:

```python
anonymizer.add_apply_patterns(['CPF', 'RG'], custom_patterns=[
    {
        "label": "MATRICULA_ALUNO",
        "regex": r"[A-Z]{2}\d{5}[A-Z]",
        "description": "Matrícula de aluno (2 letras + 5 números + 1 letra)"
    },
    {
        "label": "PLACA_CARRO",
        "regex": r"[A-Z]{3}-\d{4}",
        "description": "Placa de carro no formato AAA-9999"
    }
])
```

Utilizar o método `extract_entities` para extrair as entidades:

```python
entidades = anonymizer.extract_entities(
    text_or_path="<filename_or_text>", 
    return_type="label_position"  # ou "label_text", "label_detail"
)
```

- Apenas arquivos no formato `.md` são aceitos em `text_or_path`; para outros formatos, forneça o texto diretamente.
- Todos os parâmetros de `extract_entities` exceto `text_or_path` são opcionais.

Exemplo de saída com `return_type="label_detail"`:

```python
[{'label': 'CPF', 'start_position': 10, 'end_position': 24,
  'text': '123.456.789-09', 'detected_by': 'regex'}]
```

**Tipos de retorno — quando usar cada um:**

| Tipo | Cenário de Uso | O que Retorna |
|------|----------------|---------------|
| `label_position` | **Pré-processamento interativo** — exiba as marcações para o usuário revisar, corrigir ou complementar antes da anonimização definitiva. Ideal para sistemas de anotação humana (estilo FalaBR). | Lista de dicionários com `label`, `start_position` e `end_position`. |
| `label_text` | **Auditoria rápida** — quando você precisa apenas saber quais valores foram encontrados, sem coordenadas. | Lista com `label` e `text`. |
| `label_detail` | **Depuração e rastreabilidade** — mostra qual método detectou cada entidade. | `label`, `start_position`, `end_position`, `text` e `detected_by`. |

**Anonimizando Texto (processamento massivo):**

```python
# Para processamento em lote: substitui entidades automaticamente
# por tags como <|CPF|>, <|RG|>, etc. Sem intervenção humana.
entidades = anonymizer.extract_entities(texto)
texto_anonimizado = anonymizer.anonymize_text(texto, entidades)
print(texto_anonimizado)  # Ex: "Meu CPF é <|CPF|>."
```


## Documentação dos Módulos

Esta seção apresenta uma visão detalhada dos três módulos principais do sistema.

### 1. Módulo Anonimizar

**Descrição:** Módulo principal que fornece funcionalidades para detecção e anonimização de dados pessoais em textos, utilizando modelos spaCy treinados e padrões regex personalizáveis.

**Características Principais:**

- Combina detecção por **Machine Learning** (spaCy NER) e **padrões regex**
- Suporte a múltiplos tipos de dados pessoais
- Validação contextual de entidades detectadas
- Anonimização por substituição com tags formatadas

**Métodos Principais:**


| Método | Descrição |
| :-- | :-- |
| `extract_entities()` | Extrai entidades usando modelo e regex |
| `anonymize_text()` | Substitui entidades por tags de anonimização |
| `add_apply_patterns()` | Configura padrões de detecção |
| `add_custom_pattern()` | Adiciona padrões regex customizados |
| `get_active_labels()` | Retorna informações sobre labels ativos |
| `valida_cpf()` | Valida CPF com dígitos verificadores |

**Exemplo de Uso Básico:**

Consulte a seção [Uso Básico](#uso-básico) para um exemplo completo de
instanciação, extração de entidades e anonimização de texto.


### 2. Módulo Trainer

**Descrição:** Módulo especializado no treinamento de modelos NER (Named Entity Recognition) usando spaCy, com foco em anonimização de dados pessoais.

**Funcionalidades:**

- **Validação rigorosa** de dados com esquema BILUO
- **Múltiplos formatos** de entrada (dicionário, lista, DataFrame)
- **Estratégias flexíveis** de tratamento de erros
- **Logging detalhado** do processo de treinamento
- **Divisão automática** para treino e validação

**Labels Suportados por Padrão:**

```python
["CPF", "RG", "SIAPE", "ENDEREÇO", "TELEFONE", 
 "EMAIL", "DADOS_BANCARIOS", "CNH", 
 "PASSAPORTE", "TITULO_ELEITOR", "DATA_NASCIMENTO", "CID", "GEO_COORD",
 "PIS", "CNS", "RESERVISTA"]
```

- Todos os parâmetros de `Trainer()` (`model_name`, `output_dir`, `labels`) são opcionais; quando `labels` não é informado, a lista padrão acima é usada.

**Métodos Principais:**


| Método | Descrição |
| :-- | :-- |
| `add_data()` | Adiciona dados de treinamento com validação |
| `train()` | Executa treinamento iterativo do modelo |
| `save_model()` | Salva modelo treinado em disco |
| `split_data()` | Divide dados em treino e validação |
| `debug_entities()` | Função para debug de entidades |

**Estratégias de Tratamento de Erros:**

- **`raise`**: Lança exceção para erros
- **`coerce`**: Corrige/remove dados problemáticos
- **`ignore`**: Ignora erros silenciosamente

**Exemplo de Uso:**

```python
from anonimizar import Trainer

# Inicializar
trainer = Trainer(
    model_name="pt_core_news_sm",
    output_dir="./modelo_personalizado",
    labels=["CPF", "RG", "EMAIL"]
)

# Adicionar dados
dados = [{
    "text": "João Silva, CPF 123.456.789-00",
    "entities": [(12, 26, "CPF")]
}]
trainer.add_data(dados, errors='coerce')

# Treinar
trainer.train(n_iter=20, validation_split=0.2)
trainer.save_model()
```


### 3. Módulo Evaluation

**Descrição:** Módulo para avaliação completa de modelos NER especializados em anonimização, oferecendo métricas detalhadas e análise de erros.

**Funcionalidades Principais:**

- **Avaliação com múltiplos thresholds** de sobreposição
- **Métricas detalhadas** por tipo de entidade
- **Análise de erros** (TP, FP, FN, TN)
- **Comparação entre modelos** diferentes
- **Exportação de resultados** em múltiplos formatos

**Configurações:**

- **`overlap_threshold`**: Threshold mínimo para match (padrão: 0.8)
- **`beta`**: Valor para F-beta score (padrão: 2.0)
- **`entity_mapping`**: Normalização de entidades

**Métodos Principais:**


| Método | Descrição |
| :-- | :-- |
| `extract_predictions()` | Extrai predições do modelo |
| `evaluate_model()` | Executa avaliação completa |
| `get_detailed_report()` | Gera relatório detalhado |
| `get_error_analysis()` | Análise de erros por tipo |
| `compare_reports()` | Compara diferentes relatórios |
| `export_results()` | Exporta resultados |

**Métricas Calculadas:**

- **F-beta Score, Precision, Recall**
- **True Positives (TP), False Positives (FP)**
- **False Negatives (FN), True Negatives (TN)**
- **Distribuição de sobreposição**

**Exemplo de Uso:**

```python
from anonimizar import Evaluation

# Inicializar
evaluator = Evaluation(
    texts_path="texts.parquet",
    ground_truth_path="gt.parquet",
    overlap_threshold=0.8,
    beta=2.0
)
```

- Todos os parâmetros de `Evaluation()` são opcionais; os dados também podem ser carregados posteriormente via `load_data()`.


## Workflow Completo

### 1. **Anonimização**

```python
# Carregar modelo com padrões built-in automáticos
anonymizer = Anonimizar("modelo_treinado")

# Processar textos
entities = anonymizer.extract_entities(texto)
texto_anonimo = anonymizer.anonymize_text(texto, entities)
```


### 2. **Treinamento**

```python
# Preparar dados e treinar
trainer = Trainer()
trainer.add_data(dados_treinamento)
trainer.train(n_iter=30)
trainer.save_model()
```


### 3. **Avaliação**

```python
# Avaliar performance
evaluator = Evaluation()
evaluator.load_data(textos, ground_truth)
results = evaluator.evaluate_model()
```


## Exemplo Completo de Uso

```python
# EXEMPLO COMPLETO: TREINAMENTO, AVALIAÇÃO E APLICAÇÃO
from anonimizar import (
    Trainer,
    Evaluation,
    Anonimizar,
)

# Atribuindo os objetos
trainer = Trainer(output_dir='./treinamento_ner')
evaluator = Evaluation(overlap_threshold=0.8, beta=2.0)
anonymizer = Anonimizar('./treinamento_ner/')

# Gerando dados de treinamento
data_list = [
    {
        'text': "Exemplo de texto com CPF 123.456.789-09.",
        'entities': [(25, 39, 'CPF')]
    },
    {
        'text': "Outro exemplo de texto com CPF 123.456.789-09.",
        'entities': [(31, 45, 'CPF')]
    },
    {
        'text': "Exemplo de texto com CPF 123.456.789-09.",
        'entities': [(25, 50, 'CPF')]
    },
    {
        "text": "João Silva, CPF 123.456.789-09, mora na Rua das Flores, 123.",
        "entities": [(16, 30, "CPF"), (40, 59, "ENDEREÇO")]
    },
    {
        "text": "Email de contato: joao@empresa.com.br, seu CPF é 123.456.789-09",
        "entities": [(18, 37, "EMAIL"), (49, 63, 'CPF')]
    },
    {
        "text": "Documento sem entidades para teste.",
        "entities": []
    }
]

trainer.add_data(data_list, errors='coerce')

# Treino
try:
    trainer.train(
        n_iter=30,
        drop=0.2,
        batch_size=4,
        validation_split=0.2
    )

    trainer.save_model()

    print("Treinamento concluído com sucesso!")
    print(f"Total de exemplos de treinamento: {len(trainer.training_data)}")

except Exception as e:
    print(f"Erro durante treinamento: {e}")

# Validação
df_texts, df_gt = trainer.val_data_to_evaluation()

evaluator.load_data(df_texts, df_gt)
preds = evaluator.extract_predictions(anonymizer)
results = evaluator.evaluate_model()
print(evaluator.get_summary_report())

# Aplicando em novos casos
texto_exemplo = 'Oi, meu nome é Matheus, meu email é exemplo@gmail.com, e meu CPF é 123.456.789-09'

entidades = anonymizer.extract_entities(texto_exemplo)
texto_anonimizado = anonymizer.anonymize_text(texto_exemplo, entidades)

print(f"Texto original: {texto_exemplo}")
print(f"Texto anonimizado: {texto_anonimizado}")
```


## Testes

Para executar os testes use o comando:

```bash
pytest --verbose
```

Também é possível filtrar os testes utilizando os markers listados no arquivo `pytest.ini`:

```bash
pytest --verbose -m <marker>
```

## Descrição Detalhada dos Padrões Suportados

### Dados Pessoais
- **CPF**: Cadastro de Pessoa Física em múltiplos formatos (com pontos/hífens, contíguo, com asteriscos). `FISTEL` também é aceito como label de compatibilidade e ativa os padrões de CPF.
- **RG**: Registro Geral com suporte a diferentes estados brasileiros (SP, MG, AM/RR, Ceará, etc.) e, por padrão com `auto_patterns=True`, RG estrangeiro (`RNE`/`CRNM`) retornado com label `RG`. Registros de conselhos profissionais (`CRM`, `CRO`, `CRP`, `CREA`, `CRF`, `CRESS`, `CRECI`, `COREN`, `CRMV`) também são reconhecidos como `RG`.
- **DATA_NASCIMENTO**: Data de nascimento em diversos formatos (DD/MM/YYYY, DD-MM-YYYY, etc.)
- **PASSAPORTE**: Passaporte brasileiro e estrangeiro
- **PIS**: PIS/PASEP/NIT — número de inscrição do trabalhador em programas sociais

### Identificação Profissional
- **SIAPE**: Sistema Integrado de Administração de Pessoal (servidor público federal)
- **CNH**: Carteira Nacional de Habilitação
- **RESERVISTA**: Certificado de Reservista

### Dados de Contato
- **EMAIL**: Endereços de e-mail padrão
- **TELEFONE**: Números de telefone brasileiros (com DDD e formato)

### Localização
- **ENDEREÇO**: Endereços postais e CEPs brasileiros (múltiplos formatos)
- **GEO_COORD**: Coordenadas geográficas (latitude/longitude em vários formatos)

### Documentos Especiais
- **TITULO_ELEITOR**: Título de Eleitor

### Dados Financeiros
- **DADOS_BANCARIOS**: Números de conta, agência e dados bancários

### Saúde
- **CID**: Classificação Internacional de Doenças (códigos CID-10/CID-11)
- **CNS**: Cartão Nacional de Saúde (SUS)

## Configuração de Logging

O pacote usa o módulo padrão `logging` do Python. Por padrão, nenhum handler é configurado — o comportamento fica a cargo da aplicação que usa a biblioteca.

### Níveis disponíveis

| Nível | O que é registrado |
| :-- | :-- |
| `WARNING` / `ERROR` | Apenas erros e avisos críticos (falhas de validação, exceções) |
| `INFO` | Inicialização do modelo, início/fim de treinamento, métricas de fold |
| `DEBUG` | Tudo acima + logs por documento e por entidade (útil para depuração) |

### Uso operacional (lote)

Para silenciar mensagens rotineiras em processamento de muitos documentos, configure o nível como `WARNING`:

```python
import logging
logging.getLogger("anonimizar").setLevel(logging.WARNING)  # Apenas avisos e erros
```

Ou `INFO` para manter mensagens de inicialização e progresso de treinamento:

```python
import logging
logging.getLogger("anonimizar").setLevel(logging.INFO)
```

### Depuração detalhada

Para inspecionar o que ocorre em cada documento ou entidade:

```python
import logging

logger = logging.getLogger("anonimizar")
logger.setLevel(logging.DEBUG)

# Adicionar handler para ver a saída no console
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
logger.addHandler(handler)
```

## Dicas e Resolução de Problemas

- **Erro ao instanciar `Anonimizar`:** Certifique-se de passar o caminho correto para um modelo spaCy treinado no parâmetro `model_path`.
- **Arquivo não suportado:** Apenas arquivos `.md` são aceitos; para outros formatos, forneça o texto diretamente.
- **Entidade não detectada:** Verifique se o label está no modelo, se `auto_patterns` está ativo ou se o padrão correspondente foi adicionado via `add_apply_patterns` em configuração manual.
- **Problemas de performance:** Para textos muito longos, considere processar em blocos menores.
- **Validação de dados:** Use a estratégia `errors='coerce'` no treinamento para tratamento automático de dados problemáticos.
- **Logs excessivos em lote:** Configure `logging.getLogger("anonimizar").setLevel(logging.WARNING)` antes de processar grandes volumes.

