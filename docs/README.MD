# SEI Anonimizar

O SEI Anonimizar é uma ferramenta completa para detecção e anonimização de entidades sensíveis em textos, com suporte a modelos spaCy e treinamento personalizado. O sistema combina técnicas de Machine Learning (NER) com padrões regex para identificar e substituir informações sensíveis como:

## Entidades Suportadas

| Rótulo | Descrição |
| :-- | :-- |
| CPF | Cadastro de Pessoa Física |
| RG | Registro Geral |
| CNH | Carteira Nacional de Habilitação |
| TITULO_ELEITOR | Título de Eleitor |
| PASSAPORTE | Passaporte |
| SIAPE | Matrícula SIAPE |
| DATA_NASCIMENTO | Data de Nascimento |
| DADOS_BANCARIOS | Dados Bancários |
| EMAIL | Endereço de E-mail |
| TELEFONE | Número de Telefone |
| ENDEREÇO | Endereço Postal |

## Índice

- [SEI Anonimizar](#sei-anonimizar)
  - [Entidades Suportadas](#entidades-suportadas)
  - [Índice](#índice)
  - [Instalação](#instalação)
  - [Uso Básico](#uso-básico)
    - [Linha de Comando](#linha-de-comando)
    - [Como Biblioteca Python](#como-biblioteca-python)
  - [Documentação dos Módulos](#documentação-dos-módulos)
    - [1. Módulo SeiAnonimizar](#1-módulo-seianonimizar)
    - [2. Módulo SeiAnonimizarNERTrainer](#2-módulo-seianonimizarnertrainer)
    - [3. Módulo SeiAnonimizarEvaluation](#3-módulo-seianonimizarevaluation)
  - [Workflow Completo](#workflow-completo)
    - [1. **Anonimização**](#1-anonimização)
    - [2. **Treinamento**](#2-treinamento)
    - [3. **Avaliação**](#3-avaliação)
  - [Exemplo Completo de Uso](#exemplo-completo-de-uso)
  - [Testes](#testes)
  - [Dicas e Resolução de Problemas](#dicas-e-resolução-de-problemas)


## Instalação

```bash
pip install git+https://git.anatel.gov.br/processo_eletronico/sei-ia/sei-anonimizar.git@v0.0.2-beta
```


## Uso Básico

### Linha de Comando

Executar o comando abaixo para obter as entidades de um arquivo ou texto direto:

Inicialmente defina a sua variável de ambiente `SPACY_MODEL_PATH`.

```bash
python -m anonimizar.sei_anonimizar --text_or_path="<filename_or_text>"
```

**Parâmetros disponíveis:**

- `--text_or_path`: Caminho para o arquivo .md ou texto direto.
- `--return_type`: (opcional) Formato de retorno: label_position (padrão), label_text ou label_detail.
- `--model_path`: (opcional) Caminho para o modelo treinado, caso esteja em branco irá buscar da variável de ambiente.


### Como Biblioteca Python

Fazer o import das bibliotecas necessárias:

```python
from anonimizar import SeiAnonimizar
```

Instanciar o objeto do anonimizador:

```python
anonymizer = SeiAnonimizar(model_path)
```

- O parâmetro model_path é obrigatório e deve apontar para um modelo spaCy treinado.

Adicionar os padrões das variáveis de interesse:

```python
anonymizer.add_apply_patterns(['CPF', 'RG', 'CNH', 'TITULO_ELEITOR', 
                               'PASSAPORTE', 'SIAPE', 'DATA_NASCIMENTO', 'FISTEL', 
                               'DADOS_BANCARIOS', 'EMAIL', 'TELEFONE', 'ENDERECO'])
```

Caso deseje, pode adicionar padrões customizados:

```python
anon.add_apply_patterns(['CPF', 'RG'], custom_patterns=[
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
anonymizer.extract_entities(
    text_or_path="<filename_or_text>", 
    return_type="label_position"  # ou "label_text", "label_detail"
)
```

**Tipos de retorno:**

- `label_position`: retorna lista de dicionários com rótulo e posições.
- `label_text`: retorna rótulo e texto encontrado.
- `label_detail`: retorna rótulo, posições, texto e método de detecção.

**Anonimizando Texto:**

```python
anon_text = anonymizer.anonymize_text("<texto>")
print(anon_text)
```


## Documentação dos Módulos

Esta seção apresenta uma visão detalhada dos três módulos principais do sistema.

Para mais informções, consulte a documentação completa em html [SEI-Anonimizar](./anonimizar/index.html).

### 1. Módulo SeiAnonimizar

**Descrição:** Módulo principal que fornece funcionalidades para detecção e anonimização de entidades sensíveis em textos, utilizando modelos spaCy treinados e padrões regex personalizáveis.

**Características Principais:**

- Combina detecção por **Machine Learning** (spaCy NER) e **padrões regex**
- Suporte a múltiplos tipos de entidades sensíveis
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

```python
from anonimizar import SeiAnonimizar

# Inicializar
anonymizer = SeiAnonimizar(model_path="pt_core_news_lg")

# Configurar padrões
anonymizer.add_apply_patterns(['CPF', 'EMAIL', 'TELEFONE'])

# Processar texto
texto = "Meu CPF é 123.456.789-00 e meu e-mail é exemplo@dominio.com"
entidades = anonymizer.extract_entities(texto, return_type="label_detail")
texto_anonimizado = anonymizer.anonymize_text(texto, entidades)
```


### 2. Módulo SeiAnonimizarNERTrainer

**Descrição:** Módulo especializado no treinamento de modelos NER (Named Entity Recognition) usando spaCy, com foco em anonimização de dados sensíveis.

**Funcionalidades:**

- **Validação rigorosa** de dados com esquema BILUO
- **Múltiplos formatos** de entrada (dicionário, lista, DataFrame)
- **Estratégias flexíveis** de tratamento de erros
- **Logging detalhado** do processo de treinamento
- **Divisão automática** para treino e validação

**Labels Suportados por Padrão:**

```python
["CPF", "RG", "SIAPE", "ENDEREÇO", "TELEFONE", 
 "EMAIL", "FISTEL", "DADOS_BANCARIOS", "CNH", 
 "PASSAPORTE", "TITULO_ELEITOR", "DATA_NASCIMENTO"]
```

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
from anonimizar import SeiAnonimizarNERTrainer

# Inicializar
trainer = SeiAnonimizarNERTrainer(
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


### 3. Módulo SeiAnonimizarEvaluation

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
from anonimizar import SeiAnonimizarEvaluation

# Inicializar
evaluator = SeiAnonimizarEvaluation(
    texts_path="texts.parquet",
    ground_truth_path="gt.parquet",
    overlap_threshold=0.8,
    beta=2.0
)

# Extrair predições e avaliar
predictions = evaluator.extract_predictions(anonymizer)
results = evaluator.evaluate_model()

# Obter relatórios
summary = evaluator.get_summary_report()
detailed = evaluator.get_detailed_report()
```


## Workflow Completo

### 1. **Anonimização**

```python
# Carregar modelo e configurar padrões
anonymizer = SeiAnonimizar("modelo_treinado")
anonymizer.add_apply_patterns(['CPF', 'RG', 'EMAIL'])

# Processar textos
entities = anonymizer.extract_entities(texto)
texto_anonimo = anonymizer.anonymize_text(texto, entities)
```


### 2. **Treinamento**

```python
# Preparar dados e treinar
trainer = SeiAnonimizarNERTrainer()
trainer.add_data(dados_treinamento)
trainer.train(n_iter=30)
trainer.save_model()
```


### 3. **Avaliação**

```python
# Avaliar performance
evaluator = SeiAnonimizarEvaluation()
evaluator.load_data(textos, ground_truth)
results = evaluator.evaluate_model(anonymizer)
```


## Exemplo Completo de Uso

```python
# EXEMPLO COMPLETO: TREINAMENTO, AVALIAÇÃO E APLICAÇÃO
from anonimizar import (
    SeiAnonimizarNERTrainer,
    SeiAnonimizarEvaluation,
    SeiAnonimizar,
)

# Atribuindo os objetos
trainer = SeiAnonimizarNERTrainer(output_dir='./treinamento_ner')
evaluator = SeiAnonimizarEvaluation(overlap_threshold=0.8, beta=2.0)
anonymizer = SeiAnonimizar('./treinamento_ner/')
anonymizer.add_apply_patterns(use_model_labels=True)

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

## Dicas e Resolução de Problemas

- **Erro ao instanciar `SeiAnonimizar`:** Certifique-se de passar o caminho correto para um modelo spaCy treinado no parâmetro `model_path`.
- **Arquivo não suportado:** Apenas arquivos `.md` são aceitos; para outros formatos, forneça o texto diretamente.
- **Entidade não detectada:** Verifique se o padrão correspondente foi adicionado via `add_apply_patterns`.
- **Problemas de performance:** Para textos muito longos, considere processar em blocos menores.
- **Validação de dados:** Use a estratégia `errors='coerce'` no treinamento para tratamento automático de dados problemáticos.

