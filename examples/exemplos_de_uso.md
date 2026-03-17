# Exemplos de uso — sei-anonimizar

Abaixo está um guia de uso simples, avaliação e treino.

## 1 Uso simples (carregar modelo, extrair entidades, anonimizar, regex custom)

### 1.1 Import e inicialização

```python
from sei_anonimizar import SeiAnonimizar

model_path = "PATH_Modelo_TREINADO"
anon = SeiAnonimizar(model_path=model_path)
```


### 1.2 Exemplo com CPF inválido (entidade não detectada para CPF)

Texto de entrada:

```python
texto_original = """
Nome: João Silva
CPF: 123.456.789-00
RG: 12.345.678-9
Telefone: (61) 9999-8888
"""
```

Extração de entidades:

```python
entidades = anon.extract_entities(texto_original)
print(entidades)
# Saída esperada:
# [{'label': 'RG', 'start_position': 42, 'end_position': 54},
#  {'label': 'TELEFONE', 'start_position': 65, 'end_position': 79}]
```

Anonimização:

```python
texto_anonimizado = anon.anonymize_text(texto_original, entidades)
print(texto_anonimizado)
# Saída esperada:
# Nome: João Silva
# CPF: 123.456.789-00
# RG: <|RG|>
# Telefone: <|TELEFONE|>
#
```

Nota: O CPF inválido (terminado em 00) não foi mascarado pois a validação de CPF está ativa.

### 1.3 Exemplo com CPF válido (entidade detectada para CPF)

```python
texto_original = """
Nome: João Sauro
CPF: 123.456.789-09
"""

entidades = anon.extract_entities(texto_original)
print(entidades)
# [{'label': 'CPF', 'start_position': 23, 'end_position': 37}]

texto_anonimizado = anon.anonymize_text(texto_original, entidades)
print(texto_anonimizado)
# Nome: João Sauro
# CPF: <|CPF|>
#
```


### 1.4 Adicionando regex customizada (ex.: PLACA ABC-1234)

```python
anon.add_custom_pattern(
    label="PLACA",
    regex_pattern=r"[A-Z]{3}-\d{4}",
    description="Placa de carro com 3 letras, traço, 4 números"
)

texto_original = "O suspeito tem um carro com a placa UHU-0420, um celta preto com um farol queimado."

entidades = anon.extract_entities(text_or_path=texto_original, return_type="label_position")
print(entidades)
# [{'label': 'PLACA', 'start_position': 36, 'end_position': 44}]

texto_anonimizado = anon.anonymize_text(texto_original, entidades)
print(texto_anonimizado)
# O suspeito tem um carro com a placa <|PLACA|>, um celta preto com um farol queimado.
```


***

## 2 Avaliação (SeiAnonimizarEvaluation)

### 2.1 Setup (modelo, regexs e evaluator)

```python
import pandas as pd
from sei_anonimizar_evaluation import SeiAnonimizarEvaluation
from sei_anonimizar import SeiAnonimizar

model_path = "PATH_modelo_teste"
anon = SeiAnonimizar(model_path=model_path, labels=['CPF', 'PLACA'])

# Aplica padrões internos (ex.: CPF)
anon.add_apply_patterns(['CPF'])

# Adiciona regex custom para PLACA
anon.add_custom_pattern(
    label="PLACA",
    regex_pattern=r"[A-Z]{3}-\d{4}",
    description="Placa de carro com 3 letras, traço, 4 números"
)

# Instancia o avaliador
evaluator = SeiAnonimizarEvaluation(
    texts_path="PATH_textos.parquet",
    ground_truth_path="PATH_entidades.parquet"
)
```


### 2.2 Extração de predições e avaliação

```python
predictions = evaluator.extract_predictions(
    anon,
    save_path="PATH_predictions_model_plus_regex.parquet"
)

results = evaluator.evaluate_model()

current_report = evaluator.get_detailed_report()
previous_report = pd.read_parquet(
    "PATH_results_only_model.parquet"
)

comparison = evaluator.compare_reports(current_report, previous_report)
```

Exemplo de comparação agregada esperada (resumo):

- CPF: Melhorou (de 0 TP para 2 TP; FN de 2 para 0)
- PLACA: Melhorou (de 2 TP/2 FP para 4 TP/0 FP)
- TODAS: Melhorou (de 2 TP/2 FP/5 FN para 6 TP/0 FP/1 FN)

Exemplo de linhas detalhadas esperadas (amostra):

- CPF 123.456.789-09: TP (overlap=1.0)
- PLACA ERI-1007: TP (overlap=1.0)
- EMAIL duvidas@exemplo.com.br: FN (não detectado)

***

## 3 Treino (SeiAnonimizarNERTrainer)

### 3.1 Setup do trainer

```python
import pandas as pd
from sei_anonimizar_treino import SeiAnonimizarNERTrainer
from sei_anonimizar import SeiAnonimizar

trainer = SeiAnonimizarNERTrainer(
    output_dir="PATH_modelo_teste",
    labels=['CPF', 'PLACA']
)
```


### 3.2 Debug de entidades (checagem de offsets/BILUO)

```python
trainer.debug_entities(
    "Erivan, cujo documento é 123.456.789-09, tem um fusca azul com a placa ERI-1007.",
    [(25,41,'CPF'), (71,80,'PLACA')]
)
# Observações típicas:
# - CPF com final de offset incorreto (inclui vírgula/espaço)
# - PLACA inclui o ponto final (.) no offset
```


### 3.3 Inclusão de exemplo inválido (erros esperados)

```python
invalid_data = {
    'text': "Erivan, cujo documento é 123.456.789-09, tem um fusca azul com a placa ERI-1007.",
    'entities': [(25,41,'CPF')]  # fim incorreto
}

try:
    trainer.add_data(invalid_data)
except Exception as e:
    print(e)
# Esperado: erro BILUO/offset desalinhado; dado não adicionado
```


### 3.4 Inclusão de exemplo válido

```python
valid_data = {
    'text': "Erivan, cujo documento é 123.456.789-09, tem um fusca azul com a placa ERI-1007.",
    'entities': [(25,39,'CPF')]  # ajustado ao token exato
}

trainer.add_data(valid_data)
trainer.training_data
# Esperado: exemplo aparece com entities [(25,39,'CPF')]
```


### 3.5 Mistura de válidos e inválidos com controle de errors

Dados de exemplo:

```python
mix_data = {
    'text': "Erivan, cujo documento é 123.456.789-09, tem um fusca azul com a placa ERI-1007.",
    'entities': [(25,39,'CPF'), (72,85,'PLACA')]  # PLACA fora do tamanho do texto
}
```

- errors='raise' (lança erro e não adiciona nada)

```python
trainer.training_data = []
try:
    trainer.add_data(mix_data)
except Exception as e:
    print(e)
# Esperado: "Posições inválidas: 72-85 em texto de tamanho 80"
```

- errors='ignore' (ignora casos inválidos e mantém os válidos)

```python
trainer.training_data = []
trainer.add_data(mix_data, errors='ignore')
trainer.training_data
# Esperado: mantém apenas CPF [(25,39,'CPF')]
```

- errors='coerce' (tenta ajustar; se não der, ignora o inválido)

```python
trainer.training_data = []
trainer.add_data(mix_data, errors='coerce')
trainer.training_data
# Esperado: mantém CPF; PLACA inválida é ignorada com warning
```


### 3.5.1 Parâmetro auto_clean (limpeza automática)

O parâmetro `auto_clean` controla se a limpeza automática de entidades é aplicada:

```python
# auto_clean=True (padrão) - aplica limpeza automática
trainer.add_data(data, auto_clean=True)
# - Remove espaços extras nos limites das entidades
# - Valida alinhamento BILUO (offsets devem corresponder a tokens spaCy)
# - Corrige offsets quando possível

# auto_clean=False - sem limpeza automática
trainer.add_data(data, auto_clean=False)
# - Apenas valida entidades sem aplicar correções
# - Mais rápido, mas não corrige problemas comuns
```

**Tabela de combinações:**

| errors  | auto_clean | Comportamento                                      |
|---------|------------|---------------------------------------------------|
| 'raise' | False      | Levanta erro em entidades inválidas               |
| 'raise' | True       | Corrige espaços, levanta erro se persistir        |
| 'coerce'| False      | Descarta entidades inválidas                     |
| 'coerce'| True       | Corrige espaços, descarta BILUO inválido         |
| 'ignore'| False      | Mantém entidades como estão                      |
| 'ignore'| True       | Mantém entidades como estão (sem correção)       |

**Exemplos:**

```python
# errors='ignore' + auto_clean=True - mantém entidades como estão (sem correção)
data = [{"text": "CPF: 123.456.789-00  ", "entities": [(5, 21, "CPF")]}]
trainer.add_data(data, errors="ignore", auto_clean=True)
# Entidades são mantidas com espaços

# errors='coerce' + auto_clean=True - corrige espaços
data = [{"text": "CPF: 123.456.789-00  ", "entities": [(5, 21, "CPF")]}]
trainer.add_data(data, errors="coerce", auto_clean=True)
# Entidades são corrigidas: (5, 21) -> (5, 19)

# errors='raise' + auto_clean=True - levanta erro se não conseguir corrigir
data = [{"text": "EMAIL: teste@exemplo.com", "entities": [(14, 21, "EMAIL")]}]
trainer.add_data(data, errors="raise", auto_clean=True)
# ValueError: Offsets desalinhados aos tokens (BILUO inválido)
```


### 3.6 Adição via DataFrame — 1 linha por documento

```python
trainer.training_data = []

_por_documento = [
    {
        'text': "Erivan, cujo documento é 123.456.789-09, tem um fusca azul com a placa ERI-1007.",
        'entities': [(25,39,'CPF'), (71,79,'PLACA')]  # nota: PLACA ajustada sem ponto final
    },
    {
        'text': "O suspeito tem um carro com a placa UHU-0420, um celta preto com um farol queimado.",
        'entities': [(36,44,'PLACA')]
    },
    {
        'text': "João, cujo documento é 123.456.789-09, tem um fusca azul com a placa JOA-1337.",
        'entities': [(25,41,'CPF'), (72,85,'PLACA')]  # CPF e PLACA com offsets inválidos
    },
    {
        'text': "O suspeito tem um carro com a placa TTT-0690, um celta preto com um farol queimado.",
        'entities': [(34,44,'PLACA')]  # possível desalinhamento
    },
    {
        "text": "Qualquer dívida enviar para duvidas@exemplo.com.br",
        "entities": [(28, 50, "EMAIL")]  # label não listada no trainer
    },
    {
        "text": "The book is on the table.",
        "entities": []
    }
]

df_documento = pd.DataFrame(_por_documento)
trainer.add_data(df_documento, errors='ignore', keep_empty_entities=False)
trainer.training_data
# Esperado: 3 exemplos mantidos (CPF/PLACA válidos e texto sem entidade opcionalmente excluído se keep_empty_entities=False)
```

Observações típicas ao adicionar:

- Textos com entidades desalinhadas são descartados.
- Labels fora da lista (ex.: EMAIL) são ignoradas e o texto pode ser descartado se ficar sem entidades.
- Ajuste os offsets para não incluir espaços/pontuação.


### 3.7 Adição via DataFrame — 1 linha por entidade

Estrutura do DataFrame:

```python
_por_entidade = {
    'texto': [
        "Erivan, cujo documento é 123.456.789-09, tem um fusca azul com a placa ERI-1007.",
        "Erivan, cujo documento é 123.456.789-09, tem um fusca azul com a placa ERI-1007.",
        "O suspeito tem um carro com a placa UHU-0420, um celta preto com um farol queimado.",
        "João, cujo documento é 123.456.789-09, tem um fusca azul com a placa JOA-1337.",
        "João, cujo documento é 123.456.789-09, tem um fusca azul com a placa JOA-1337.",
        "O suspeito tem um carro com a placa TTT-0690, um celta preto com um farol queimado.",
        "Qualquer dívida enviar para duvidas@exemplo.com.br"
    ],
    'start': ,
    'end':   ,
    'entidade': ['CPF','PLACA','PLACA','CPF','PLACA','PLACA','EMAIL']
}

df_entidade = pd.DataFrame(_por_entidade)

# Exemplo de uso típico (a função do trainer deve consolidar por texto):
trainer.training_data = []
trainer.add_data(df_entidade, errors='ignore', keep_empty_entities=False)
trainer.training_data
# Ajuste offsets e remova labels fora da configuração para maximizar aproveitamento.
```

Dicas:

- Offsets devem cobrir exatamente o span da entidade, sem espaços/pontuação adicionais.
- Garanta que todas as labels usadas estejam na lista passada ao trainer (labels=['CPF','PLACA'] neste exemplo).
- Quando um texto tiver múltiplas linhas no DF (uma por entidade), o método deve consolidar corretamente as entidades do mesmo texto.

***

## 3.8 Importar/Exportar formato JSONL (Doccano)

### 3.8.1 Carregar dados de arquivo JSONL

```python
# Carregar dados anotados do Doccano
trainer = SeiAnonimizarNERTrainer(labels=['CPF', 'EMAIL', 'TELEFONE'])

# Adicionar dados direto do arquivo JSONL
trainer.add_data("./anotacoes_doccano.jsonl", errors='coerce')

print(f"Dados carregados: {len(trainer.training_data)} exemplos")
```


### 3.8.2 Exportar dados de treinamento para JSONL

```python
# Após adicionar dados via DataFrame ou dict
trainer.save_to_doccano_jsonl("./dados_treinamento.jsonl")
print("Dados exportados para formato Doccano!")
```


### 3.8.3 Converter JSONL em DataFrames

```python
# Útil para análise ou pré-processamento
df_textos, df_entidades = trainer._load_jsonl_to_dataframes("./anotacoes.jsonl")

print(f"Textos: {len(df_textos)}")
print(f"Entidades: {len(df_entidades)}")
print(df_entidades.head())
# Colunas: id, start, end, entidade
```


***

## 3.9 Cross-Validation com Holdout Test

### 3.9.1 Cross-Validation básico (K-folds simples)

```python
import pandas as pd
from sei_anonimizar_treino import SeiAnonimizarNERTrainer

# Preparar dados
df_entidades = pd.read_parquet("./entidades_anotadas.parquet")
df_textos = pd.read_parquet("./textos.parquet")

trainer = SeiAnonimizarNERTrainer(
    model_name=None,  # modelo em branco
    labels=['CPF', 'EMAIL', 'TELEFONE']
)

# Executar CV com 5 folds
reports, summaries, results, _ = trainer.cross_validate(
    df_entidades=df_entidades,
    df_textos=df_textos,
    n_splits=5,
    stratified=False,
    output_dir="./cv_results",
    n_jobs=1,
    train_params={'n_iter': 20, 'drop': 0.3, 'batch_size': 8},
    eval_params={'overlap_threshold': 0.8, 'beta': 2.0},
    replace=True
)

# Analisar resultados
for i, summary in enumerate(summaries, 1):
    print(f"Fold {i}: Precision={summary['precision']:.3f}, Recall={summary['recall']:.3f}, F-beta={summary['fbeta']:.3f}")
```


### 3.9.2 Cross-Validation estratificado por tipo de entidade

```python
# CV com balanceamento por tipo de entidade
reports, summaries, results, _ = trainer.cross_validate(
    df_entidades=df_entidades,
    df_textos=df_textos,
    n_splits=5,
    stratified=True,
    features=['CPF', 'EMAIL', 'TELEFONE'],  # Estratificar por estes tipos
    output_dir="./cv_estratificado",
    n_jobs=2,  # Paralelização
    train_params={'n_iter': 30},
    replace=True
)
```


### 3.9.3 **Cross-Validation com Holdout Test Set**

```python
# Separar 20% dos dados como test fixo antes do CV
reports, summaries, results, holdout_results = trainer.cross_validate(
    df_entidades=df_entidades,
    df_textos=df_textos,
    n_splits=5,
    output_dir="./cv_with_holdout",
    holdout_test_size=0.2,  # 20% para test fixo
    holdout_stratify=True,   # Estratificar o holdout também
    train_params={'n_iter': 25},
    replace=True
)

# Analisar métricas no holdout test (comparável entre folds)
if holdout_results is not None:
    print("\n=== Métricas no Holdout Test ===")
    for result in holdout_results:
        print(f"Fold {result['fold']}: F-beta={result['fbeta']:.3f}")
```


### 3.9.4 Cross-Validation com entrada JSONL

```python
# Usar arquivo JSONL diretamente no CV
reports, summaries, _, _ = trainer.cross_validate(
    df_entidades="./anotacoes_completas.jsonl",  # Arquivo JSONL
    df_textos=None,  # Mesmo arquivo contém textos e entidades
    n_splits=5,
    output_dir="./cv_from_jsonl",
    train_params={'n_iter': 15},
    replace=True
)
```


### 3.9.5 Rastreabilidade de Folds

```python
# Após executar CV, verificar IDs usados em cada fold
import json
from pathlib import Path

cv_dir = Path("./cv_results")

for fold_dir in sorted(cv_dir.glob("fold_*")):
    if fold_dir.is_dir():
        # Carregar IDs do fold
        with open(fold_dir / "fold_ids.json", 'r') as f:
            fold_info = json.load(f)
        
        print(f"\n{fold_dir.name}:")
        print(f"  Train IDs: {len(fold_info['train_ids'])} documentos")
        print(f"  Val IDs: {len(fold_info['val_ids'])} documentos")
        
        # Também disponível em CSV
        train_csv = pd.read_csv(fold_dir / "train_ids.csv")
        val_csv = pd.read_csv(fold_dir / "val_ids.csv")
```


### 3.9.6 Arquivos gerados pelo Cross-Validation

```
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
├── holdout_test_ids.json         # IDs separados para holdout test
├── holdout_test_ids.csv          # IDs do holdout em CSV
├── holdout_test_summary.parquet  # Métricas agregadas no holdout
└── holdout_test_stats.json       # Estatísticas gerais do holdout
```


***

## 4 Boas práticas e dicas rápidas

- **Para CPF**: mantenha validação ativa se desejar evitar falsos positivos; lembre-se de que CPFs inválidos não serão mascarados quando a validação estiver ligada.
- **Para regex custom**: adicione padrões específicos (ex.: placas) para complementar o modelo NER.
- **Offsets**: sempre confira se os offsets começam exatamente no primeiro caractere e terminam no último caractere da entidade, sem incluir espaço, vírgula, ou ponto final.
- **Modo de erros no treino**:
    - `raise`: interrompe no primeiro problema.
    - `ignore`: ignora entidades inválidas e segue com o resto.
    - `coerce`: tenta ajustar; se não for possível, ignora o inválido.
- **Avaliação**: compare relatórios atuais com anteriores para medir ganhos (TP/FP/FN por label e no agregado).
- **JSONL/Doccano**: Use para integração fácil com ferramentas de anotação; exporte/importe dados sem conversão manual.
- **Cross-Validation**:
    - Use `stratified=True` para datasets desbalanceados.
    - Use `holdout_test_size` quando precisar de métricas comparáveis entre folds.
    - Use `n_jobs > 1` para acelerar treinamento em múltiplos folds.
    - Arquivos `fold_ids.json` garantem reprodutibilidade e rastreabilidade.
- **Validação de entidades**: Novos checks (`start >= end`, `start < 0`) previnem erros comuns de anotação.
