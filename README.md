# Anonimizador-dp

Anonimizador-dp é uma biblioteca Python open source, desenvolvida pela Anatel, que combina aprendizado de máquina tradicional (NER) com expressões regulares para detectar e anonimizar dados pessoais em textos em português brasileiro.

Ferramenta avançada para anonimização de documentos e textos em **português brasileiro**, desenvolvida para proteção de dados pessoais, como números de documentos pessoais, telefone, e-mail, data de nascimento, dados bancários, coordenadas geográficas e endereço.

> **Dados Fictícios:** todos os dados usados nos exemplos e no treinamento dos
> modelos NER distribuídos no projeto são fictícios. Eles preservam somente
> padrões estruturais existentes em documentos reais, sem reproduzir dados
> pessoais ou conteúdos reais. **Apenas parecem dados reais**, mas são
> totalmente fictícios e fora do contexto do conteúdo do texto de origem
> utilizado no treinamento dessa biblioteca.

## Funcionalidades

- Detecção de 16 tipos de dados pessoais (CPF, RG, CNH, CID, GEO_COORD, PIS, CNS, RESERVISTA, etc.)
- Combinação de modelos SpaCy e expressões regulares
- Suporte a documentos em formato Markdown
- Processamento eficiente de grandes volumes de texto
- Suporte a padrões personalizados/customizados
- Treinamento por curriculum learning com janelas de dificuldade (`w0`/`w1`/`w2`/`full`/`w00`)

## Padrões Suportados

| Padrão | Descrição | Método de Detecção |
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
- **validação contextual** = palavras-chave no entorno

## Benchmark

> **Nota de validade:** os números abaixo referem-se a uma rodada específica de benchmark, realizada com o modelo NER distribuído na época dessa análise.
> O comparativo deve ser refeito conforme novos modelos forem publicados; os resultados não garantem o desempenho em versões futuras.

Comparativo do Anonimizador-dp com outras soluções de anonimização em documentos
jurídicos brasileiros:

| Sistema | Qtd. labels | F-beta | Precisão | Recall |
|---|---:|---:|---:|---:|
| **Anonimizador-dp** | **16** | **0.8748** | **0.9222** | **0.8637** |
| Microsoft Presidio | 16 | 0.5378 | 0.5197 | 0.5425 |
| Azure Language PII + patterns | 16 | 0.5458 | 0.5230 | 0.5518 |
| GLiNER (gliner_multi_pii-v1) | 16 | 0.5419 | 0.6691 | 0.5173 |

**Metodologia:**
- 97 documentos e 1.071 entidades de ground truth (usando dados fictícios).
- 16 labels avaliados.
- Overlap threshold 0.8 e beta 2.0, mesma configuração da `Evaluation`.
- Comparação justa com o serviço gerenciado: no recorte das 4 labels nativas do Azure Language PII (CPF, EMAIL, ENDEREÇO, TELEFONE), o Anonimizador-dp alcança F-beta 0.9409 contra 0.5292 do Azure puro.

## Dados de Treinamento

Os modelos NER distribuídos são treinados com dados fictícios, seguindo o fluxo:

1. **Extração inicial** — textos públicos foram extraídos, porém apresentavam pouca presença de dados pessoais (PII).
2. **Fontes com PII → dados sintéticos** — foram coletados textos de processos do SEI contendo dados pessoais; os valores PII reais foram substituídos por valores sintéticos plausíveis, preservando o formato e o contexto originais.
3. **Treinamento** — os modelos são treinados sobre a base sintética.
4. **Avaliação em dados reais** — as métricas são calculadas sobre ground truth real (não sintético), evitando overfitting ao gerador de dados e garantindo a validade e a qualidade dos modelos em ambiente real.


## Uso Rápido

### Instalação

```bash
pip install anonimizador-dp
```

O nome do pacote no PyPI (`anonimizador-dp`) é diferente do nome do módulo de
import (`anonimizar`), como em `scikit-learn`/`sklearn`:

```python
from anonimizar import Anonimizar
```

### Modelo NER (download via GitHub Releases)

O modelo treinado nao deve ficar no repo. Baixe o zip publicado na release e
configure `SPACY_MODEL_PATH`.

### Exemplo Básico

```python
from anonimizar import Anonimizar

anonymizer = Anonimizar(model_path="modelo_spacy")
# Por padrão, auto_patterns=True aplica os padrões built-in e inclui
# RG estrangeiro (RNE/CRNM) como label RG.
# Para desativar, inicialize com auto_patterns=False e registre RG com foreign_rg=False.

texto = "Meu CPF é 123.456.789-09."
entidades = anonymizer.extract_entities(texto, return_type="label_position")
texto_anonimizado = anonymizer.anonymize_text(texto, entidades)

print(entidades)
print(texto_anonimizado)  # Meu CPF é <|CPF|>.
```

### Tipos de Retorno — Quando Usar Cada Um

| Tipo | Cenário de Uso | Exemplo de Retorno |
|------|----------------|-------------------|
| `anonymize_text()` | **Processamento massivo** — substitui automaticamente todas as entidades por tags como `<\|CPF\|>`, `<\|RG\|>`, etc. Sem intervenção humana. | `"Meu CPF é <\|CPF\|>."` |
| `label_position` | **Pré-processamento interativo** — exiba as marcações para o usuário revisar, corrigir ou complementar antes da anonimização definitiva. Ideal para sistemas de anotação humana. | `[{"label": "CPF", "start_position": 10, "end_position": 24}]` |
| `label_text` | **Auditoria rápida** — quando você precisa apenas saber quais valores foram encontrados, sem se preocupar com coordenadas. | `[{"label": "CPF", "text": "123.456.789-09"}]` |
| `label_detail` | **Depuração e rastreabilidade** — mostra qual método detectou cada entidade (modelo, regex ou tabela markdown). | `[{"label": "CPF", "start_position": 10, "end_position": 24, "text": "123.456.789-09", "detected_by": "regex"}]` |

Para adicionar um padrão customizado:

```python
anonymizer.add_custom_pattern(
    label="PLACA_CARRO",
    regex_pattern=r"[A-Z]{3}-\d{4}",
    description="Placa de carro no formato AAA-9999",
)
```

### Normalização de Entidades

Por padrão, entidades detectadas são normalizadas para remover prefixos e sufixos
textuais (ex: `"RG: 1234567 SSP/DF"` vira `"1234567"`). Para desativar:

```python
anonymizer = Anonimizar(model_path="modelo_spacy", normalize_entities=False)
```

O mesmo vale para `Trainer` e `Evaluation`:

### Treinamento com Épocas Personalizadas

O `Trainer` permite ajustar o número de épocas (iterações) do treinamento pelo parâmetro `n_iter` (padrão: 20):

```python
from anonimizar import Trainer

trainer = Trainer(
    model_name="pt_core_news_sm",
    output_dir="./meu_modelo_ner",
    labels=["CPF", "RG", "EMAIL"],
)

dados = [
    {
        "text": "João Silva, CPF 123.456.789-00, email: joao@email.com",
        "entities": [(12, 26, "CPF"), (35, 50, "EMAIL")],
    }
]
trainer.add_data(dados, errors="coerce")

# n_iter define o número de épocas; aumente para melhorar o ajuste do modelo
trainer.train(n_iter=30, validation_split=0.2)

trainer.save_model()
```

No `cross_validate`, as épocas e demais hiperparâmetros são passados via `train_params`:

```python
train_params = {"n_iter": 30, "drop": 0.2, "batch_size": 8}

reports, summaries, results, holdout = trainer.cross_validate(
    df_entidades=df_entidades,
    df_textos=df_textos,
    n_splits=5,
    train_params=train_params,
)
```

### Curriculum Learning com Janelas de Dificuldade

O `Trainer` também suporta curriculum learning: o modelo é treinado por fases
sequenciais, começando por exemplos fáceis (entidade isolada ou parágrafo) e
evoluindo para o documento completo — padrão validado nos experimentos da
estória 942. Há dois fluxos de entrada, misturáveis entre fases:

1. **End-to-end**: informe `df_textos` + `df_entidades` (ou um caminho
   `.jsonl`) e referencie cada fase pelo nome da janela em `dataset` (`w0`,
   `w1`, `w2`, `full` ou `w00`). As janelas são geradas internamente, uma
   única vez.
2. **Separado**: informe dados prontos em `dataset` (dicts no formato do
   `add_data`, tuplas `(texto, {"entities": [...]})` ou caminho `.jsonl`).

Valor string de `dataset` que não termina em `.jsonl` é tratado como nome de
janela (fluxo e2e); qualquer outro valor é dado pronto (fluxo separado).

```python
from anonimizar import Trainer

trainer = Trainer(labels=["CPF", "EMAIL"])

# Fluxo end-to-end: dois DataFrames (ou um .jsonl em df_textos)
metrics = trainer.train_curriculum(
    df_textos=df_textos,
    df_entidades=df_entidades,
    phases=[
        {"name": "w0", "dataset": "w0", "epochs": 2},
        {"name": "full", "dataset": "full", "epochs": 8},
    ],
)

# Fluxo separado: dados já preparados por fase (ex.: joblibs da 942)
metrics = trainer.train_curriculum(
    phases=[
        {"dataset": datasets["sujo"]["w0"], "epochs": 2},
        {"dataset": datasets["ouro"]["full"], "epochs": 8},
    ],
)

trainer.save_model()
```

Janelas entre fases são geradas pelo `build_curriculum_datasets` e podem ser
persistidas/recarregadas com `save_curriculum_datasets`/`load_curriculum_datasets`
(formato joblib, compatível com o dos experimentos da estória 942). Labels novos
encontrados nas fases (ex.: `PIS`, `CNS`) são registrados automaticamente.

### Exemplo completo
Para exemplos completos consulte: [Exemplos](./examples/README.md)

### Documentação Completa

Consulte nossa [documentação](./docs/README.md) detalhada para:

- Guia de instalação completa
- Lista de entidades suportadas
- Exemplos avançados de uso
- Configuração de logging

### Configuração de Logging

O pacote usa o módulo padrão `logging` do Python. Configure conforme o cenário:

```python
import logging

# Uso operacional (lote) — apenas erros e avisos
logging.getLogger("anonimizar").setLevel(logging.WARNING)

# Uso operacional com progresso visível — sem logs por documento
logging.getLogger("anonimizar").setLevel(logging.INFO)

# Depuração detalhada — logs por documento e por entidade
logging.getLogger("anonimizar").setLevel(logging.DEBUG)
```

## Como Funciona

O pipeline de detecção combina três fontes de extração em etapas sequenciais:

```
               Texto de entrada
                      │
                      ▼
    ┌─────────────────────────────┐
    │  1. Modelo spaCy NER        │  ← identifica entidades via ML
    └──────────┬──────────────────┘
                      │
                      ▼
    ┌─────────────────────────────┐
    │  2. Padrões REGEX           │  ← captura padrões conhecidos
    └──────────┬──────────────────┘
                      │
                      ▼
    ┌─────────────────────────────┐
    │  3. Tabelas Markdown        │  ← extrai colunas com dados pessoais
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

Cada entidade detectada passa por um validador específico que pode ser:
- **Validação algorítmica** — confere dígitos verificadores (CPF, CNH, TITULO_ELEITOR, PIS, CNS)
- **Validação estrutural** — verifica formato esperado (EMAIL, CID, GEO_COORD, RG)
- **Validação contextual** — busca palavras-chave no entorno do texto (SIAPE, TELEFONE, PASSAPORTE, etc.)

## Licença

Este projeto está licenciado sob a GNU General Public License v3.0 - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Contribuição

Siga estas etapas para contribuir com o projeto:

1. Abra uma Issue descrevendo o problema, a melhoria ou a nova funcionalidade proposta.

   Se a sua contribuição envolver dados, você pode disponibilizá-los anotados no
   padrão do Doccano (JSONL) para adicionarmos à nossa base de treinamento.

2. Fork o repositório

3. Crie uma branch para sua feature:

```bash
git checkout -b feature/nova-funcionalidade
```

4. Commit suas alterações:

```bash
git commit -m "Adiciona nova funcionalidade"
```

5. Push para a branch:

```bash
git push origin feature/nova-funcionalidade
```

6. Abra um Pull Request

### Padrões de Código

- Siga o estilo PEP 8
- Documente novas funcionalidades com docstrings
- Adicione testes para novas features

#### Linting

Para verificar se o código está passando nos testes de lint, use o comando:

```bash
ruff check .
```

Desenvolvido por ANATEL.
