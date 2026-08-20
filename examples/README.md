# Exemplos de Uso

Pasta contendo notebooks Jupyter, um guia Markdown e um script Python que demonstram o uso do Anonimizar em diferentes cenários.

## Padrões Disponíveis

O Anonimizar suporta detecção e anonimização dos seguintes padrões:

- **CPF** - Cadastro de Pessoa Física
- **RG** - Registro Geral
- **RNE/CRNM** - Registro Nacional Migratório/Carteira de Registro Nacional Migratório, detectados como `RG` quando `auto_patterns=True`
- **CNH** - Carteira Nacional de Habilitação
- **TITULO_ELEITOR** - Título de Eleitor
- **PASSAPORTE** - Passaporte
- **SIAPE** - Sistema Integrado de Administração de Pessoal
- **DATA_NASCIMENTO** - Data de Nascimento
- **DADOS_BANCARIOS** - Dados Bancários
- **EMAIL** - Endereço de E-mail
- **TELEFONE** - Número de Telefone
- **ENDEREÇO** - Endereço Postal / CEP
- **CID** - Classificação Internacional de Doenças
- **GEO_COORD** - Coordenadas Geográficas
- **PIS** - Programa de Integração Social
- **CNS** - Cartão Nacional de Saúde
- **RESERVISTA** - Certificado de reservista

## Comportamento Atual

Os exemplos consideram o comportamento padrão da versão atual: ao instanciar `Anonimizar(model_path)`, `auto_patterns=True` aplica automaticamente os padrões built-in. Isso inclui RNE/CRNM como entidades com label `RG`.

Para controlar os padrões manualmente, inicialize com `auto_patterns=False` e chame `add_apply_patterns()` explicitamente:

```python
anonymizer = Anonimizar(model_path, auto_patterns=False)
anonymizer.add_apply_patterns(["CPF", "RG", "EMAIL"], foreign_rg=False)
```

Configure logging conforme o cenário de uso:

```python
import logging

logging.getLogger("anonimizar").setLevel(logging.WARNING)  # lote operacional
logging.getLogger("anonimizar").setLevel(logging.DEBUG)    # depuração detalhada
```

### 1. [exemplo.ipynb](exemplo.ipynb)

Demonstração básica das funcionalidades principais:

- Carga e processamento de texto individual
- Detecção de dados pessoais
- Anonimização básica
- Visualização dos resultados

### 2. [exemplo_em_lote.ipynb](exemplo_em_lote.ipynb)

Exemplo avançado para processamento em lote:

- Processamento de múltiplos documentos
- Configuração de padrões personalizados (CPF, RG, EMAIL, CID, GEO_COORD, etc.)
- Exportação de resultados estruturados

### 3. [exemplo_completo.ipynb](exemplo_completo.ipynb)

Exemplo completo de pipeline de treinamento, validação e uso:

- Preparação de dados de treinamento
- Treinamento de modelo NER
- Validação e avaliação
- Aplicação prática em textos fictícios com padrões estruturais realistas

### 4. [exemplo_curriculum.ipynb](exemplo_curriculum.ipynb)

Exemplo de treinamento por **curriculum learning** (fases sequenciais de dificuldade, janelas `w0`/`full`/`w00`):

- Fluxo end-to-end: `train_curriculum()` com `df_textos` + `df_entidades` e fases por `dataset`
- Fluxo separado: datasets gerados com `build_curriculum_datasets` e persistidos em joblib (`save`/`load`)
- Fluxo misto (`data` pronto + `dataset` gerado no mesmo curriculum)
- Dados 100% fictícios

### Outros exemplos

- [exemplos_de_uso.md](exemplos_de_uso.md): guia textual de extração, anonimização, treino, JSONL e cross-validation.
- [test_errors_examples.py](test_errors_examples.py): cenários executáveis das políticas de erro e limpeza do treinamento.

