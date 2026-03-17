# SEI Anonimizar

Ferramenta avançada para anonimização de documentos e textos, desenvolvida para proteção de dados sensíveis em conformidade com a LGPD.

## Funcionalidades

- Detecção de +15 tipos de dados sensíveis (CPF, RG, CNH, etc.)
- Combinação de modelos SpaCy e expressões regulares
- Suporte a documentos em formato Markdown
- Processamento eficiente de grandes volumes de texto

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Uso Rápido

### Instalação

```bash
pip install anonimizador-dp
```

### Modelo NER (download via GitHub Releases)

O modelo treinado nao deve ficar no repo. Baixe o zip publicado na release e
configure `SPACY_MODEL_PATH`.

### Exemplo Básico

```python
from anonimizar.sei_anonimizar import SeiAnonimizar

anon = SeiAnonimizar(model_path="modelo_spacy")
anon.add_apply_patterns(['CPF', 'RG'])
# Para adicionar padroes proprios:
# anon.add_apply_patterns(['CPF', 'RG'], custom_patterns=[
#     {
#         "label": "MATRICULA_ALUNO",
#         "regex": r"[A-Z]{2}\d{5}[A-Z]",
#         "description": "Matrícula de aluno (2 letras + 5 números + 1 letra)"
#     },
#     {
#         "label": "PLACA_CARRO",
#         "regex": r"[A-Z]{3}-\d{4}",
#         "description": "Placa de carro no formato AAA-9999"
#     }
# ])

resultados = anon.extract_entities("Meu CPF é 123.456.789-00")
print(resultados)
```

### Exemplo completo
Para exemplos completos consulte: [Exemplos](./examples/README.MD)

### Documentação Completa

Consulte nossa [documentação](./docs/README.MD) detalhada para:

- Guia de instalação completa
- Lista de entidades suportadas
- Exemplos avançados de uso

## Contribuição

Siga estas etapas para contribuir com o projeto:

1. Fork o repositório

2. Crie uma branch para sua feature:

```bash
git checkout -b feature/nova-funcionalidade
```

3. Commit suas alterações:

```bash
git commit -m "Adiciona nova funcionalidade"
```

4. Push para a branch:

```bash
git push origin feature/nova-funcionalidade
```

5. Abra um Pull Request

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
