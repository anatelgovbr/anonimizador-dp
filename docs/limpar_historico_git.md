# Limpeza de historico do repo publico (Opcao B)

Este guia reescreve o historico do Git para remover pastas sensiveis.
Atencao: todos que clonaram o repo precisarao re-clonar depois.

## 1) Instalar git-filter-repo

```bash
pip install git-filter-repo
```

## 2) Fazer backup do repo

```bash
git clone --mirror https://github.com/anatelgovbr/anonimizador-dp.git anonimizador-dp.git.backup
```

## 3) Clonar o repo publico para limpeza

```bash
git clone https://github.com/anatelgovbr/anonimizador-dp.git anonimizador-dp-clean
cd anonimizador-dp-clean
```

## 4) Remover do historico pastas/arquivos sensiveis

```bash
git filter-repo --force --path notebooks --path docker --path htmlcov --path .pytest_cache --path .ruff_cache --invert-paths
```

Se precisar remover outros caminhos, adicione mais `--path` na lista.

## 5) Forcar push do historico limpo

```bash
git push origin --force --all
git push origin --force --tags
```

## 6) Bloquear novos arquivos no futuro

Atualize `.gitignore` e faca um commit normal no repo limpo.

---

# Limpar tudo e recriar historico (Opcao B2)

Use esta opcao se voce quer apagar TODO o historico e publicar um unico commit limpo.

## 1) Garantir o remote

```bash
git remote add origin https://github.com/anatelgovbr/anonimizador-dp.git
```
git remote set-url origin https://matheusgysi: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@github.com/anatelgovbr/anonimizador-dp.git
## 2) Criar branch orfa (historico novo)

```bash
git checkout --orphan clean-main
```

## 3) Remover tudo do indice

```bash
git rm -r --cached .
```

## 4) Adicionar somente o que deve ficar publico

```bash
git add docs examples src tests pyproject.toml setup.cfg README.md LICENSE pytest.ini CHANGELOG.MD AGENTS.md .gitignore
```

## 5) Criar commit limpo

```bash
git commit -m "Publicacao inicial limpa"
```

## 6) Renomear branch principal

```bash
git branch -M main
```

## 7) Forcar push substituindo todo o historico remoto

```bash
git push origin --force --all
git push origin --force --tags
```
