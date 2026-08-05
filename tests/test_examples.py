"""Testes de regressão para os exemplos distribuídos com o pacote."""

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EXAMPLES_DIR = ROOT / "examples"


def _load_example_script():
    """Carrega o script de exemplos sem executá-lo no momento da importação."""
    path = EXAMPLES_DIR / "test_errors_examples.py"
    spec = importlib.util.spec_from_file_location("test_errors_examples", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _markdown_python_blocks() -> list[str]:
    """Retorna os blocos Python do guia de exemplos."""
    content = (EXAMPLES_DIR / "exemplos_de_uso.md").read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", content, flags=re.DOTALL)


def _notebook_code_cells() -> list[tuple[Path, int, str]]:
    """Retorna código e posição das células Python dos notebooks."""
    cells = []
    for notebook_path in sorted(EXAMPLES_DIR.glob("*.ipynb")):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                cells.append((notebook_path, index, "".join(cell["source"])))
    return cells


def test_markdown_example_blocks_are_valid_python() -> None:
    """Garante que blocos Python do guia não contenham sintaxe incompleta."""
    blocks = _markdown_python_blocks()

    assert blocks
    for index, block in enumerate(blocks):
        ast.parse(block, filename=f"exemplos_de_uso.md:bloco-{index + 1}")


def test_markdown_examples_use_public_imports() -> None:
    """Protege os imports públicos usados no guia Markdown."""
    content = (EXAMPLES_DIR / "exemplos_de_uso.md").read_text(encoding="utf-8")

    assert "from anonimizar import Anonimizar" in content
    assert "from anonimizar import Anonimizar, Evaluation" in content
    assert "from anonimizar import Anonimizar, Trainer" in content
    assert "from sei_anonimizar import" not in content
    assert "from sei_anonimizar_evaluation import" not in content
    assert "from sei_anonimizar_treino import" not in content


@pytest.mark.parametrize(("notebook_path", "cell_index", "source"), _notebook_code_cells())
def test_notebook_code_cells_compile(notebook_path: Path, cell_index: int, source: str) -> None:
    """Garante que todas as células de código dos notebooks sejam compiláveis."""
    assert cell_index >= 0
    ast.parse(source, filename=str(notebook_path))


def test_notebook_examples_have_no_placeholder_model_path() -> None:
    """Garante que os notebooks obtenham o modelo pela configuração do ambiente."""
    for notebook_path in (EXAMPLES_DIR / "exemplo.ipynb", EXAMPLES_DIR / "exemplo_em_lote.ipynb"):
        content = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join("".join(cell["source"]) for cell in content["cells"] if cell["cell_type"] == "code")

        assert 'os.getenv("SPACY_MODEL_PATH")' in code
        assert 'model_path = "Seu_caminho"' not in code


def test_batch_notebook_builds_structured_anonymization_results() -> None:
    """Protege o contrato anunciado pelo exemplo de processamento em lote."""
    notebook = json.loads((EXAMPLES_DIR / "exemplo_em_lote.ipynb").read_text(encoding="utf-8"))
    content = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")

    assert '"id": indice' in content
    assert '"texto": texto_original' in content
    assert '"entidades": doc_ents' in content
    assert '"anonimizado": texto_anonimizado' in content
    assert "entitities" not in content


def test_error_policy_script_completes(capsys) -> None:
    """Garante que o script de políticas não termine com erro de exemplo."""
    module = _load_example_script()

    module.main()

    output = capsys.readouterr().out
    assert "Execução concluída com sucesso!" in output
    assert "5.2 Sobreposição" in output


def test_documented_example_offsets_match_their_texts() -> None:
    """Protege os spans principais documentados no guia Markdown."""
    examples = [
        (
            "\nNome: João Silva\nCPF: 123.456.789-00\nRG: 12.345.678-9\nTelefone: (61) 9999-8888\n",
            [("12.345.678-9", 42, 54), ("(61) 9999-8888", 65, 79)],
        ),
        (
            "\nNome: João Sauro\nCPF: 123.456.789-09\n",
            [("123.456.789-09", 23, 37)],
        ),
    ]

    for text, spans in examples:
        for entity_text, start, end in spans:
            assert text[start:end] == entity_text


def test_example_script_documents_strict_clean_behavior() -> None:
    """Garante que o script demonstre explicitamente strict_clean."""
    content = (EXAMPLES_DIR / "test_errors_examples.py").read_text(encoding="utf-8")

    assert "strict_clean=False" in content
    assert 'resolve_conflicts="coerce"' in content
