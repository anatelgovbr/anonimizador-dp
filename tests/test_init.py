from anonimizar import SeiAnonimizar, SeiAnonimizarEvaluation, SeiAnonimizarNERTrainer, __version__


def test_version():
    """Testa a versao da biblioteca"."""
    assert __version__ == "1.0.0"


def test_imports():
    """Teste de imports nao nulos"""
    assert SeiAnonimizar is not None
    assert SeiAnonimizarEvaluation is not None
    assert SeiAnonimizarNERTrainer is not None
