import pytest

from anonimizar import Anonimizar, Evaluation, Trainer, __version__


def test_version():
    assert __version__ == "1.0.6"


def test_imports():
    assert Anonimizar is not None
    assert Evaluation is not None
    assert Trainer is not None


def test_deprecated_aliases_warn_and_preserve_identity():
    with pytest.warns(DeprecationWarning):
        from anonimizar import SeiAnonimizar
    assert SeiAnonimizar is Anonimizar

    with pytest.warns(DeprecationWarning):
        from anonimizar import SeiAnonimizarEvaluation
    assert SeiAnonimizarEvaluation is Evaluation

    with pytest.warns(DeprecationWarning):
        from anonimizar import SeiAnonimizarNERTrainer
    assert SeiAnonimizarNERTrainer is Trainer
