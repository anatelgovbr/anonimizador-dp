"""Testes de compatibilidade para imports legados."""

import subprocess
import sys

import pytest


class TestNewImportsNoWarning:
    """Import novo no pacote não deve emitir warning."""

    def test_import_anonimizar(self):
        from anonimizar import Anonimizar

        assert Anonimizar is not None

    def test_import_evaluation(self):
        from anonimizar import Evaluation

        assert Evaluation is not None

    def test_import_trainer(self):
        from anonimizar import Trainer

        assert Trainer is not None


class TestDeprecatedAliasesWarn:
    """Alias antigo no pacote deve emitir DeprecationWarning."""

    def test_sei_anonimizar_alias_warns(self):
        with pytest.warns(DeprecationWarning):
            from anonimizar import SeiAnonimizar
        assert SeiAnonimizar is not None

    def test_sei_anonimizar_evaluation_alias_warns(self):
        with pytest.warns(DeprecationWarning):
            from anonimizar import SeiAnonimizarEvaluation
        assert SeiAnonimizarEvaluation is not None

    def test_sei_anonimizar_trainer_alias_warns(self):
        with pytest.warns(DeprecationWarning):
            from anonimizar import SeiAnonimizarNERTrainer
        assert SeiAnonimizarNERTrainer is not None


class TestBridgeModuleWarns:
    """Import pelo módulo-ponte deve emitir DeprecationWarning."""

    def test_bridge_sei_anonimizar_warns(self):
        with pytest.warns(DeprecationWarning):
            from anonimizar.sei_anonimizar import SeiAnonimizar
        assert SeiAnonimizar is not None

    def test_bridge_evaluation_warns(self):
        with pytest.warns(DeprecationWarning):
            from anonimizar.sei_anonimizar_evaluation import SeiAnonimizarEvaluation
        assert SeiAnonimizarEvaluation is not None

    def test_bridge_trainer_warns(self):
        with pytest.warns(DeprecationWarning):
            from anonimizar.sei_anonimizar_treino import SeiAnonimizarNERTrainer
        assert SeiAnonimizarNERTrainer is not None


class TestIdentity:
    """Alias e classe canônica devem ser o mesmo objeto."""

    def test_sei_anonimizar_is_anonimizar(self):
        from anonimizar import Anonimizar, SeiAnonimizar

        assert SeiAnonimizar is Anonimizar

    def test_sei_anonimizar_evaluation_is_evaluation(self):
        from anonimizar import Evaluation, SeiAnonimizarEvaluation

        assert SeiAnonimizarEvaluation is Evaluation

    def test_sei_anonimizar_trainer_is_trainer(self):
        from anonimizar import SeiAnonimizarNERTrainer, Trainer

        assert SeiAnonimizarNERTrainer is Trainer


class TestCLILegacy:
    """CLI legada deve funcionar via módulo-ponte."""

    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "anonimizar.sei_anonimizar", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "Anonimizador SEI" in result.stdout
