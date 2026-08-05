"""Módulo-ponte para compatibilidade legada.

Use ``from anonimizar import Trainer`` em vez de
``from anonimizar.sei_anonimizar_treino import SeiAnonimizarNERTrainer``.
"""

import warnings

from anonimizar._training.trainer_facade import Trainer


def __getattr__(name: str) -> type:
    if name == "SeiAnonimizarNERTrainer":
        warnings.warn(
            "anonimizar.sei_anonimizar_treino.SeiAnonimizarNERTrainer está depreciado; use anonimizar.Trainer.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Trainer
    raise AttributeError(name)
