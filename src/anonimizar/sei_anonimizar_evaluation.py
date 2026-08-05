"""Módulo-ponte para compatibilidade legada.

Use ``from anonimizar import Evaluation`` em vez de
``from anonimizar.sei_anonimizar_evaluation import SeiAnonimizarEvaluation``.
"""

import warnings

from anonimizar._evaluation.evaluation import Evaluation


def __getattr__(name: str) -> type:
    if name == "SeiAnonimizarEvaluation":
        warnings.warn(
            "anonimizar.sei_anonimizar_evaluation.SeiAnonimizarEvaluation está depreciado; use anonimizar.Evaluation.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Evaluation
    raise AttributeError(name)
