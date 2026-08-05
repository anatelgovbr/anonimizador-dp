"""Módulo-ponte para compatibilidade legada.

Use ``from anonimizar import Anonimizar`` em vez de
``from anonimizar.sei_anonimizar import SeiAnonimizar``.
"""

import warnings

from anonimizar._anonymization.anonymizer import Anonimizar, main


def __getattr__(name: str) -> type:
    if name == "SeiAnonimizar":
        warnings.warn(
            "anonimizar.sei_anonimizar.SeiAnonimizar está depreciado; use anonimizar.Anonimizar.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Anonimizar
    raise AttributeError(name)


if __name__ == "__main__":
    raise SystemExit(main())
