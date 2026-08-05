"""Funções offset-safe para normalização de entidades.

Todas as funções recebem ``(text, start, end, label)`` e retornam
``(new_text, new_start, new_end, rule)``, onde ``rule`` é uma string opcional
identificando a regra aplicada.

Uso típico:

.. code:: python

    from anonimizar._normalization import normalize_entity

    text = "RG 123456 SSP/SP"
    cleaned, start, end, rule = normalize_entity(text, 0, 16, "RG")
    # cleaned = "123456", start = 3, end = 9, rule = "prefix | sufixo"
"""

from __future__ import annotations

import re

# ─── helpers ────────────────────────────────────────────

_MIN_ALNUM_RESTANTE = 1


def _so_espacos(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _aparar_bordas(text: str, start: int, end: int) -> tuple[str, int, int]:
    n_esq = len(text) - len(text.lstrip())
    n_dir = len(text) - len(text.rstrip())
    return text.strip(), start + n_esq, end - n_dir


# ─── cauda de órgão emissor (RG) ───────────────────────

_CAUDA_ORGAO = (
    r"\s+[/\-]?\s*[A-ZÁ-Ú]{2,}"
    r"(?:\s*[/\-]?\s*[A-ZÁ-Ú]{2,})*"
    r"\s*$"
)

# ─── prefixos por label ─────────────────────────────────

#: Regras regex de prefixo por label. Cada lista é tentada em ordem e a primeira
#: regra aplicável define a normalização; trate o mapa como configuração interna.
PREFIXOS: dict[str, list[str]] = {
    "CID": [r"^\s*CID(?![A-Za-zÁ-Úá-ú])\s*[:\-]?\s*"],
    "CNH": [r"^\s*CNH\s*n?[º°o]?\.?\s*[:\-]?\s*"],
    "SIAPE": [
        r"^\s*SIAPE\s*n?[º°o]?\.?\s*[:\-]?\s*",
        r"^\s*n[º°o]\.?\s*[:\-]?\s*",
    ],
    "CPF": [
        r"^\s*CPF\s*n?[º°o]?\.?\s*[:\-]?\s*",
        r"^\s*n[º°o]\.?\s*[:\-]?\s*",
    ],
    "RG": [r"^\s*RG\s*n?[º°o]?\.?\s*[:\-]?\s*"],
    "TITULO_ELEITOR": [
        r"^\s*t[íi]tulo(?:\s+de)?\s+eleitor\s*[:\-]?\s*",
        r"^\s*n[ºo°]\.?\s*[:\-]?\s*",
    ],
    "GEO_COORD": [
        r"^\s*(?:(?:LAT|LONG|LOG)(?:ITUDE)?|COORDENADA[S]?)(?![A-Za-z])\s*\.?\s*(?::|-(?!\s*\d))?\s*",
    ],
    "EMAIL": [r"^\s*E-?MAIL(?![A-Za-z@])\s*[:\-]?\s*"],
}

# ─── sufixos por label ──────────────────────────────────

#: Regras regex de sufixo por label. Cada lista é tentada em ordem após a
#: remoção de prefixo; trate o mapa como configuração interna.
SUFIXOS: dict[str, list[str]] = {
    "CPF": [r"\s+-\s*\d{1,2}\s*$", _CAUDA_ORGAO],
    "RG": [
        _CAUDA_ORGAO,
        r"/\s*[A-ZÁ-Ú]{2,}\s*$",
    ],
    "CNH": [r"\s+DETRAN(\s*[/\-]?\s*[A-ZÁ-Ú]{2})?\s*$"],
}

# ─── funções públicas ───────────────────────────────────


def remover_prefixo(
    text: str,
    start: int,
    end: int,
    label: str,
    padroes_por_classe: dict[str, list[str]] | None = None,
) -> tuple[str, int, int, str | None]:
    """Remove prefixo de rótulo e apara bordas (offset-safe).

    Args:
        text: Texto bruto da entidade.
        start: Offset inicial.
        end: Offset final.
        label: Label da entidade.
        padroes_por_classe: Dict opcional de padrões (default ``PREFIXOS``).

    Returns:
        Tupla ``(texto_limpo, novo_start, novo_end, regra)``.
    """
    if padroes_por_classe is None:
        padroes_por_classe = PREFIXOS
    for pat in padroes_por_classe.get(label, []):
        m = re.match(pat, text, flags=re.IGNORECASE)
        if not m or m.end() == 0:
            continue
        restante = text[m.end() :]
        if sum(c.isalnum() for c in restante) < _MIN_ALNUM_RESTANTE:
            continue
        t, s, e = _aparar_bordas(restante, start + m.end(), end)
        return t, s, e, pat
    return text, start, end, None


def remover_sufixo(
    text: str,
    start: int,
    end: int,
    label: str,
    sufixos_por_classe: dict[str, list[str]] | None = None,
) -> tuple[str, int, int, str | None]:
    """Remove sufixo (órgão emissor, DV extra) e apara bordas (offset-safe).

    Args:
        text: Texto bruto da entidade.
        start: Offset inicial.
        end: Offset final.
        label: Label da entidade.
        sufixos_por_classe: Dict opcional de padrões (default ``SUFIXOS``).

    Returns:
        Tupla ``(texto_limpo, novo_start, novo_end, regra)``.
    """
    if sufixos_por_classe is None:
        sufixos_por_classe = SUFIXOS
    for pat in sufixos_por_classe.get(label, []):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        k = len(text) - m.start()
        if k == 0:
            continue
        restante = text[: m.start()]
        if sum(c.isalnum() for c in restante) < _MIN_ALNUM_RESTANTE:
            continue
        t, s, e = _aparar_bordas(restante, start, end - k)
        return t, s, e, pat
    return text, start, end, None


def remover_prefixo_sufixo(
    text: str,
    start: int,
    end: int,
    label: str,
) -> tuple[str, int, int, str | None]:
    """Aplica remoção de prefixo, sufixo e bordas.

    Args:
        text: Texto bruto da entidade.
        start: Offset inicial.
        end: Offset final.
        label: Label da entidade.

    Returns:
        Tupla ``(texto_limpo, novo_start, novo_end, regra)``.
    """
    t, s, e, r1 = remover_prefixo(text, start, end, label)
    t, s, e, r2 = remover_sufixo(t, s, e, label)
    regras = [r for r in (r1, r2) if r is not None]
    return t, s, e, " | ".join(regras) if regras else None


def normalize_entity(
    text_entity: str,
    start: int,
    end: int,
    label: str,
) -> tuple[str, int, int, str | None]:
    """Normaliza uma entidade: remove prefixos e sufixos conhecidos.

    Args:
        text_entity: Texto bruto da entidade.
        start: Offset inicial no documento original.
        end: Offset final no documento original.
        label: Label da entidade (``"RG"``, ``"GEO_COORD"``, etc.).

    Returns:
        Tupla ``(texto_normalizado, novo_start, novo_end, regra)``.
        Se nenhuma regra foi aplicada, retorna os valores originais.
    """
    return remover_prefixo_sufixo(text_entity, start, end, label)
