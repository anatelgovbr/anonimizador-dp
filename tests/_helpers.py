import logging


def _make_logger() -> logging.Logger:
    return logging.getLogger("test_sei_anonimizar")


def _ent(label: str, start: int, end: int) -> dict:
    return {"label": label, "label_": label, "start_position": start, "end_position": end}


def make_ent(label: str, text: str, frag: str) -> dict:
    s = text.index(frag)
    return {"label_": label, "start_char": s, "end_char": s + len(frag)}


def _find_span(text: str, substr: str) -> tuple[int, int]:
    start = text.find(substr)
    if start == -1:
        msg = f"substring not found: {substr}"
        raise ValueError(msg)
    return start, start + len(substr)
