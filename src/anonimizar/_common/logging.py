"""Factory de logger padrão para o pacote anonimizar."""

import logging

#: Formato aplicado por :func:`create_default_logger` a novos handlers.
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def create_default_logger(name: str) -> logging.Logger:
    """Cria um logger padrão com saída para console.

    Args:
        name: Nome do logger (tipicamente __name__ do módulo chamador).

    Returns:
        Logger configurado com ``StreamHandler``, formato padrão e nível
        ``INFO`` quando ainda não possui handlers. Um logger já configurado é
        retornado sem alterações.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
