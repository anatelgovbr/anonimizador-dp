"""Pipeline de extração de entidades combinando múltiplas fontes.

Este módulo orquestra a extração de entidades combinando modelo spaCy,
padrões regex e tabelas markdown.
"""

import os
import re
from collections.abc import Callable
from logging import Logger
from typing import Any

from anonimizar._common.overlap import remove_overlap_positions
from anonimizar._extraction.markdown import extract_entities_from_markdown_tables
from anonimizar._extraction.model import extract_from_model
from anonimizar._extraction.regex import extract_entities_regex_re

PREFIXOS_ENTIDADE: dict[str, list[str]] = {
    "SIAPE": [
        r"^siape\s*n[º°o]?\s*",
        r"^siape\s*",
        r"^matr[íi]cula\s+siape\s*",
    ],
    "CPF": [
        r"^cpf\s*n[º°o]?\s*:?\s*",
        r"^n[º°o]\s*:?\s*",
    ],
    "CNH": [
        r"^cnh\s*[:°º]?\s*",
    ],
    "TITULO_ELEITOR": [
        r"^t[íi]tulo\s+de\s+eleitor\s*[:\-]?\s*",
        r"^n[º°o]\s*:?\s*",
    ],
    "RG": [
        r"^rg\s*n[º°o]?\s*:?\s*",
    ],
}


def _limpar_prefixo_entidade(text: str, label: str) -> tuple[str, int]:
    """Remove prefixos comuns de entidades detectadas fora de tabelas.

    Args:
        text: Texto da entidade detectada.
        label: Tipo de entidade (CPF, SIAPE, etc).

    Returns:
        Tupla com (texto_limpo, caracteres_removidos).
    """
    original_len = len(text)
    if label not in PREFIXOS_ENTIDADE:
        return text, 0

    for prefix in PREFIXOS_ENTIDADE[label]:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE).strip()

    return text, original_len - len(text)


def extract_entities(
    nlp_trained: Any,  # noqa: ANN401
    text_or_path: str,
    labels: set[str],
    patterns: list[dict],
    return_type: str,
    verify_fn: Callable[[Any, str], bool | str],
    logger: Logger,
) -> list[dict]:
    """Extrai entidades combinando modelo spaCy, regex e tabelas markdown.

    Args:
        nlp_trained (Any): Modelo spaCy treinado
        text_or_path (str): Texto ou caminho para arquivo .md
        labels (set[str]): Labels ativos
        patterns (list[dict]): Padrões regex configurados
        return_type (str): Formato de retorno
        verify_fn (Callable): Função de verificação contextual
        logger (Logger): Logger para debug

    Returns:
        list[dict]: Lista de entidades extraídas e validadas

    Raises:
        ValueError: Se arquivo não for .md ou return_type inválido
        FileNotFoundError: Se arquivo não existir
    """
    # Carregar texto de arquivo se necessário
    if os.path.isfile(text_or_path):  # noqa: PTH113
        if text_or_path.lower().endswith(".md"):
            try:
                with open(text_or_path, encoding="utf-8") as file:  # noqa: PTH123
                    text = file.read()
                logger.info("Arquivo carregado. Tamanho: %d caracteres", len(text))
            except UnicodeDecodeError as e:
                logger.exception("Erro de codificação ao ler arquivo '%s': %s", text_or_path, e)  # noqa: TRY401
                raise
            except OSError as e:
                logger.exception("Erro de I/O ao ler arquivo '%s': %s", text_or_path, e)  # noqa: TRY401
                raise
        else:
            msg_erro = "O arquivo fornecido não é um MD!"
            logger.exception("%s Arquivo: %s", msg_erro, text_or_path)
            raise ValueError(msg_erro)
    else:
        logger.debug("Processando texto direto. Tamanho: %d caracteres", len(text_or_path))
        text = text_or_path

    logger.debug("Extraindo entidades: return_type=%s len_texto=%d", return_type, len(text))

    # Extrair de todas as fontes
    entities_from_model = extract_from_model(nlp_trained, text, labels, return_type, verify_fn, logger)
    entities_from_regex = extract_entities_regex_re(text, patterns, return_type, verify_fn, logger)
    entities_from_tables = extract_entities_from_markdown_tables(text, return_type, logger, labels)

    logger.debug(
        "Detecções: modelo=%d regex=%d tabelas=%d",
        len(entities_from_model),
        len(entities_from_regex),
        len(entities_from_tables),
    )

    # Combinar e remover sobreposições
    if return_type in {"label_position", "label_detail"}:
        merged = remove_overlap_positions(
            entities_from_model + entities_from_regex + entities_from_tables, logger=logger
        )
        logger.debug("Após remoção de overlap: %d", len(merged))

        # Remover prefixos de entidades detectadas fora de tabelas markdown
        for entity in merged:
            detected_by = entity.get("detected_by", "")
            if detected_by != "tabela_markdown":
                label = entity.get("label", "")
                original_text = entity.get("text", "")
                cleaned_text, delta = _limpar_prefixo_entidade(original_text, label)

                entity["text"] = cleaned_text
                if delta > 0:
                    entity["start_position"] = entity.get("start_position", 0) + delta

        return merged

    if return_type == "label_text":
        unique_entities = {
            (e["label"], e["text"]): e for e in entities_from_model + entities_from_regex + entities_from_tables
        }.values()
        result = list(unique_entities)
        logger.debug("Total após unificação (sem posições): %d", len(result))

        # Remover prefixos de entidades detectadas fora de tabelas markdown
        for entity in result:
            detected_by = entity.get("detected_by", "")
            if detected_by != "tabela_markdown":
                label = entity.get("label", "")
                original_text = entity.get("text", "")
                cleaned_text, delta = _limpar_prefixo_entidade(original_text, label)

                entity["text"] = cleaned_text
                if delta > 0:
                    entity["start_position"] = entity.get("start_position", 0) + delta

        return result

    msg = f"Tipo de retorno não permitido {return_type}"
    logger.error(msg)
    raise ValueError(msg)
