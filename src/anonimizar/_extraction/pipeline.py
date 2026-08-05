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
from anonimizar._normalization import normalize_entity

#: Regras regex internas para remover prefixos de entidades detectadas fora de
#: tabelas Markdown. O mapa é um detalhe de implementação, não uma API estável.
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
        r"^rg\s*(?:n[º°o]?\s*)?:?\s*",
    ],
    "TELEFONE": [
        r"^(?:telefone|tel\.?|fone|celular)\s*[:]\s*",
    ],
    "PASSAPORTE": [
        r"^(?:passaporte|passport)\s*n?[ºo°]?\s*[:]\s*",
    ],
    "DADOS_BANCARIOS": [
        r"^(?:conta|ag[êe]ncia|banco)\s*[:]\s*",
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


def _limpar_prefixos_entidades(entities: list[dict[str, Any]]) -> None:
    """Remove prefixos e sufixos textuais antes de resolver overlaps."""
    for entity in entities:
        detected_by = entity.get("detected_by", "")
        if detected_by == "tabela_markdown":
            continue

        label = entity.get("label", "")
        original_text = entity.get("text", "")
        start = entity.get("start_position", 0)
        end = entity.get("end_position", len(original_text))

        cleaned_text, new_start, new_end, _rule = normalize_entity(original_text, start, end, label)

        entity["text"] = cleaned_text
        if new_start != start or new_end != end:
            entity["start_position"] = new_start
            entity["end_position"] = new_end


def _split_entities_by_newline(entities: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    r"""Divide entidades que contenham quebras de linha em entidades separadas.

    Cobre todas as variantes: ``\n``, ``\r\n``, ``\r`` e combinações.
    A função recalcula os offsets ``start_position``/``end_position`` e o
    texto de cada entidade preservando as demais chaves.

    Args:
        entities: Lista de entidades no formato do pipeline de extração.
        text: Texto completo do documento de onde as entidades foram extraídas.

    Returns:
        Nova lista com entidades separadas por quebra de linha.
    """
    newline_re = re.compile(r"\r?\n|\r")
    result: list[dict[str, Any]] = []
    for ent in entities:
        start = ent["start_position"]
        end = ent["end_position"]
        ent_text = ent.get("text", text[start:end])

        positions = list(newline_re.finditer(ent_text))
        if not positions:
            result.append(ent)
            continue

        cursor = start
        prev_end = 0
        for m in positions:
            seg_text = ent_text[prev_end : m.start()]
            if seg_text:
                new_ent = dict(ent)
                new_ent["text"] = seg_text
                new_ent["start_position"] = cursor
                new_ent["end_position"] = cursor + len(seg_text)
                result.append(new_ent)
                cursor += len(seg_text)
            cursor += m.end() - m.start()
            prev_end = m.end()

        seg_text = ent_text[prev_end:]
        if seg_text:
            new_ent = dict(ent)
            new_ent["text"] = seg_text
            new_ent["start_position"] = cursor
            new_ent["end_position"] = cursor + len(seg_text)
            result.append(new_ent)

    return result


def extract_entities(
    nlp_trained: Any,  # noqa: ANN401
    text_or_path: str,
    labels: set[str],
    patterns: list[dict],
    return_type: str,
    verify_fn: Callable[[Any, str], bool | str],
    logger: Logger,
    *,
    normalize: bool = True,
) -> list[dict]:
    """Extrai entidades combinando modelo spaCy, regex e tabelas markdown.

    Args:
        nlp_trained (Any): Modelo spaCy treinado
        text_or_path (str): Texto literal ou caminho existente para arquivo ``.md``.
            Uma string que não referencia um arquivo existente é processada como texto.
        labels (set[str]): Labels ativos
        patterns (list[dict]): Padrões regex configurados
        return_type (str): Formato de retorno
        verify_fn (Callable): Função de verificação contextual
        logger (Logger): Logger para debug
        normalize (bool): Se True, remove prefixos/sufixos das entidades.

    Returns:
        list[dict]: Lista de entidades extraídas e validadas

    Raises:
        ValueError: Se um arquivo existente não for ``.md`` ou se ``return_type``
            for inválido.
        UnicodeDecodeError: Se a leitura de um arquivo ``.md`` UTF-8 falhar.
        OSError: Se ocorrer uma falha de I/O ao ler um arquivo ``.md``.
    """
    # Carregar texto de arquivo se necessário
    if os.path.isfile(text_or_path):  # noqa: PTH113
        if text_or_path.lower().endswith(".md"):
            try:
                with open(text_or_path, encoding="utf-8") as file:  # noqa: PTH123
                    text = file.read()
                logger.debug("Arquivo carregado. Tamanho: %d caracteres", len(text))
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

    # Extrair de todas as fontes - sempre usamos label_detail internamente para processamento
    entities_from_model = extract_from_model(nlp_trained, text, labels, "label_detail", verify_fn, logger)
    entities_from_regex = extract_entities_regex_re(text, patterns, "label_detail", verify_fn, logger)
    entities_from_tables = extract_entities_from_markdown_tables(text, "label_detail", logger, labels)

    logger.debug(
        "Detecções: modelo=%d regex=%d tabelas=%d",
        len(entities_from_model),
        len(entities_from_regex),
        len(entities_from_tables),
    )

    # Combinar e remover sobreposições
    all_entities = entities_from_model + entities_from_regex + entities_from_tables
    all_entities = _split_entities_by_newline(all_entities, text)
    if normalize:
        _limpar_prefixos_entidades(all_entities)
    merged = remove_overlap_positions(all_entities, logger=logger)
    logger.debug("Após remoção de overlap: %d", len(merged))

    # Converter para o formato de retorno solicitado
    if return_type == "label_detail":
        return merged

    if return_type == "label_position":
        return [
            {"label": e["label"], "start_position": e["start_position"], "end_position": e["end_position"]}
            for e in merged
        ]

    if return_type == "label_text":
        # Unificar duplicatas label/text
        unique_entities = {(e["label"], e["text"]): {"label": e["label"], "text": e["text"]} for e in merged}.values()
        return list(unique_entities)

    msg = f"Tipo de retorno não permitido {return_type}"
    logger.error(msg)
    raise ValueError(msg)
