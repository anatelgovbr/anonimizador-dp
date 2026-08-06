"""Funções para extração de entidades de tabelas markdown.

Este módulo fornece funções para detectar e extrair dados pessoais
de tabelas markdown, identificando colunas que contenham palavras-chave
relacionadas a dados pessoais.
"""

import re
from logging import Logger

from anonimizar._constants import (
    MIN_TABLE_LINES,
    MIN_TABLE_PIPE_COUNT,
    PERSONAL_DATA_TABLE_KEYWORDS,
)
from anonimizar._validators import valida_cnh as _valida_cnh
from anonimizar._validators import valida_cpf as _valida_cpf

#: Labels aceitos após a classificação de células em tabelas Markdown. Este
#: conjunto funciona como filtro interno e não deve ser alterado in-place.
VALID_TABLE_LABELS = {
    "CPF",
    "CNH",
    "RG",
    "PASSAPORTE",
    "TITULO_ELEITOR",
    "SIAPE",
    "EMAIL",
    "TELEFONE",
    "ENDEREÇO",
    "FISTEL",
    "N DE IDENTIFICACAO PESSOAL",
    "N DE IDENTIFICAÇÃO PESSOAL",
    "N DE IDENTIFICACAO",
    "N DE IDENTIFICAÇÃO",
    "DOCUMENTO",
}


def _has_minimum_numbers(text: str, min_count: int = 2) -> bool:
    """Verifica se o texto contém pelo menos min_count dígitos numéricos.

    Args:
        text: Texto a verificar
        min_count: Quantidade mínima de dígitos requerida

    Returns:
        True se contém pelo menos min_count dígitos
    """
    return len(re.findall(r"\d", text)) >= min_count


def _adjust_entity_label(entity: dict) -> dict | None:  # noqa: PLR0911
    """Ajusta o label da entidade baseado em validação de conteúdo.

    Args:
        entity (dict): Dicionário da entidade com pelo menos 'label' e 'text'

    Returns:
        Entidade com label ajustado conforme validação, ou None se não for válida
    """
    label = entity.get("label", "")
    cell_text = entity.get("text", "")

    if not cell_text:
        return entity if label in VALID_TABLE_LABELS else None

    if _valida_cpf(cell_text):
        if label == "FISTEL":
            entity["label"] = "CPF"
        else:
            entity["label"] = "CPF"
        return entity
    if _valida_cnh(cell_text):
        entity["label"] = "CNH"
        return entity

    if label == "FISTEL":
        return None

    if label == "CNPJ":
        return None

    no_number_check_labels = {"EMAIL", "TELEFONE", "ENDEREÇO", "ENDERECO"}
    if label not in no_number_check_labels and not _has_minimum_numbers(cell_text, min_count=2):
        return None

    return entity if label in VALID_TABLE_LABELS else None


def _parse_table_header(header_line: str) -> list[str]:
    """Extrai nomes de colunas da linha de cabeçalho de uma tabela.

    Args:
        header_line (str): Linha de cabeçalho da tabela markdown

    Returns:
        list[str]: Lista de nomes de colunas (strings vazias removidas)
    """
    return [h.strip() for h in header_line.split("|") if h.strip()]


def _clean_header_numbering(header_text: str) -> str:
    """Remove numeração como '1.1.', '1.2.', '2.3.1' do início do texto.

    Args:
        header_text: Texto do cabeçalho da coluna

    Returns:
        Texto sem a numeração inicial
    """
    return re.sub(r"^\d+(\.\d+)*\.\s*", "", header_text)


def _find_personal_data_columns(header_cols: list[str]) -> list[tuple[int, str]]:
    """Identifica colunas com dados pessoais no cabeçalho da tabela.

    Args:
        header_cols (list[str]): Lista de nomes de colunas

    Returns:
        list[tuple[int, str]]: Lista de tuplas (índice, label) para colunas com dados pessoais
    """
    personal_data_cols = []
    for idx, col_name in enumerate(header_cols):
        cleaned = _clean_header_numbering(col_name)
        if any(keyword in cleaned.lower() for keyword in PERSONAL_DATA_TABLE_KEYWORDS):
            personal_data_cols.append((idx, cleaned.upper()))
    return personal_data_cols


def _parse_table_row(row_line: str) -> list[str]:
    """Extrai células de uma linha de tabela markdown.

    Args:
        row_line (str): Linha de dados da tabela markdown

    Returns:
        list[str]: Lista de valores de células (inclui células vazias para manter estrutura)
    """
    cells = row_line.split("|")
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return [c.strip() for c in cells]


def _create_entity_dict(
    label: str,
    cell_text: str,
    start_pos: int,
    return_type: str,
) -> dict:
    """Cria dicionário de entidade com base no tipo de retorno solicitado.

    Args:
        label (str): Rótulo da entidade (ex: "CPF", "RG")
        cell_text (str): Texto da célula
        start_pos (int): Posição inicial no texto original
        return_type (str): Formato de retorno ("label_position", "label_text", "label_detail")

    Returns:
        dict: Dicionário de entidade no formato solicitado
    """
    end_pos = start_pos + len(cell_text)

    if return_type == "label_position":
        return {
            "label": label,
            "start_position": start_pos,
            "end_position": end_pos,
        }
    if return_type == "label_text":
        return {"label": label, "text": cell_text}
    # label_detail
    return {
        "label": label,
        "start_position": start_pos,
        "end_position": end_pos,
        "text": cell_text,
        "detected_by": "tabela_markdown",
    }


def _extract_table_lines(lines: list[str], start_idx: int) -> tuple[list[str], int]:
    """Extrai linhas consecutivas que formam uma tabela markdown.

    Args:
        lines (list[str]): Lista de todas as linhas do texto
        start_idx (int): Índice da primeira linha da tabela

    Returns:
        tuple[list[str], int]: Tupla contendo (linhas_da_tabela, próximo_índice)
    """
    tb_lines = []
    j = start_idx

    while j < len(lines) and "|" in lines[j]:
        tb_lines.append(lines[j])
        j += 1

    return tb_lines, j


def _is_separator_line(line: str) -> bool:
    """Verifica se a linha é um separador de tabela markdown (---|---).

    Args:
        line: Linha a verificar

    Returns:
        True se for uma linha de separador
    """
    cleaned = line.replace("|", "").replace("-", "").replace(":", "").strip()
    return len(cleaned) == 0


def _detect_table_format(tb_lines: list[str]) -> tuple[bool, list[int]]:
    """Detecta o formato da tabela (padrão ou alternado).

    Args:
        tb_lines: Linhas da tabela

    Returns:
        Tupla (has_separator, header_indices) onde:
        - has_separator: True se a tabela tem linha separadora (---)
        - header_indices: Lista de índices que são headers
    """
    if len(tb_lines) < 2:  # noqa: PLR2004
        return False, [0] if tb_lines else []

    has_separator = _is_separator_line(tb_lines[1])

    header_indices = list(range(0, len(tb_lines), 2))

    return has_separator, header_indices


def _process_table(  # noqa: C901, PLR0912
    tb_lines: list[str],
    text: str,
    return_type: str,
    logger: Logger,
    labels: set[str] | None = None,
) -> list[dict]:
    """Processa uma tabela markdown e extrai entidades de colunas com dados pessoais.

    Args:
        tb_lines (list[str]): Linhas da tabela markdown
        text (str): Texto original completo (para encontrar posições)
        return_type (str): Formato de retorno das entidades
        logger (Logger): Logger para registrar erros
        labels (set[str] | None): Labels permitidos para filtragem

    Returns:
        list[dict]: Lista de entidades encontradas na tabela
    """
    entities = []
    search_start = 0  # Busca incremental para evitar posições duplicadas (B-18)

    try:
        has_separator = _is_separator_line(tb_lines[1]) if len(tb_lines) > 1 else False

        if has_separator:
            if len(tb_lines) >= 4 and _is_separator_line(tb_lines[3]):  # noqa: PLR2004
                header_line_indices = list(range(0, len(tb_lines), 2))
            else:
                header_line_indices = [0]
        else:
            header_line_indices = list(range(0, len(tb_lines), 2))

        for header_line_idx in header_line_indices:
            if header_line_idx >= len(tb_lines):
                break

            header_cols = _parse_table_header(tb_lines[header_line_idx])
            personal_data_cols = _find_personal_data_columns(header_cols)

            if personal_data_cols:
                data_line_idx = header_line_idx + 1

                if data_line_idx < len(tb_lines):
                    if has_separator and _is_separator_line(tb_lines[data_line_idx]):
                        data_line_idx += 1

                    while data_line_idx < len(tb_lines):
                        row_line = tb_lines[data_line_idx]

                        if not row_line.strip().startswith("|"):
                            break

                        row_cells = _parse_table_row(row_line)
                        for col_idx, label in personal_data_cols:
                            if col_idx < len(row_cells):
                                cell_text = row_cells[col_idx]
                                if cell_text:
                                    start_pos = text.find(cell_text, search_start)
                                    if start_pos != -1:
                                        search_start = start_pos + len(cell_text)
                                        entity = _create_entity_dict(
                                            label,
                                            cell_text,
                                            start_pos,
                                            return_type,
                                        )
                                        adjusted = _adjust_entity_label(entity)
                                        if adjusted is not None:
                                            entities.append(adjusted)

                        if len(header_line_indices) > 1:
                            break

                        data_line_idx += 1

    except (IndexError, ValueError, AttributeError) as e:
        logger.debug("Erro ao processar tabela: %s", e)

    if labels is not None:
        entities = [e for e in entities if e.get("label") in labels]

    return entities


def extract_entities_from_markdown_tables(
    text: str,
    return_type: str,
    logger: Logger,
    labels: set[str] | None = None,
) -> list[dict]:
    """Extrai entidades de colunas com dados pessoais em tabelas markdown.

    Detecta automaticamente tabelas markdown no texto e extrai entidades de colunas
    que contenham nomes relacionados a dados pessoais (CPF, RG, Titulo, Documento).

    Args:
        text (str): Texto contendo tabelas markdown
        return_type (str): Formato de retorno das entidades
        logger (Logger): Logger para registrar informações de debug
        labels (set[str] | None): Labels permitidos para filtragem

    Returns:
        list[dict]: Lista de entidades encontradas nas tabelas
    """
    entities = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if "|" in line and line.count("|") >= MIN_TABLE_PIPE_COUNT:
            tb_lines, next_idx = _extract_table_lines(lines, i)

            if len(tb_lines) >= MIN_TABLE_LINES:
                table_entities = _process_table(tb_lines, text, return_type, logger, labels)
                entities.extend(table_entities)

            i = next_idx
        else:
            i += 1

    logger.debug("Entidades extraídas de tabelas markdown: %d", len(entities))
    return entities
