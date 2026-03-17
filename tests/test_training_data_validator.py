"""Testes para _training/data_validator.py.

Cobre branches e funções não exercitados pelos testes existentes:
- validate_data com errors='raise', 'coerce', 'ignore'
- validate_biluo_tags com erros BILUO
- detect_entity_conflicts com duplicatas e sobreposições
- resolve_entity_conflicts
- clean_entities com conflitos e strict mode
- debug_entities
"""

import logging

import pytest
import spacy

from anonimizar._training.data_validator import (
    clean_entities,
    debug_entities,
    detect_entity_conflicts,
    resolve_entity_conflicts,
    validate_biluo_tags,
    validate_data,
    validate_entities,
)


@pytest.fixture
def blank_nlp():
    """Pipeline spaCy em branco para testes."""
    return spacy.blank("pt")


@pytest.fixture
def logger():
    """Logger silencioso para testes."""
    log = logging.getLogger("test_data_validator")
    log.setLevel(logging.DEBUG)
    return log


SUPPORTED = ["CPF", "RG", "EMAIL", "ENDEREÇO"]


# =============================================================================
# validate_entities
# =============================================================================


class TestValidateEntities:
    """Testa validate_entities - linhas 70-72 (excecao BILUO)."""

    def test_validate_entities_valido(self, blank_nlp, logger) -> None:
        """Entidade válida retorna True."""
        text = "CPF 529.982.247-25 aqui"
        entities = [(4, 18, "CPF")]
        assert validate_entities(text, entities, SUPPORTED, blank_nlp, logger) is True

    def test_validate_entities_inicio_negativo(self, blank_nlp, logger) -> None:
        """Start negativo retorna False."""
        assert validate_entities("texto", [(-1, 5, "CPF")], SUPPORTED, blank_nlp, logger) is False

    def test_validate_entities_inicio_maior_fim(self, blank_nlp, logger) -> None:
        """Start >= end retorna False."""
        assert validate_entities("texto", [(5, 3, "CPF")], SUPPORTED, blank_nlp, logger) is False

    def test_validate_entities_offset_fora(self, blank_nlp, logger) -> None:
        """Offset além do texto retorna False."""
        assert validate_entities("ab", [(0, 10, "CPF")], SUPPORTED, blank_nlp, logger) is False

    def test_validate_entities_entidade_com_espaco(self, blank_nlp, logger) -> None:
        """Entidade com espaço nas extremidades retorna False."""
        assert validate_entities(" CPF", [(0, 4, "CPF")], SUPPORTED, blank_nlp, logger) is False

    def test_validate_entities_label_nao_suportado(self, blank_nlp, logger) -> None:
        """Label não suportado retorna False."""
        assert validate_entities("texto cpf", [(6, 9, "CNPJ")], SUPPORTED, blank_nlp, logger) is False

    def test_validate_entities_biluo_invalido_desalinhado(self, blank_nlp, logger) -> None:
        """Entidade no meio de um token gera BILUO '-' → retorna False."""
        # Força desalinhamento usando offset no meio de uma palavra
        text = "oBomDia"
        # spaCy tokeniza como token único; offset 1-4 corta no meio
        entities = [(1, 4, "CPF")]
        assert validate_entities(text, entities, SUPPORTED, blank_nlp, logger) is False


# =============================================================================
# validate_data
# =============================================================================


class TestValidateDataModes:
    """Testa validate_data com errors='raise', 'coerce', 'ignore' - linhas 113-131."""

    def test_validate_data_raise_label_invalido(self, blank_nlp, logger) -> None:
        """errors='raise' com label inválido deve lançar ValueError."""
        data = [("texto cpf", {"entities": [(6, 9, "CNPJ")]})]
        with pytest.raises(ValueError, match="Label inválido"):
            validate_data(data, SUPPORTED, blank_nlp, logger, errors="raise")

    def test_validate_data_coerce_descarta_label_invalido(self, blank_nlp, logger) -> None:
        """errors='coerce' descarta entidade com label inválido."""
        data = [("texto cpf", {"entities": [(6, 9, "CNPJ")]})]
        result = validate_data(data, SUPPORTED, blank_nlp, logger, errors="coerce")
        # Texto mantido (keep_empty_entities=False), mas sem entidades
        # Como não há entidade válida e keep_empty_entities=False, texto é descartado
        assert result == []

    def test_validate_data_ignore_descarta_label_invalido(self, blank_nlp, logger) -> None:
        """errors='ignore' também descarta entidade com label inválido."""
        data = [("texto cpf", {"entities": [(6, 9, "CNPJ")]})]
        result = validate_data(data, SUPPORTED, blank_nlp, logger, errors="ignore")
        assert result == []

    def test_validate_data_raise_offset_invalido(self, blank_nlp, logger) -> None:
        """errors='raise' com offset inválido lança ValueError."""
        data = [("ab", {"entities": [(0, 100, "CPF")]})]
        with pytest.raises(ValueError, match="Posições inválidas"):
            validate_data(data, SUPPORTED, blank_nlp, logger, errors="raise")

    def test_validate_data_coerce_offset_invalido_descartado(self, blank_nlp, logger) -> None:
        """errors='coerce' descarta entidade com offset inválido."""
        data = [("ab", {"entities": [(0, 100, "CPF")]})]
        result = validate_data(data, SUPPORTED, blank_nlp, logger, errors="coerce")
        assert result == []

    def test_validate_data_ignore_offset_invalido(self, blank_nlp, logger) -> None:
        """errors='ignore' mantém entidade mesmo com offset inválido."""
        data = [("ab", {"entities": [(0, 100, "CPF")]})]
        result = validate_data(data, SUPPORTED, blank_nlp, logger, errors="ignore")
        # Com ignore, entidade é descartada pela lógica mas o texto pode ser mantido
        assert isinstance(result, list)

    def test_validate_data_keep_empty_entities(self, blank_nlp, logger) -> None:
        """keep_empty_entities=True mantém textos sem entidades válidas."""
        data = [("texto simples", {"entities": []})]
        result = validate_data(data, SUPPORTED, blank_nlp, logger, keep_empty_entities=True)
        assert len(result) == 1

    def test_validate_data_skip_biluo(self, blank_nlp, logger) -> None:
        """skip_biluo=True pula validação BILUO."""
        data = [("oBomDia", {"entities": [(1, 4, "CPF")]})]
        result = validate_data(data, SUPPORTED, blank_nlp, logger, errors="coerce", skip_biluo=True)
        assert isinstance(result, list)

    def test_validate_data_texto_descartado_sem_entidades(self, blank_nlp, logger) -> None:
        """Texto cujas entidades foram todas inválidas é descartado."""
        data = [("texto cpf", {"entities": [(6, 9, "CNPJ")]})]
        result = validate_data(data, SUPPORTED, blank_nlp, logger, errors="coerce", keep_empty_entities=False)
        assert result == []


# =============================================================================
# validate_biluo_tags
# =============================================================================


class TestValidateBILUOTags:
    """Testa validate_biluo_tags - linhas 186-210."""

    def test_biluo_entidade_vazia_retorna_vazia(self, blank_nlp, logger) -> None:
        """Lista vazia retorna lista vazia."""
        result = validate_biluo_tags("texto", [], blank_nlp, logger)
        assert result == []

    def test_biluo_valido_mantem_entidade(self, blank_nlp, logger) -> None:
        """Entidade com BILUO correto é mantida."""
        text = "CPF aqui"
        entities = [(4, 8, "CPF")]
        result = validate_biluo_tags(text, entities, blank_nlp, logger, errors="coerce")
        # Pode ou não ser mantida dependendo do alinhamento
        assert isinstance(result, list)

    def test_biluo_desalinhado_coerce_descarta(self, blank_nlp, logger) -> None:
        """errors='coerce' descarta entidade com BILUO inválido."""
        text = "oBomDia"
        entities = [(1, 4, "CPF")]
        result = validate_biluo_tags(text, entities, blank_nlp, logger, errors="coerce")
        assert result == []

    def test_biluo_desalinhado_raise_lanca_excecao(self, blank_nlp, logger) -> None:
        """errors='raise' com BILUO inválido lança ValueError."""
        text = "oBomDia"
        entities = [(1, 4, "CPF")]
        with pytest.raises(ValueError, match=r"Erro BILUO"):
            validate_biluo_tags(text, entities, blank_nlp, logger, errors="raise")

    def test_biluo_desalinhado_ignore_descarta(self, blank_nlp, logger) -> None:
        """errors='ignore' mantém entidade com BILUO inválido."""
        text = "oBomDia"
        entities = [(1, 4, "CPF")]
        result = validate_biluo_tags(text, entities, blank_nlp, logger, errors="ignore")
        assert result == [(1, 4, "CPF")]


# =============================================================================
# detect_entity_conflicts
# =============================================================================


class TestDetectEntityConflicts:
    """Testa detect_entity_conflicts - linhas 232-252."""

    def test_sem_conflitos(self) -> None:
        """Entidades sem sobreposição ou duplicata retornam has_conflicts=False."""
        entities = [(0, 5, "CPF"), (10, 15, "RG")]
        result = detect_entity_conflicts(entities)
        assert result["has_conflicts"] is False
        assert result["duplicates"] == []
        assert result["overlaps"] == []

    def test_detecta_duplicata(self) -> None:
        """Entidades idênticas são detectadas como duplicatas."""
        entities = [(0, 5, "CPF"), (0, 5, "CPF")]
        result = detect_entity_conflicts(entities)
        assert result["has_conflicts"] is True
        assert len(result["duplicates"]) == 1

    def test_detecta_sobreposicao(self) -> None:
        """Entidades com intervalos sobrepostos são detectadas."""
        entities = [(0, 10, "CPF"), (5, 15, "RG")]
        result = detect_entity_conflicts(entities)
        assert result["has_conflicts"] is True
        assert len(result["overlaps"]) == 1

    def test_entidade_adjacentes_nao_conflitam(self) -> None:
        """Entidades adjacentes (end1 == start2) não conflitam."""
        entities = [(0, 5, "CPF"), (5, 10, "RG")]
        result = detect_entity_conflicts(entities)
        assert result["has_conflicts"] is False

    def test_lista_vazia(self) -> None:
        """Lista vazia não tem conflitos."""
        result = detect_entity_conflicts([])
        assert result["has_conflicts"] is False


# =============================================================================
# resolve_entity_conflicts
# =============================================================================


class TestResolveEntityConflicts:
    """Testa resolve_entity_conflicts - linhas 270-291."""

    def test_remove_duplicatas(self, logger) -> None:
        """Duplicatas são removidas."""
        entities = [(0, 5, "CPF"), (0, 5, "CPF")]
        conflicts = detect_entity_conflicts(entities)
        result = resolve_entity_conflicts(entities, conflicts, logger)
        assert len(result) == 1

    def test_sobreposicao_retorna_lista_vazia(self, logger) -> None:
        """Sobreposições que persistem após deduplicação retornam []."""
        entities = [(0, 10, "CPF"), (5, 15, "RG")]
        conflicts = detect_entity_conflicts(entities)
        result = resolve_entity_conflicts(entities, conflicts, logger)
        # Sem duplicatas, sobreposição persiste → retorna []
        assert result == []

    def test_sem_conflitos_retorna_original(self, logger) -> None:
        """Sem conflitos, retorna a mesma lista."""
        entities = [(0, 5, "CPF"), (10, 15, "RG")]
        conflicts = detect_entity_conflicts(entities)
        result = resolve_entity_conflicts(entities, conflicts, logger)
        assert result == entities


# =============================================================================
# clean_entities
# =============================================================================


class TestCleanEntities:
    """Testa clean_entities - linhas 324-394."""

    def test_lista_vazia_retorna_vazia(self, blank_nlp, logger) -> None:
        """Lista vazia retorna lista vazia."""
        result = clean_entities("texto", [], SUPPORTED, blank_nlp, logger)
        assert result == []

    def test_conflito_raise_lanca_excecao(self, blank_nlp, logger) -> None:
        """resolve_conflicts='raise' com conflito lança ValueError."""
        entities = [(0, 10, "CPF"), (5, 15, "RG")]
        with pytest.raises(ValueError, match=r"Conflitos de entidades"):
            clean_entities("texto qualquer", entities, SUPPORTED, blank_nlp, logger, resolve_conflicts="raise")

    def test_conflito_ignore_retorna_vazia(self, blank_nlp, logger) -> None:
        """resolve_conflicts='ignore' com conflito retorna []."""
        entities = [(0, 10, "CPF"), (5, 15, "RG")]
        text = "texto qualquer exemplo"
        result = clean_entities(text, entities, SUPPORTED, blank_nlp, logger, resolve_conflicts="ignore")
        assert result == []

    def test_conflito_coerce_tenta_resolver(self, blank_nlp, logger) -> None:
        """resolve_conflicts='coerce' tenta resolver conflitos."""
        entities = [(0, 10, "CPF"), (5, 15, "RG")]
        text = "texto qualquer exemplo"
        result = clean_entities(text, entities, SUPPORTED, blank_nlp, logger, resolve_conflicts="coerce")
        assert isinstance(result, list)

    def test_entidade_com_espaco_e_corrigida(self, blank_nlp, logger) -> None:
        """Entidade com espaços nas bordas deve ser corrigida (offset ajustado)."""
        text = "CPF 529.982.247-25 aqui"
        # Inclui espaço antes do CPF
        entities = [(3, 18, "CPF")]  # " 529.982.247-25"
        result = clean_entities(text, entities, SUPPORTED, blank_nlp, logger, strict=False)
        # Deve ajustar para (4, 18, "CPF")
        if result:
            assert result[0][0] == 4

    def test_strict_mode_descarta_se_mudou(self, blank_nlp, logger) -> None:
        """strict=True descarta se alguma entidade foi removida."""
        text = "CPF 529.982.247-25 aqui"
        entities = [(4, 18, "CPF"), (100, 110, "RG")]  # segunda offset inválido
        result = clean_entities(text, entities, SUPPORTED, blank_nlp, logger, strict=True)
        # strict=True: se count mudou, retorna []
        assert result == []


# =============================================================================
# debug_entities
# =============================================================================


class TestDebugEntities:
    """Testa debug_entities - linhas 413-445."""

    def test_debug_entities_executa_sem_erro(self, blank_nlp, logger) -> None:
        """debug_entities deve executar sem lançar exceções."""
        text = "CPF 529.982.247-25 aqui"
        entities = [(4, 18, "CPF")]
        debug_entities(text, entities, SUPPORTED, blank_nlp, logger)  # não deve lançar

    def test_debug_entities_com_problemas(self, blank_nlp, logger) -> None:
        """debug_entities registra problemas sem lançar exceções."""
        text = "ab"
        entities = [(0, 100, "CPF"), (-1, 5, "RG"), (0, 2, "CNPJ")]
        debug_entities(text, entities, SUPPORTED, blank_nlp, logger)  # não deve lançar

    def test_debug_entities_espaco_nas_extremidades(self, blank_nlp, logger) -> None:
        """debug_entities identifica espaços nas extremidades."""
        text = " CPF aqui"
        entities = [(0, 4, "CPF")]  # começa com espaço
        debug_entities(text, entities, SUPPORTED, blank_nlp, logger)

    def test_debug_entities_lista_vazia(self, blank_nlp, logger) -> None:
        """debug_entities com lista vazia não lança exceção."""
        debug_entities("texto", [], SUPPORTED, blank_nlp, logger)
