"""Testes das melhorias de recall/privacidade.

Cobre: ``detected_by`` serializável, pontuação final de ENDEREÇO, modo
``prefer_recall`` (CPF com DV inválido em contexto explícito e telefone em
formato forte sem palavra-chave) e detecção opcional de nomes (``NOME``).
"""

import json
from itertools import pairwise

import pytest
import spacy
from spacy.tokens import Doc

from anonimizar import Anonimizar
from anonimizar._extraction.names import extract_names
from anonimizar._extraction.regex import extract_entities_regex_re
from anonimizar._normalization import normalize_entity
from anonimizar._validators.context import validate_cpf_context, validate_telefone_context
from tests._helpers import _find_span, _make_logger

NOMES_MODEL = "pt_core_news_sm"
requer_modelo_nomes = pytest.mark.skipif(
    not spacy.util.is_package(NOMES_MODEL),
    reason=f"modelo spaCy '{NOMES_MODEL}' não instalado",
)


# ─── detected_by serializável ─────────────────────────────────────────────────


class TestDetectedByRegex:
    """``detected_by`` de entidades por regex deve ser texto, não ``re.Pattern``."""

    def test_regex_extractor_devolve_string(self):
        texto = "CPF 52998224725 do titular."
        patterns = [{"label": "CPF", "pattern": {"REGEX": r"(?<!\d)\d{11}(?!\d)"}}]
        result = extract_entities_regex_re(texto, patterns, "label_detail", lambda *_: True, _make_logger())
        assert len(result) == 1
        assert result[0]["detected_by"] == "regex"

    def test_label_detail_e_serializavel_em_json(self, model_path):
        anonymizer = Anonimizar(model_path=model_path)
        entidades = anonymizer.extract_entities("CPF 52998224725 do titular.", return_type="label_detail")
        assert entidades, "esperava ao menos uma entidade"
        json.dumps(entidades)  # não deve levantar TypeError


# ─── pontuação final de ENDEREÇO ──────────────────────────────────────────────


class TestEnderecoPontuacaoFinal:
    """O ponto final da frase não faz parte do endereço."""

    def test_remove_ponto_final(self):
        texto = "Rua das Flores, 100, CEP 70000-000, Brasília."
        limpo, start, end, regra = normalize_entity(texto, 10, 10 + len(texto), "ENDEREÇO")
        assert limpo == "Rua das Flores, 100, CEP 70000-000, Brasília"
        assert (start, end) == (10, 10 + len(texto) - 1)
        assert regra is not None

    @pytest.mark.parametrize("sufixo", [".", ",", ";", ":", " ."])
    def test_remove_pontuacao_variada(self, sufixo):
        base = "Av. Paulista, 1000"
        limpo, _, end, _ = normalize_entity(base + sufixo, 0, len(base + sufixo), "ENDEREÇO")
        assert limpo == base
        assert end == len(base)

    def test_sem_pontuacao_nao_altera(self):
        texto = "Rua das Flores, 100"
        limpo, start, end, regra = normalize_entity(texto, 5, 5 + len(texto), "ENDEREÇO")
        assert (limpo, start, end, regra) == (texto, 5, 5 + len(texto), None)

    def test_pontuacao_interna_preservada(self):
        texto = "Rua A, 10, Brasília"
        limpo, *_ = normalize_entity(texto, 0, len(texto), "ENDEREÇO")
        assert limpo == texto

    def test_nao_afeta_outros_labels(self):
        limpo, *_ = normalize_entity("a@b.com.", 0, 8, "EMAIL")
        assert limpo == "a@b.com."


# ─── prefer_recall: CPF ───────────────────────────────────────────────────────


class TestPreferRecallCpf:
    """CPF com DV inválido, mas com a palavra CPF por perto, é digitação errada, não "não-CPF"."""

    DV_INVALIDO = "Meu CPF é 123.456.789-00 para cadastro."

    def _valida(self, texto: str, trecho: str, *, prefer_recall: bool) -> bool | str:
        start, end = _find_span(texto, trecho)
        return validate_cpf_context(texto, start, end, trecho, _make_logger(), prefer_recall=prefer_recall)

    def test_padrao_rejeita_dv_invalido(self):
        assert self._valida(self.DV_INVALIDO, "123.456.789-00", prefer_recall=False) is False

    def test_prefer_recall_aceita_dv_invalido_com_contexto(self):
        assert self._valida(self.DV_INVALIDO, "123.456.789-00", prefer_recall=True) is True

    def test_prefer_recall_continua_rejeitando_sem_contexto(self):
        texto = "O número 123.456.789-00 foi anotado no papel."
        assert self._valida(texto, "123.456.789-00", prefer_recall=True) is False

    def test_prefer_recall_so_aceita_11_digitos(self):
        texto = "Meu CPF é 123.456.789-0 para cadastro."
        assert self._valida(texto, "123.456.789-0", prefer_recall=True) is False

    def test_cpf_valido_inalterado(self):
        texto = "Meu CPF é 529.982.247-25 para cadastro."
        assert self._valida(texto, "529.982.247-25", prefer_recall=False) is True
        assert self._valida(texto, "529.982.247-25", prefer_recall=True) is True

    def test_integracao_anonimizar(self, model_path):
        padrao = Anonimizar(model_path=model_path)
        recall = Anonimizar(model_path=model_path, prefer_recall=True)
        assert padrao.prefer_recall is False
        assert recall.prefer_recall is True
        assert not [e for e in padrao.extract_entities(self.DV_INVALIDO) if e["label"] == "CPF"]
        assert [e for e in recall.extract_entities(self.DV_INVALIDO) if e["label"] == "CPF"]


# ─── prefer_recall: TELEFONE ──────────────────────────────────────────────────


class TestPreferRecallTelefone:
    """Telefone em formato inequívoco é PII mesmo sem palavra-chave por perto."""

    def _valida(self, texto: str, trecho: str, *, prefer_recall: bool) -> bool:
        start, end = _find_span(texto, trecho)
        return validate_telefone_context(texto, start, end, trecho, _make_logger(), prefer_recall=prefer_recall)

    @pytest.mark.parametrize(
        "trecho",
        ["(61) 99999-1234", "(61) 3333-4444", "(61)99999-1234", "+55 61 99999-1234", "+55 (61) 99999-1234"],
    )
    def test_formato_forte_sem_palavra_chave(self, trecho):
        texto = f"O número {trecho} consta no cadastro."
        assert self._valida(texto, trecho, prefer_recall=False) is False
        assert self._valida(texto, trecho, prefer_recall=True) is True

    @pytest.mark.parametrize("trecho", ["1234567890", "1234-5678", "61 99999 1234", "12345678901"])
    def test_formato_ambiguo_continua_exigindo_contexto(self, trecho):
        texto = f"O código {trecho} consta no cadastro."
        assert self._valida(texto, trecho, prefer_recall=True) is False

    def test_com_palavra_chave_inalterado(self):
        texto = "Telefone: 1234567890."
        assert self._valida(texto, "1234567890", prefer_recall=False) is True
        assert self._valida(texto, "1234567890", prefer_recall=True) is True

    def test_integracao_anonimizar(self, model_path):
        texto = "Ligue para (61) 99999-1234 amanhã."
        padrao = Anonimizar(model_path=model_path)
        recall = Anonimizar(model_path=model_path, prefer_recall=True)
        assert not [e for e in padrao.extract_entities(texto) if e["label"] == "TELEFONE"]
        assert [e for e in recall.extract_entities(texto) if e["label"] == "TELEFONE"]


# ─── detecção opcional de nomes ───────────────────────────────────────────────


def _doc(words: list[str], spaces: list[bool], pos: list[str], ents: list[str]) -> Doc:
    return Doc(spacy.blank("pt").vocab, words=words, spaces=spaces, pos=pos, ents=ents)


class TestExtractNames:
    """Unidade: conversão de ``PER`` do spaCy em ``NOME`` com filtro de falsos positivos."""

    def test_nome_completo(self):
        doc = _doc(
            ["Ana", "Souza", "assinou", "."],
            [True, True, False, False],
            ["PROPN", "PROPN", "VERB", "PUNCT"],
            ["B-PER", "I-PER", "O", "O"],
        )
        result = extract_names(lambda _t: doc, doc.text, _make_logger())
        assert result == [
            {
                "label": "NOME",
                "start_position": 0,
                "end_position": 9,
                "text": "Ana Souza",
                "detected_by": "modelo_nomes",
            }
        ]

    def test_primeiro_nome_isolado_proprio_e_mantido(self):
        doc = _doc(
            ["Falei", "com", "a", "Marta", "."],
            [True, True, True, False, False],
            ["VERB", "ADP", "DET", "PROPN", "PUNCT"],
            ["O", "O", "O", "B-PER", "O"],
        )
        result = extract_names(lambda _t: doc, doc.text, _make_logger())
        assert [r["text"] for r in result] == ["Marta"]

    @pytest.mark.parametrize(("palavra", "pos"), [("Encaminhe", "VERB"), ("Despacho", "NOUN")])
    def test_palavra_comum_marcada_como_pessoa_e_descartada(self, palavra, pos):
        doc = _doc(
            [palavra, "ao", "setor", "."],
            [True, True, False, False],
            [pos, "ADP", "NOUN", "PUNCT"],
            ["B-PER", "O", "O", "O"],
        )
        assert extract_names(lambda _t: doc, doc.text, _make_logger()) == []

    def test_ignora_outros_labels(self):
        doc = _doc(["Anatel", "decidiu", "."], [True, False, False], ["PROPN", "VERB", "PUNCT"], ["B-ORG", "O", "O"])
        assert extract_names(lambda _t: doc, doc.text, _make_logger()) == []

    def test_modelo_sem_pos_mantem_por_seguranca(self):
        doc = Doc(
            spacy.blank("pt").vocab,
            words=["Marta", "chegou", "."],
            spaces=[True, False, False],
            ents=["B-PER", "O", "O"],
        )
        result = extract_names(lambda _t: doc, doc.text, _make_logger())
        assert [r["text"] for r in result] == ["Marta"]


@requer_modelo_nomes
class TestNomesIntegracao:
    """Integração com o ``Anonimizar`` e um modelo spaCy genérico."""

    TEXTO = "A servidora Maria Aparecida Oliveira Santos informou o CPF 529.982.247-25."

    def test_desligado_por_padrao(self, model_path):
        entidades = Anonimizar(model_path=model_path).extract_entities(self.TEXTO)
        assert "NOME" not in {e["label"] for e in entidades}

    def test_ligado_detecta_nome_e_mantem_cpf(self, model_path):
        anonymizer = Anonimizar(model_path=model_path, names_model=NOMES_MODEL)
        entidades = anonymizer.extract_entities(self.TEXTO)
        por_label = {e["label"]: self.TEXTO[e["start_position"] : e["end_position"]] for e in entidades}
        assert por_label["NOME"] == "Maria Aparecida Oliveira Santos"
        assert por_label["CPF"] == "529.982.247-25"

    def test_anonymize_text_troca_nome_por_tag(self, model_path):
        anonymizer = Anonimizar(model_path=model_path, names_model=NOMES_MODEL)
        entidades = anonymizer.extract_entities(self.TEXTO)
        assert anonymizer.anonymize_text(self.TEXTO, entidades) == "A servidora <|NOME|> informou o CPF <|CPF|>."

    def test_nome_nao_sobrepoe_endereco(self, model_path):
        texto = "Mora na Rua Marechal Deodoro, 100, CEP 70000-000, Brasília."
        anonymizer = Anonimizar(model_path=model_path, names_model=NOMES_MODEL)
        spans = sorted((e["start_position"], e["end_position"]) for e in anonymizer.extract_entities(texto))
        assert all(a_end <= b_start for (_, a_end), (b_start, _) in pairwise(spans))

    def test_label_detail_serializavel(self, model_path):
        anonymizer = Anonimizar(model_path=model_path, names_model=NOMES_MODEL)
        json.dumps(anonymizer.extract_entities(self.TEXTO, return_type="label_detail"))
