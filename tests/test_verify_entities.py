import pytest

from anonimizar import Anonimizar
from anonimizar._validators.context import (
    validate_cnh_context,
    validate_email_context,
    validate_rg_context,
    validate_siape_context,
)
from tests._helpers import _find_span, _make_logger, make_ent

# =============================================================================
# CPF / CNPJ
# =============================================================================


class TestVerifyCpf:
    """Testes de verificação para CPF/CNPJ."""

    def test_cpf_valid_with_context(self, model_path):
        anonymizer = Anonimizar(model_path)
        text = "meu cpf é 529.982.247-25"
        ent = make_ent("CPF", text, "529.982.247-25")
        assert anonymizer.verify_entities_unified(ent, text) is True

    def test_cpf_invalid_with_context(self, model_path):
        anonymizer = Anonimizar(model_path)
        text = "meu dado é 529.982.247-29"
        ent = make_ent("CPF", text, "529.982.247-29")
        assert anonymizer.verify_entities_unified(ent, text) is False

    def test_cnpj_valid(self, model_path):
        anonymizer = Anonimizar(model_path)
        text = "meu cpf é 18.781.203/0001-28"
        ent = make_ent("CPF", text, "18.781.203/0001-28")
        assert anonymizer.verify_entities_unified(ent, text) is False

    def test_cnpj_invalid(self, model_path):
        anonymizer = Anonimizar(model_path)
        text = "meu cpf é 18.781.203/0001-27"
        ent = make_ent("CPF", text, "18.781.203/0001-27")
        assert anonymizer.verify_entities_unified(ent, text) is False

    def test_cpf_invalid_with_validator(self, model_path):
        anonymizer = Anonimizar(model_path)
        text = "cpf 000.000.000-00"
        ent = make_ent("CPF", text, "000.000.000-00")
        assert anonymizer.verify_entities_unified(ent, text) is False

    def test_cpf_reclassifies_to_cnh_on_context(self, model_path):
        anonymizer = Anonimizar(model_path)
        text = "cnh 12345678909"
        ent = make_ent("CPF", text, "12345678909")
        assert anonymizer.verify_entities_unified(ent, text) == "CNH"


# =============================================================================
# RG
# =============================================================================


class TestVerifyRg:
    """Testes de verificação para RG (contexto, órgão emissor, fallback, P0)."""

    def test_rg_needs_context(self, model_path):
        anonymizer = Anonimizar(model_path)
        t1 = "12.345.678-9"
        ent1 = make_ent("RG", t1, t1)
        assert anonymizer.verify_entities_unified(ent1, t1) is False

        t_short = "1234"
        ent_short = make_ent("RG", t_short, t_short)
        assert anonymizer.verify_entities_unified(ent_short, t_short) is False

        t2 = "identidade 12.345.678-9"
        ent2 = make_ent("RG", t2, "12.345.678-9")
        assert anonymizer.verify_entities_unified(ent2, t2) is True

    # --- B-04: Órgão emissor ---

    def test_rg_com_ssp_aceito(self):
        text = "RG 12.345.678-9 SSP/SP"
        result = validate_rg_context(text, 3, 15, "12.345.678-9", _make_logger())
        assert result is True

    def test_rg_com_detran_aceito(self):
        text = "documento 12.345.678-9 DETRAN/RJ"
        result = validate_rg_context(text, 10, 22, "12.345.678-9", _make_logger())
        assert result is True

    def test_rg_com_igp_aceito(self):
        text = "identidade 12.345.678-9 IGP/RS"
        result = validate_rg_context(text, 11, 23, "12.345.678-9", _make_logger())
        assert result is True

    def test_rg_sem_contexto_e_sem_orgao_rejeita(self):
        text = "valor 12345678 aqui"
        result = validate_rg_context(text, 6, 14, "12345678", _make_logger())
        assert result is False

    def test_rg_prefixo_dois_pontos_com_uf_hifen_aceito(self):
        text = "RG: 19118939 - DF"
        result = validate_rg_context(text, 4, 17, "19118939 - DF", _make_logger())
        assert result is True

    # --- B-29: Fallback estrito ---

    def test_rg_pontuado_com_contexto_aceito(self):
        text = "identidade 1.234.567-8"
        result = validate_rg_context(text, 11, 22, "1.234.567-8", _make_logger())
        assert result is True

    def test_numero_solto_5_digitos_rejeitado(self):
        text = "valor 12345 reais"
        result = validate_rg_context(text, 6, 11, "12345", _make_logger())
        assert result is False

    def test_numero_solto_6_digitos_rejeitado(self):
        text = "valor 123456 reais"
        result = validate_rg_context(text, 6, 12, "123456", _make_logger())
        assert result is False

    def test_numero_solto_7_digitos_aceito(self):
        text = "documento 1234567"
        result = validate_rg_context(text, 10, 17, "1234567", _make_logger())
        assert result is True

    def test_numero_solto_9_digitos_rejeitado(self):
        text = "processo 123456789 aberto"
        result = validate_rg_context(text, 9, 18, "123456789", _make_logger())
        assert result is False

    def test_rg_formato_pontuado_sem_contexto_aceito(self):
        text = "documento 12.345.678-9"
        result = validate_rg_context(text, 10, 22, "12.345.678-9", _make_logger())
        assert result is True

    # --- P0 melhorias RG ---

    def test_rg_6_digitos_com_contexto_rg_aceito(self):
        text = "rg 418947"
        result = validate_rg_context(text, 3, 9, "418947", _make_logger())
        assert result is True

    def test_rg_6_digitos_com_contexto_identidade_aceito(self):
        text = "identidade 418947"
        result = validate_rg_context(text, 11, 17, "418947", _make_logger())
        assert result is True

    def test_rg_6_digitos_sem_contexto_rejeitado(self):
        text = "valor 418947 reais"
        result = validate_rg_context(text, 6, 12, "418947", _make_logger())
        assert result is False

    def test_rg_sufixo_uf_hifen_aceito(self):
        text = "rg 1993018 -DF"
        result = validate_rg_context(text, 3, 14, "1993018 -DF", _make_logger())
        assert result is True

    def test_rg_sufixo_uf_barra_aceito(self):
        text = "identidade 3182580/SP"
        result = validate_rg_context(text, 11, 21, "3182580/SP", _make_logger())
        assert result is True

    def test_rg_sufixo_uf_sem_espaco_aceito(self):
        text = "documento 3088061-DF emitido"
        result = validate_rg_context(text, 10, 20, "3088061-DF", _make_logger())
        assert result is True

    def test_rg_contexto_orgao_emissor_ssp_aceito(self):
        text = "ssp 1234567"
        result = validate_rg_context(text, 4, 11, "1234567", _make_logger())
        assert result is True

    def test_rg_contexto_detran_aceito(self):
        text = "detran 1234567"
        result = validate_rg_context(text, 7, 14, "1234567", _make_logger())
        assert result is True

    def test_rg_contexto_orgao_emissor_por_extenso_aceito(self):
        text = "órgão emissor: 1234567"
        result = validate_rg_context(text, 15, 22, "1234567", _make_logger())
        assert result is True

    # --- Regressão: formatos válidos e falsos positivos ---

    @pytest.mark.parametrize(
        ("full", "entity"),
        [
            ("RG\n\n11427720-5 DETRAN RJ", "11427720-5"),
            ("portadora do RG n 92013003102, SSP/CE", "92013003102"),
            ("RG: 61464229 SSP/SP", "61464229 SSP/SP"),
            ("RG 3.037.792-3/PR", "3.037.792-3"),
            ("RG 6567702-4 SSP/PR", "6567702-4"),
            ("| RG | 2588524 SSP SC |", "2588524 SSP SC"),
            ("Os\u00edas Fonseca\nRG: 1882069 SSP/DF.", "1882069 SSP/DF"),
            ("portador da c\u00e9dula de identidade n\u00ba 3.684.329 SSP/GO", "3.684.329"),
        ],
    )
    def test_validate_rg_accepts_valid_formats(self, full, entity):
        start, end = _find_span(full, entity)
        ok = validate_rg_context(full, start, end, entity, _make_logger())
        assert ok is True

    @pytest.mark.parametrize(
        ("full", "entity"),
        [
            ("160028 - 35 BATALHAO DE INFANTARIA", "160028 - 35"),
            ("19118939 - DF", "19118939 - DF"),
            ("58.896.796-3", "58.896.796-3"),
            ("5249269/DF", "5249269/DF"),
            ("3112662/DF", "3112662/DF"),
        ],
    )
    def test_validate_rg_rejects_common_false_positives(self, full, entity):
        start, end = _find_span(full, entity)
        ok = validate_rg_context(full, start, end, entity, _make_logger())
        assert ok is False


# =============================================================================
# CNH
# =============================================================================


class TestVerifyCnh:
    """Testes de verificação para CNH (B-01: contexto primeiro)."""

    def test_cnh_with_validator_and_context(self, model_path):
        anonymizer = Anonimizar(model_path)
        t1 = "cnh 11111111111"
        ent1 = make_ent("CNH", t1, "11111111111")
        assert anonymizer.verify_entities_unified(ent1, t1) is True

        t2 = "documento 11111111111"
        ent2 = make_ent("CNH", t2, "11111111111")
        assert anonymizer.verify_entities_unified(ent2, t2) is False

        anonymizer.use_cnh_validator = False
        assert anonymizer.verify_entities_unified(ent1, t1) is True

    def test_cnh_keyword_no_contexto_aceita_digitos_invalidos(self):
        text = "cnh 11111111111"
        result = validate_cnh_context(text, 4, 15, "11111111111", _make_logger(), use_cnh_validator=True)
        assert result is True

    def test_cnh_keyword_habilitacao_aceita(self):
        text = "carteira de habilitação 11111111111"
        result = validate_cnh_context(text, 24, 35, "11111111111", _make_logger(), use_cnh_validator=True)
        assert result is True

    def test_cnh_sem_contexto_digitos_invalidos_rejeita(self):
        text = "documento 11111111111"
        result = validate_cnh_context(text, 10, 21, "11111111111", _make_logger(), use_cnh_validator=True)
        assert result is False

    def test_cnh_sem_contexto_validator_desativado_retorna_false(self):
        text = "documento 11111111111"
        result = validate_cnh_context(text, 10, 21, "11111111111", _make_logger(), use_cnh_validator=False)
        assert result is False

    def test_cnh_com_contexto_validator_desativado_aceita(self):
        text = "cnh 11111111111"
        result = validate_cnh_context(text, 4, 15, "11111111111", _make_logger(), use_cnh_validator=False)
        assert result is True


# =============================================================================
# SIAPE
# =============================================================================


class TestVerifySiape:
    """Testes de verificação para SIAPE (contexto e keywords P0)."""

    def test_siape_needs_siape_and_not_sei(self, model_path):
        anonymizer = Anonimizar(model_path)
        t1 = "siape 1234567"
        ent1 = make_ent("SIAPE", t1, "1234567")
        assert anonymizer.verify_entities_unified(ent1, t1) is True

        t2 = "1234567 sei"
        ent2 = make_ent("SIAPE", t2, "1234567")
        assert anonymizer.verify_entities_unified(ent2, t2) is False

    def test_siape_keyword_matricula_com_acento_aceito(self):
        text = "matrícula 1234567"
        result = validate_siape_context(text, 10, 17, "1234567", _make_logger())
        assert result is True

    def test_siape_keyword_matricula_sem_acento_aceito(self):
        text = "matricula 1234567"
        result = validate_siape_context(text, 10, 17, "1234567", _make_logger())
        assert result is True

    def test_siape_keyword_mat_ponto_aceito(self):
        text = "mat. 1234567"
        result = validate_siape_context(text, 5, 12, "1234567", _make_logger())
        assert result is True

    def test_siape_keyword_mat_dois_pontos_aceito(self):
        text = "mat: 1234567"
        result = validate_siape_context(text, 5, 12, "1234567", _make_logger())
        assert result is True

    def test_siape_matricula_em_tabela_distante_rejeitado(self):
        header = "Matricula" + " " * 150
        text = header + "1234567"
        start = len(header)
        end = start + 7
        result = validate_siape_context(text, start, end, "1234567", _make_logger())
        assert result is False

    def test_siape_sem_keyword_rejeitado(self):
        text = "valor 1234567 reais"
        result = validate_siape_context(text, 6, 13, "1234567", _make_logger())
        assert result is False


# =============================================================================
# EMAIL (B-06)
# =============================================================================


class TestVerifyEmail:
    """B-06: pontuação final (.,;:) não deve invalidar o email."""

    def test_email_com_ponto_final_aceito(self):
        text = "contato: joao@email.com."
        result = validate_email_context(text, 9, 24, "joao@email.com.", _make_logger())
        assert result is True

    def test_email_com_virgula_aceito(self):
        text = "emails: a@b.com, c@d.com"
        result = validate_email_context(text, 8, 16, "a@b.com,", _make_logger())
        assert result is True

    def test_email_com_ponto_e_virgula_aceito(self):
        result = validate_email_context("x@y.com;", 0, 8, "x@y.com;", _make_logger())
        assert result is True

    def test_email_sem_arroba_rejeita(self):
        result = validate_email_context("site.com", 0, 8, "site.com", _make_logger())
        assert result is False

    def test_email_com_barra_rejeita(self):
        result = validate_email_context("user@domain.com/page", 0, 20, "user@domain.com/page", _make_logger())
        assert result is False

    def test_email_url_no_contexto_rejeita(self):
        text = "acesse http://user@site.com"
        result = validate_email_context(text, 16, 27, "user@site.com", _make_logger())
        assert result is False

    def test_email_valido_aceito(self):
        result = validate_email_context("maria@empresa.gov.br", 0, 20, "maria@empresa.gov.br", _make_logger())
        assert result is True

    def test_email_subdominio_aceito(self):
        result = validate_email_context("x@mail.servidor.gov.br", 0, 22, "x@mail.servidor.gov.br", _make_logger())
        assert result is True


# =============================================================================
# ENDEREÇO
# =============================================================================


class TestVerifyEndereco:
    def test_endereco_requires_context(self, model_path):
        anonymizer = Anonimizar(model_path)
        text = "01234-567"
        ent = make_ent("ENDEREÇO", text, "01234-567")
        assert anonymizer.verify_entities_unified(ent, text) is False

        text2 = "CEP: 01234-567"
        ent2 = make_ent("ENDEREÇO", text2, "01234-567")
        assert anonymizer.verify_entities_unified(ent2, text2) is True


# =============================================================================
# TELEFONE
# =============================================================================


class TestVerifyTelefone:
    def test_telefone_requires_context(self, model_path):
        anonymizer = Anonimizar(model_path)
        t1 = "(11) 99999-9999"
        ent1 = make_ent("TELEFONE", t1, t1)
        assert anonymizer.verify_entities_unified(ent1, t1) is False

        t2 = "telefone (11) 99999-9999"
        ent2 = make_ent("TELEFONE", t2, "(11) 99999-9999")
        assert anonymizer.verify_entities_unified(ent2, t2) is True


# =============================================================================
# DATA_NASCIMENTO
# =============================================================================


class TestVerifyDataNascimento:
    def test_data_nascimento_numeric(self, model_path):
        anonymizer = Anonimizar(model_path)
        t1 = "nascimento 15/03/1985"
        ent1 = make_ent("DATA_NASCIMENTO", t1, "15/03/1985")
        assert anonymizer.verify_entities_unified(ent1, t1) is True


# =============================================================================
# PASSAPORTE
# =============================================================================


class TestVerifyPassaporte:
    def test_passaporte_length_and_context(self, model_path):
        anonymizer = Anonimizar(model_path)
        t1 = "passaporte AB123456"
        ent1 = make_ent("PASSAPORTE", t1, "AB123456")
        assert anonymizer.verify_entities_unified(ent1, t1) is True

        t2 = "AB123456"
        ent2 = make_ent("PASSAPORTE", t2, "AB123456")
        assert anonymizer.verify_entities_unified(ent2, t2) is False

        t3 = "passaporte ABCDEF0123X"
        ent3 = make_ent("PASSAPORTE", t3, "ABCDEF0123X")
        assert anonymizer.verify_entities_unified(ent3, t3) is False


# =============================================================================
# TITULO_ELEITOR
# =============================================================================


class TestVerifyTituloEleitor:
    def test_titulo_eleitor_validator_and_context(self, model_path):
        anonymizer = Anonimizar(model_path)
        t1 = "eleitor 000000000000"
        ent1 = make_ent("TITULO_ELEITOR", t1, "000000000000")
        assert anonymizer.verify_entities_unified(ent1, t1) is False

        anonymizer.use_titulo_validator = False
        assert anonymizer.verify_entities_unified(ent1, t1) is True

    def test_titulo_eleitor_contexto_explicito_aceita_dez_a_doze_digitos(self, model_path):
        anonymizer = Anonimizar(model_path)
        text = "Título de Eleitor: 8185632089"
        ent = make_ent("TITULO_ELEITOR", text, "8185632089")
        assert anonymizer.verify_entities_unified(ent, text) is True

    def test_titulo_eleitor_zona_secao_curta_rejeita(self, model_path):
        anonymizer = Anonimizar(model_path)
        text = "TITULO DE ELEITOR ATUAL Nº 040716170310 ZONA: 036 SEÇÃO: 0258"
        zona = make_ent("TITULO_ELEITOR", text, "036")
        secao = make_ent("TITULO_ELEITOR", text, "0258")
        assert anonymizer.verify_entities_unified(zona, text) is False
        assert anonymizer.verify_entities_unified(secao, text) is False


# =============================================================================
# DADOS_BANCARIOS
# =============================================================================


class TestVerifyDadosBancarios:
    def test_dados_bancarios_context(self, model_path):
        anonymizer = Anonimizar(model_path)
        t1 = "1234-5"
        ent1 = make_ent("DADOS_BANCARIOS", t1, "1234-5")
        assert anonymizer.verify_entities_unified(ent1, t1) is False

        t2 = "agência 1234-5"
        ent2 = make_ent("DADOS_BANCARIOS", t2, "1234-5")
        assert anonymizer.verify_entities_unified(ent2, t2) is True


# =============================================================================
# CID
# =============================================================================


class TestVerifyCid:
    def test_cid_context(self, model_path):
        anonymizer = Anonimizar(model_path)
        t2 = "Paciente com diagnóstico CID F32.1"
        ent2 = make_ent("CID", t2, "F32.1")
        assert anonymizer.verify_entities_unified(ent2, t2) is True


# =============================================================================
# GEO_COORD
# =============================================================================


class TestVerifyGeoCoord:
    def test_geo_coord_variants(self, model_path):
        anonymizer = Anonimizar(model_path)
        t1 = "24°27'13.0\"S"
        ent1 = make_ent("GEO_COORD", t1, t1)
        assert anonymizer.verify_entities_unified(ent1, t1) is True

        t2 = "latitude -15,789509 -47,911627"
        ent2 = make_ent("GEO_COORD", t2, "-15,789509 -47,911627")
        assert anonymizer.verify_entities_unified(ent2, t2) is True

        t3 = "coordenada x"
        ent3 = make_ent("GEO_COORD", t3, "x")
        assert anonymizer.verify_entities_unified(ent3, t3) is False
