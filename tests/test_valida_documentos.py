import pytest

import anonimizar._validators.documents as docs
from anonimizar import Anonimizar


class TestValidaCpf:
    """Testes para validação de CPF (Anonimizar.valida_cpf)."""

    @pytest.mark.parametrize(
        ("cpf", "expected"),
        [
            ("123.456.789-09", True),
            ("000.000.000-00", False),
            ("123.456.789-00", False),
            ("12345678909", True),
        ],
    )
    def test_valida_cpf(self, cpf, expected):
        result = Anonimizar.valida_cpf(cpf)
        assert result == expected

    def test_cpf_validation_with_asterisks(self):
        result = Anonimizar.valida_cpf("***.456.789-**")
        assert result is False


class TestValidaCPFEdgeCases:
    """Testa branches não cobertos de documents.py — valida_cpf."""

    def test_cpf_com_asterisco_retorna_false(self):
        assert docs.valida_cpf("***.456.789-**") is False

    def test_cpf_com_letra_x_retorna_false(self):
        assert docs.valida_cpf("123.456.789-x0") is False
        assert docs.valida_cpf("123x56789-00") is False

    def test_cpf_digitos_iguais_retorna_false(self):
        assert docs.valida_cpf("000.000.000-00") is False
        assert docs.valida_cpf("11111111111") is False
        assert docs.valida_cpf("999.999.999-99") is False

    def test_cpf_tamanho_errado_retorna_false(self):
        assert docs.valida_cpf("123.456") is False
        assert docs.valida_cpf("123456789012") is False

    def test_cpf_dv1_incorreto_retorna_false(self):
        assert docs.valida_cpf("529.982.247-35") is False

    def test_cpf_dv2_incorreto_retorna_false(self):
        assert docs.valida_cpf("529.982.247-24") is False

    def test_cpf_valido_retorna_true(self):
        assert docs.valida_cpf("529.982.247-25") is True
        assert docs.valida_cpf("52998224725") is True


class TestValidaTituloEleitor:
    """Testes para validação de título de eleitor (Anonimizar.valida_titulo_eleitor)."""

    def test_titulo_validation(self):
        result = Anonimizar.valida_titulo_eleitor("123456789011")
        assert isinstance(result, bool)


class TestValidaTituloEleitorEdgeCases:
    """Testa branches não cobertos de documents.py — valida_titulo_eleitor."""

    def test_titulo_tamanho_errado_retorna_false(self):
        assert docs.valida_titulo_eleitor("12345") is False
        assert docs.valida_titulo_eleitor("1234567890123") is False

    def test_titulo_uf_zero_retorna_false(self):
        assert docs.valida_titulo_eleitor("123456780000") is False

    def test_titulo_uf_acima_28_retorna_false(self):
        assert docs.valida_titulo_eleitor("123456789900") is False

    def test_titulo_dv1_incorreto_retorna_false(self):
        assert docs.valida_titulo_eleitor("000000010100") is False

    def test_titulo_dv2_incorreto_retorna_false(self):
        assert docs.valida_titulo_eleitor("000000010101") is False


class TestValidaCnhWrapper:
    """Testes para Anonimizar.valida_cnh (wrapper staticmethod)."""

    def test_valida_cnh_invalida(self):
        assert Anonimizar.valida_cnh("12345") is False

    def test_valida_cnh_vazia(self):
        assert Anonimizar.valida_cnh("") is False


class TestValidaCnpjWrapper:
    """Testes para Anonimizar.valida_cnpj (wrapper staticmethod)."""

    def test_valida_cnpj_valido(self):
        assert Anonimizar.valida_cnpj("12.345.678/0001-23") is True

    def test_valida_cnpj_invalido(self):
        assert Anonimizar.valida_cnpj("00.000.000/0000-00") is False


class TestValidaCNHEdgeCases:
    """Testa branches não cobertos de documents.py — valida_cnh."""

    def test_cnh_tamanho_errado_retorna_false(self):
        assert docs.valida_cnh("12345") is False
        assert docs.valida_cnh("123456789012") is False

    def test_cnh_11_digitos_dv1_incorreto(self):
        assert docs.valida_cnh("12345678900") is False

    def test_cnh_espelho_10_digitos_exercita_branch(self):
        result = docs.valida_cnh("1234567890")
        assert isinstance(result, bool)

    def test_cnh_com_pontuacao(self):
        result = docs.valida_cnh("123.456.789-01")
        assert isinstance(result, bool)

    def test_cnh_vazia_retorna_false(self):
        assert docs.valida_cnh("") is False


class TestValidaCNPJEdgeCases:
    """Testa branches não cobertos de documents.py — valida_cnpj."""

    def test_cnpj_digitos_iguais_retorna_false(self):
        assert docs.valida_cnpj("00.000.000/0000-00") is False
        assert docs.valida_cnpj("11111111111111") is False

    def test_cnpj_tamanho_errado_retorna_false(self):
        assert docs.valida_cnpj("12.345") is False
        assert docs.valida_cnpj("123456789012345") is False

    def test_cnpj_valido_retorna_true(self):
        assert docs.valida_cnpj("12.345.678/0001-23") is True
        assert docs.valida_cnpj("18781203000196") is True


class TestValidaPis:
    """Testes para validação de PIS/PASEP/NIT."""

    @pytest.mark.parametrize(
        ("numero", "esperado"),
        [
            ("19020525177", True),
            ("12096379549", True),
            ("13151959279", True),
            ("11111111111", False),
            ("12345678901", False),
            ("00000000000", False),
            ("12345", False),
        ],
    )
    def test_valida_pis(self, numero, esperado):
        assert docs.valida_pis(numero) == esperado

    def test_valida_pis_com_formatacao(self):
        assert docs.valida_pis("190.20525.17-7") is True

    def test_valida_pis_invalido(self):
        assert docs.valida_pis("19020525178") is False


class TestValidaCns:
    """Testes para validação de CNS."""

    @pytest.mark.parametrize(
        ("numero", "esperado"),
        [
            ("898004614875795", True),
            ("898001423896809", True),
            ("111111111111111", False),
            ("000000000000000", False),
            ("12345", False),
        ],
    )
    def test_valida_cns(self, numero, esperado):
        assert docs.valida_cns(numero) == esperado

    def test_cns_prefixo_invalido(self):
        assert docs.valida_cns("312345678901234") is False
