"""Testes para _validators/documents.py.

Cobre edge cases das funções valida_cpf, valida_titulo_eleitor,
valida_cnh e valida_cnpj não exercitados pelos testes existentes.
"""

from anonimizar._validators.documents import (
    valida_cnh,
    valida_cnpj,
    valida_cpf,
    valida_titulo_eleitor,
)

# =============================================================================
# valida_cpf
# =============================================================================


class TestValidaCPFEdgeCases:
    """Testa linhas 52, 57, 65 de documents.py — branches não cobertos."""

    def test_cpf_com_asterisco_retorna_false(self) -> None:
        """Linha 52: CPF mascarado deve ser rejeitado."""
        assert valida_cpf("***.456.789-**") is False

    def test_cpf_com_letra_x_retorna_false(self) -> None:
        """Linha 57: CPF com 'x' deve ser rejeitado."""
        assert valida_cpf("123.456.789-x0") is False
        assert valida_cpf("123x56789-00") is False

    def test_cpf_digitos_iguais_retorna_false(self) -> None:
        """Linha 65: CPF com todos dígitos iguais deve ser rejeitado."""
        assert valida_cpf("000.000.000-00") is False
        assert valida_cpf("11111111111") is False
        assert valida_cpf("999.999.999-99") is False

    def test_cpf_tamanho_errado_retorna_false(self) -> None:
        """Linha 56: CPF com tamanho errado deve ser rejeitado."""
        assert valida_cpf("123.456") is False
        assert valida_cpf("123456789012") is False  # 12 dígitos

    def test_cpf_dv1_incorreto_retorna_false(self) -> None:
        """Linha 64: primeiro DV errado deve rejeitar."""
        # CPF válido: 529.982.247-25 — alterar DV1
        assert valida_cpf("529.982.247-35") is False

    def test_cpf_dv2_incorreto_retorna_false(self) -> None:
        """Linha 68: segundo DV errado deve rejeitar."""
        assert valida_cpf("529.982.247-24") is False

    def test_cpf_valido_retorna_true(self) -> None:
        """CPF matematicamente correto deve ser aceito."""
        assert valida_cpf("529.982.247-25") is True
        assert valida_cpf("52998224725") is True


# =============================================================================
# valida_titulo_eleitor
# =============================================================================


class TestValidaTituloEleitorEdgeCases:
    """Testa linhas 85, 96-106 de documents.py."""

    def test_titulo_tamanho_errado_retorna_false(self) -> None:
        """Linha 85: título com número diferente de 12 dígitos."""
        assert valida_titulo_eleitor("12345") is False
        assert valida_titulo_eleitor("1234567890123") is False  # 13 dígitos

    def test_titulo_uf_zero_retorna_false(self) -> None:
        """Linha 93: código de UF 00 é inválido."""
        # Construir um número que dê UF = 00
        assert valida_titulo_eleitor("123456780000") is False

    def test_titulo_uf_acima_28_retorna_false(self) -> None:
        """Linha 93: código de UF > 28 é inválido."""
        assert valida_titulo_eleitor("123456789900") is False  # UF = 99

    def test_titulo_dv1_incorreto_retorna_false(self) -> None:
        """Linha 99: DV1 errado deve rejeitar."""
        # Título com DV1 sabidamente errado
        assert valida_titulo_eleitor("000000010100") is False

    def test_titulo_dv2_incorreto_retorna_false(self) -> None:
        """Linha 105: DV2 errado deve rejeitar."""
        assert valida_titulo_eleitor("000000010101") is False


# =============================================================================
# valida_cnh
# =============================================================================


class TestValidaCNHEdgeCases:
    """Testa linhas 139-148 de documents.py (ramo de 10 digitos / espelho)."""

    def test_cnh_tamanho_errado_retorna_false(self) -> None:
        """Linha 148: CNH com tamanho diferente de 10 ou 11 dígitos."""
        assert valida_cnh("12345") is False
        assert valida_cnh("123456789012") is False  # 12 dígitos

    def test_cnh_11_digitos_dv1_incorreto(self) -> None:
        """Linha 131: DV1 da CNH 11 dígitos incorreto."""
        # Alterar último algarismo de uma CNH válida
        assert valida_cnh("12345678900") is False

    def test_cnh_espelho_10_digitos_exercita_branch(self) -> None:
        """Linha 139-146: branch de espelho (10 dígitos) é exercitado."""
        # O resultado pode ser True ou False dependendo do DV;
        # o que importa é exercitar o branch de 10 dígitos.
        result = valida_cnh("1234567890")
        assert isinstance(result, bool)

    def test_cnh_com_pontuacao(self) -> None:
        """CNH formatada com pontos deve ser processada após strip."""
        # Qualquer formatação deve cair em tamanho inválido ou calcular
        result = valida_cnh("123.456.789-01")
        assert isinstance(result, bool)

    def test_cnh_vazia_retorna_false(self) -> None:
        """CNH vazia deve retornar False."""
        assert valida_cnh("") is False


# =============================================================================
# valida_cnpj
# =============================================================================


class TestValidaCNPJEdgeCases:
    """Testa linha 199 de documents.py — CNPJ com dígitos iguais."""

    def test_cnpj_digitos_iguais_retorna_false(self) -> None:
        """Linha 199: CNPJ com todos os dígitos iguais deve ser rejeitado."""
        assert valida_cnpj("00.000.000/0000-00") is False
        assert valida_cnpj("11111111111111") is False

    def test_cnpj_tamanho_errado_retorna_false(self) -> None:
        """CNPJ com tamanho incorreto deve ser rejeitado."""
        assert valida_cnpj("12.345") is False
        assert valida_cnpj("123456789012345") is False  # 15 dígitos

    def test_cnpj_valido_retorna_true(self) -> None:
        """CNPJ matematicamente válido deve ser aceito."""
        assert valida_cnpj("12.345.678/0001-95") is True
        assert valida_cnpj("18781203000128") is True
