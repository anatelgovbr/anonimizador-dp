"""Funções de validação de documentos brasileiros.

Este módulo fornece validadores para CPF, CNPJ, CNH, Título de Eleitor,
PIS e CNS, seguindo as regras oficiais de cada documento.
"""

import re

from anonimizar._constants import (
    CNH_DIGIT_COUNT,
    CNH_ESPELHO_DIGIT_COUNT,
    CNPJ_DIGIT_COUNT,
    CNPJ_WEIGHTS_1,
    CNS_DIGIT_COUNT,
    CNS_VALID_PREFIXES,
    CPF_DIGIT_COUNT,
    DV_MODULO,
    DV_REMAINDER_THRESHOLD,
    MAX_UF_CODE,
    PIS_DIGIT_COUNT,
    PIS_WEIGHTS,
    TITULO_DV_THRESHOLD,
    TITULO_ELEITOR_DIGIT_COUNT,
    TITULO_MODULO,
)


def valida_cpf(cpf: str) -> bool:  # noqa: PLR0911
    """Valida um CPF completo (com ou sem formatação), incluindo dígitos verificadores.

    A função:
    - Ignora CPFs com asteriscos (mascarados)
    - Aceita formatos com pontuação: 123.456.789-00
    - Aceita formatos sem pontuação: 12345678900
    - Rejeita CPFs com todos os dígitos iguais
    - Valida os dígitos verificadores

    Args:
        cpf: CPF em string, com ou sem pontuação.

    Returns:
        bool: True se o CPF for válido e completo; False caso contrário.

    Examples:
        >>> valida_cpf('529.982.247-25')
        True
        >>> valida_cpf('52998224725')
        True
        >>> valida_cpf('111.111.111-11')
        False
        >>> valida_cpf('***.456.789-**')
        False
    """
    if not cpf or len(cpf.strip()) == 0:
        return False

    if "*" in cpf:
        return False

    if "x" in str(cpf).lower():
        return False

    digitos = re.sub(r"\D", "", cpf)

    if len(digitos) != CPF_DIGIT_COUNT:
        return False

    if digitos == digitos[0] * CPF_DIGIT_COUNT:
        return False

    soma = sum(int(d) * i for d, i in zip(digitos[:9], range(10, 1, -1), strict=True))
    digito1 = (soma * DV_MODULO % CPF_DIGIT_COUNT) % DV_MODULO
    if digito1 != int(digitos[9]):
        return False

    soma = sum(int(d) * i for d, i in zip(digitos[:10], range(11, 1, -1), strict=True))
    digito2 = (soma * DV_MODULO % CPF_DIGIT_COUNT) % DV_MODULO
    return digito2 == int(digitos[10])


def valida_titulo_eleitor(numero: str) -> bool:
    """Valida Título de Eleitor (12 dígitos) conforme regra do TSE.

    Aceita com ou sem formatação; retorna True só se DV estiver correto.

    Args:
        numero (str): Número do título com ou sem formatação.

    Returns:
        bool: True se válido, False caso contrário.
    """
    dig = re.sub(r"\D", "", numero)
    if len(dig) != TITULO_ELEITOR_DIGIT_COUNT:
        return False

    seq8 = dig[:8]
    uf = dig[8:10]
    dv1o = int(dig[10])
    dv2o = int(dig[11])

    uf_int = int(uf)
    if uf_int < 1 or uf_int > MAX_UF_CODE:
        return False

    soma1 = sum(int(d) * p for d, p in zip(seq8, range(2, 10), strict=True))
    resto1 = soma1 % TITULO_MODULO
    dv1 = 0 if resto1 == TITULO_DV_THRESHOLD else resto1
    if dv1 != dv1o:
        return False

    base_dv2 = uf + str(dv1)
    soma2 = sum(int(d) * p for d, p in zip(base_dv2, [7, 8, 9], strict=True))
    resto2 = soma2 % TITULO_MODULO
    dv2 = 0 if resto2 == TITULO_DV_THRESHOLD else resto2
    return dv2 == dv2o


def valida_cnh(numero: str) -> bool:
    """Valida CNH conforme padrões oficiais.

    - 11 dígitos: Registro Nacional (9 + 2 DV) - algoritmo módulo 11
    - 10 dígitos: Espelho (9 + 1 DV) - algoritmo módulo 11

    Args:
        numero (str): Número da CNH com ou sem formatação

    Returns:
        bool: True se válido, False caso contrário
    """
    dig = re.sub(r"\D", "", numero)

    if len(dig) == CNH_DIGIT_COUNT:
        corpo, dv1o, dv2o = dig[:9], int(dig[9]), int(dig[10])

        soma1 = sum(int(d) * p for d, p in zip(corpo, range(9, 0, -1), strict=True))
        resto1 = soma1 % CPF_DIGIT_COUNT
        dv1 = 0 if resto1 < DV_REMAINDER_THRESHOLD else CPF_DIGIT_COUNT - resto1

        if dv1 != dv1o:
            return False

        soma2 = sum(int(d) * p for d, p in zip(corpo + str(dv1), range(1, 11), strict=True))
        resto2 = soma2 % CPF_DIGIT_COUNT
        dv2 = 0 if resto2 < DV_REMAINDER_THRESHOLD else CPF_DIGIT_COUNT - resto2

        return dv2 == dv2o

    if len(dig) == CNH_ESPELHO_DIGIT_COUNT:
        corpo, dvo = dig[:9], int(dig[9])

        soma = sum(int(d) * p for d, p in zip(corpo, range(9, 0, -1), strict=True))
        resto = soma % CPF_DIGIT_COUNT
        dv = 0 if resto < DV_REMAINDER_THRESHOLD else CPF_DIGIT_COUNT - resto

        return dv == dvo

    return False


def _calcula_digito_cnpj(digs: str, pesos: list) -> int:
    """Calcula dígito verificador de CNPJ.

    Faz o cálculo do dígito que é o resto da divisão por 11 da soma
    dos dígitos multiplicados pelos pesos.

    Args:
        digs: Dígitos do CNPJ
        pesos: Lista de pesos

    Returns:
        int: Dígito verificador (0-9)
    """
    if len(digs) != len(pesos):
        msg = f"Length mismatch: digs={len(digs)}, pesos={len(pesos)}, digs='{digs}'"
        raise ValueError(msg)
    soma = sum(int(d) * p for d, p in zip(digs, pesos, strict=True))
    resto = soma % CPF_DIGIT_COUNT
    dv_threshold = 2
    return 0 if resto < dv_threshold else CPF_DIGIT_COUNT - resto


def valida_cnpj(cnpj: str) -> bool:
    """Valida um CNPJ completo (com ou sem formatação), incluindo dígitos verificadores.

    Args:
        cnpj (str): CNPJ em string, com ou sem pontuação.

    Returns:
        bool: True se o CNPJ for válido e completo; False caso contrário.

    Examples:
        >>> valida_cnpj('12.345.678/0001-23')
        True
        >>> valida_cnpj('45.723.174/0001-09')
        True
        >>> valida_cnpj('18.781.203/0001-96')
        True
        >>> valida_cnpj('18781203000196')
        True
    """
    digitos = re.sub(r"\D", "", cnpj)

    if len(digitos) != CNPJ_DIGIT_COUNT:
        return False

    if digitos == digitos[0] * CNPJ_DIGIT_COUNT:
        return False

    pesos_1 = CNPJ_WEIGHTS_1
    pesos_2 = [5, *pesos_1]

    digito1 = _calcula_digito_cnpj(digitos[:12], pesos_1)
    digito2 = _calcula_digito_cnpj(digitos[:12] + str(digito1), pesos_2)

    return digitos[-2:] == f"{digito1}{digito2}"


def valida_pis(numero: str) -> bool:
    """Valida PIS/PASEP/NIT (11 dígitos, módulo 11, pesos [3,2,9,8,7,6,5,4,3,2]).

    Aceita com ou sem formatação. Difere do CPF no algoritmo de pesos e no DV.

    Args:
        numero: Número do PIS com ou sem formatação.

    Returns:
        bool: ``True`` se o dígito verificador confere.
    """
    dig = re.sub(r"\D", "", numero)
    if len(dig) != PIS_DIGIT_COUNT:
        return False
    if dig == dig[0] * PIS_DIGIT_COUNT:
        return False
    soma = sum(int(d) * p for d, p in zip(dig[:10], PIS_WEIGHTS, strict=True))
    resto = soma % CPF_DIGIT_COUNT
    dv = 0 if resto < DV_REMAINDER_THRESHOLD else CPF_DIGIT_COUNT - resto
    return dv == int(dig[10])


def valida_cns(numero: str) -> bool:
    """Valida CNS (15 dígitos, primeiro dígito em 1/2/7/8/9, módulo 11).

    Args:
        numero: Número do CNS com ou sem formatação.

    Returns:
        bool: ``True`` se o CNS for estruturalmente válido.
    """
    dig = re.sub(r"\D", "", numero)
    if len(dig) != CNS_DIGIT_COUNT:
        return False
    if int(dig[0]) not in CNS_VALID_PREFIXES:
        return False
    if dig == dig[0] * CNS_DIGIT_COUNT:
        return False
    soma = sum(int(d) * (15 - i) for i, d in enumerate(dig[:14]))
    resto = soma % CPF_DIGIT_COUNT
    dv = 0 if resto < DV_REMAINDER_THRESHOLD else CPF_DIGIT_COUNT - resto
    return dv == int(dig[14])
