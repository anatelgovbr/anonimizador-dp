import os

import pytest

from anonimizar import Anonimizar
from anonimizar._extraction.pipeline import _limpar_prefixo_entidade

# =============================================================================
# Genéricos — extração básica, return types, erros
# =============================================================================


class TestExtractEntitiesBasic:
    """Testes genéricos de extração (return_type, posições, detalhes)."""

    @pytest.mark.parametrize(
        ("input_text", "return_type", "expected_count"),
        [
            ("CPF 123.456.789-09", "label_text", 1),
            ("Email teste@email.com", "label_text", 1),
            ("Telefone (11) 99999-9999", "label_text", 1),
            ("Texto sem entidades", "label_text", 0),
        ],
    )
    def test_extract_entities_basic(self, anonymizer, input_text, return_type, expected_count):
        result = anonymizer.extract_entities(input_text, return_type=return_type)
        assert len(result) == expected_count

    def test_extract_entities_with_positions(self, anonymizer):
        text = "João, CPF 123.456.789-09, email: joao@email.com"
        result = anonymizer.extract_entities(text, return_type="label_position")

        assert len(result) == 2
        cpf_entity = next(e for e in result if e["label"] == "CPF")
        assert cpf_entity["start_position"] == 10
        assert cpf_entity["end_position"] == 24

    def test_extract_entities_with_detail(self, anonymizer):
        text = "CPF 123.456.789-09"
        result = anonymizer.extract_entities(text, return_type="label_detail")

        assert len(result) == 1
        entity = result[0]
        assert "detected_by" in entity
        assert entity["text"] == "123.456.789-09"


class TestExtractEntitiesErrors:
    """Testes de erro na extração."""

    def test_invalid_return_type_raises_valueerror(self, model_path):
        anonymizer = Anonimizar(model_path)
        with pytest.raises(ValueError, match="Tipo de retorno não permitido"):
            anonymizer.extract_entities("texto", return_type="invalid_type")

    def test_invalid_return_type(self, anonymizer):
        with pytest.raises(ValueError, match="Tipo de retorno não permitido"):
            anonymizer.extract_entities("teste", return_type="invalid_type")


# =============================================================================
# CPF
# =============================================================================


@pytest.mark.entity
class TestExtractCpf:
    @pytest.mark.cpf
    def test_extract_cpf_pontuado(self, anonymizer):
        result = anonymizer.extract_entities("Meu CPF é 123.456.789-09.", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "CPF"
        assert result[0]["text"] == "123.456.789-09"
        assert "detected_by" in result[0]

    @pytest.mark.cpf
    @pytest.mark.parametrize(
        ("input_text", "expected_cpf"),
        [("CPF 123.456.789-09", "123.456.789-09"), ("nº:12345678909", "12345678909")],
    )
    def test_extract_cpf_formats(self, anonymizer, input_text, expected_cpf):
        result = anonymizer.extract_entities(input_text, return_type="label_text")
        cpf_entities = [e for e in result if e["label"] == "CPF"]
        assert len(cpf_entities) > 0
        assert cpf_entities[0]["text"] == expected_cpf


# =============================================================================
# RG
# =============================================================================


@pytest.mark.entity
class TestExtractRg:
    @pytest.mark.rg
    def test_extract_rg_pontuado(self, anonymizer):
        result = anonymizer.extract_entities("Meu RG é 12.345.678-9.", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "RG"
        assert result[0]["text"] == "12.345.678-9"
        assert "detected_by" in result[0]

    @pytest.mark.rg
    @pytest.mark.parametrize(
        ("text", "expected_text"),
        [
            ("RG 3990139 SSP/DF", "3990139"),
            ("RG 2355046 SSP/DF", "2355046"),
            ("RG: 19118939 - DF", "19118939"),
            ("RG n 1822245 SSPDF", "1822245"),
        ],
    )
    def test_extract_rg_com_orgao_emissor_normalizado(self, anonymizer, text, expected_text):
        """RG com orgao emissor deve ter sufixo removido apos extracao."""
        result = anonymizer.extract_entities(text, return_type="label_detail")
        rgs = [e for e in result if e["label"] == "RG"]
        assert len(rgs) >= 1
        assert rgs[0]["text"] == expected_text


# =============================================================================
# EMAIL
# =============================================================================


@pytest.mark.entity
class TestExtractEmail:
    @pytest.mark.email
    def test_extract_email(self, anonymizer):
        result = anonymizer.extract_entities("Email: teste@email.com", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "EMAIL"
        assert result[0]["text"] == "teste@email.com"
        assert "detected_by" in result[0]

    @pytest.mark.email
    @pytest.mark.parametrize(
        ("input_text", "expected_email"),
        [
            ("Email: teste@dominio.com", "teste@dominio.com"),
            ("Contato via usuario@empresa.com.br", "usuario@empresa.com.br"),
            ("user_123@test-domain.org", "user_123@test-domain.org"),
        ],
    )
    def test_extract_email_formats(self, anonymizer, input_text, expected_email):
        result = anonymizer.extract_entities(input_text, return_type="label_text")
        email_entities = [e for e in result if e["label"] == "EMAIL"]
        assert len(email_entities) > 0
        assert email_entities[0]["text"] == expected_email


class TestEmailValidation:
    """Testes para a validação de EMAIL para reduzir falsos positivos."""

    def test_valid_emails_accepted(self, anonymizer):
        test_cases = [
            "O contato é joao@email.com.",
            "Email: maria.silva@empresa.gov.br",
            "Use o e-mail user+tag@domain.org para isso.",
            "Seu endereço é nome_123@dominio.com.br",
        ]
        for text in test_cases:
            result = anonymizer.extract_entities(text, return_type="label_detail")
            emails = [e for e in result if e["label"] == "EMAIL"]
            assert len(emails) >= 1, f"Falha ao detectar email válido em: {text}"
            assert "@" in emails[0]["text"]

    def test_url_like_emails_rejected(self, anonymizer):
        test_cases = [
            "Acesse http://site.com@redirect.com/path para mais.",
            "O link é https://usuario:senha@dominio.com.",
            "Visite www.site.com@dominio.com",
            "Caminho do arquivo: usuario@dominio.com/pagina/arquivo",
        ]
        for text in test_cases:
            result = anonymizer.extract_entities(text, return_type="label_detail")
            emails = [e for e in result if e["label"] == "EMAIL"]
            assert len(emails) == 0, f"Falso positivo detectado em URL: {text} -> {emails}"

    def test_invalid_email_formats_rejected(self, anonymizer):
        test_cases = [
            "Isso é apenas um texto-sem-arroba.com",
            "Email inválido: @dominio.com",
            "Faltou o domínio: usuario@",
            "Domínio sem TLD: usuario@dominio",
            "TLD numérico inválido: usuario@dominio.1",
            "TLD muito curto: usuario@dominio.c",
        ]
        for text in test_cases:
            result = anonymizer.extract_entities(text)
            emails = [e for e in result if e["label"] == "EMAIL"]
            assert len(emails) == 0, f"Falso positivo detectado em formato inválido: {text}"


# =============================================================================
# SIAPE
# =============================================================================


@pytest.mark.entity
class TestExtractSiape:
    @pytest.mark.siape
    def test_extract_siape(self, anonymizer):
        result = anonymizer.extract_entities("SIAPE 1234567", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "SIAPE"
        assert result[0]["text"] == "1234567"
        assert "detected_by" in result[0]


class TestSiapeExtract:
    """Testes P0 de extração de SIAPE (não validação de contexto)."""

    def test_siape_extract_matricula_tabela_retorna_apenas_numero(self, model_path):
        anonymizer = Anonimizar(model_path, auto_patterns=False)
        anonymizer.add_apply_patterns(["SIAPE"])
        text = "| Servidor | Matrícula SIAPE |\n|---|---|\n| Maria | 1523493 |"

        result = anonymizer.extract_entities(text, return_type="label_detail")

        assert any(entity["label"] == "SIAPE" and entity["text"] == "1523493" for entity in result)
        assert not any(entity["label"] == "SIAPE" and "SIAPE" in entity["text"] for entity in result)

    def test_siape_extract_prefixo_retorna_apenas_numero(self, model_path):
        anonymizer = Anonimizar(model_path, auto_patterns=False)
        anonymizer.add_apply_patterns(["SIAPE"])
        text = "Matrícula Siape nº 6809818, pertence ao quadro."

        result = anonymizer.extract_entities(text, return_type="label_detail")

        assert any(entity["label"] == "SIAPE" and entity["text"] == "6809818" for entity in result)
        assert not any(entity["label"] == "SIAPE" and "Siape" in entity["text"] for entity in result)


# =============================================================================
# PASSAPORTE
# =============================================================================


@pytest.mark.entity
class TestExtractPassaporte:
    @pytest.mark.passaporte
    def test_extract_passaporte(self, anonymizer):
        result = anonymizer.extract_entities("Passaporte AB123456", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "PASSAPORTE"
        assert result[0]["text"] == "AB123456"
        assert "detected_by" in result[0]


# =============================================================================
# DADOS_BANCARIOS
# =============================================================================


@pytest.mark.entity
class TestExtractDadosBancarios:
    @pytest.mark.dados_bancarios
    def test_extract_dados_bancarios(self, anonymizer):
        result = anonymizer.extract_entities("Conta bancária 1234-5", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "DADOS_BANCARIOS"
        assert result[0]["text"] == "1234-5"
        assert "detected_by" in result[0]


# =============================================================================
# TELEFONE
# =============================================================================


@pytest.mark.entity
class TestExtractTelefone:
    @pytest.mark.telefone
    def test_extract_telefone(self, anonymizer):
        result = anonymizer.extract_entities("Telefone: (11) 99999-9999", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "TELEFONE"
        assert result[0]["text"] == "(11) 99999-9999"
        assert "detected_by" in result[0]


# =============================================================================
# DATA_NASCIMENTO
# =============================================================================


@pytest.mark.entity
class TestExtractDataNascimento:
    @pytest.mark.data_nascimento
    def test_extract_data_nascimento(self, anonymizer):
        result = anonymizer.extract_entities("Data de nascimento: 15/03/1985", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "DATA_NASCIMENTO"
        assert result[0]["text"] == "15/03/1985"
        assert "detected_by" in result[0]


# =============================================================================
# CNH
# =============================================================================


@pytest.mark.entity
class TestExtractCnh:
    @pytest.mark.cnh
    def test_extract_cnh(self, anonymizer):
        result = anonymizer.extract_entities("CNH 12345678901", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "CNH"
        assert result[0]["text"] == "12345678901"
        assert "detected_by" in result[0]


# =============================================================================
# ENDEREÇO
# =============================================================================


@pytest.mark.entity
class TestExtractEndereco:
    @pytest.mark.endereco
    def test_extract_endereco(self, anonymizer):
        result = anonymizer.extract_entities("Endereço com CEP 01234-567", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "ENDEREÇO"
        assert result[0]["text"] == "01234-567"
        assert "detected_by" in result[0]


# =============================================================================
# GEO_COORD
# =============================================================================


@pytest.mark.entity
class TestExtractGeoCoord:
    @pytest.mark.geo_coord
    def test_extract_geo_coord(self, anonymizer):
        result = anonymizer.extract_entities("Coordenada: 22°31.26' S", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "GEO_COORD"
        assert result[0]["text"] == "22°31.26' S"
        assert "detected_by" in result[0]


# =============================================================================
# CID
# =============================================================================


@pytest.mark.entity
class TestExtractCid:
    @pytest.mark.entity
    def test_extract_cid(self, anonymizer):
        result = anonymizer.extract_entities("CID meu atestado referente ao A10", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "CID"
        assert result[0]["text"] == "A10"
        assert "detected_by" in result[0]


# =============================================================================
# RG Estrangeiro (RNE/CRNM)
# =============================================================================


@pytest.mark.rg
class TestRgEstrangeiro:
    """Testes para detecção de RG estrangeiro (RNE/CRNM)."""

    def test_rne_com_prefixo_detectado(self, anonymizer_rne):
        test_cases = [
            ("RNE 123456789", "123456789"),
            ("RNE: 1234567", "1234567"),
            ("CRNM 12345678", "12345678"),
            ("CRNM: 123456789", "123456789"),
            ("Nº do RNE 123456789", "123456789"),
        ]
        for text, expected_number in test_cases:
            result = anonymizer_rne.extract_entities(text, return_type="label_detail")
            rgs = [e for e in result if e["label"] == "RG"]
            assert len(rgs) >= 1, f"Falha ao detectar RNE em: {text}"
            assert expected_number in rgs[0]["text"], f"Número não encontrado em: {text} -> {rgs[0]['text']}"

    def test_rne_formato_pontuado_detectado(self, anonymizer_rne):
        test_cases = [
            "RNE 12.345.678-9",
            "CRNM: 1.234.567-8",
        ]
        for text in test_cases:
            result = anonymizer_rne.extract_entities(text, return_type="label_detail")
            rgs = [e for e in result if e["label"] == "RG"]
            assert len(rgs) >= 1, f"Falha ao detectar RNE pontuado em: {text}"

    def test_registro_nacional_por_extenso(self, anonymizer_rne):
        test_cases = [
            "Registro Nacional de Estrangeiro: 123456789",
            "Registro Nacional de Estrangeiro nº 1234567",
            "Registro Nacional Migratório: 12345678",
        ]
        for text in test_cases:
            result = anonymizer_rne.extract_entities(text, return_type="label_detail")
            rgs = [e for e in result if e["label"] == "RG"]
            assert len(rgs) >= 1, f"Falha ao detectar registro por extenso em: {text}"

    def test_numeros_soltos_nao_detectados(self, anonymizer_rne):
        test_cases = [
            "O número 123456789 não é um documento.",
            "Protocolo 12345678 recebido.",
            "Valor de R$ 1234567 registrado.",
        ]
        for text in test_cases:
            result = anonymizer_rne.extract_entities(text, return_type="label_detail")
            rgs = [e for e in result if e["label"] == "RG"]
            assert len(rgs) == 0, f"Falso positivo RNE detectado em: {text} -> {rgs}"

    def test_foreign_rg_false_nao_detecta_rne(self):
        model_path = os.getenv("SPACY_MODEL_PATH", "X:/sei-anonimizar-bckp/nlp_treinado_v5")
        anon = Anonimizar(model_path=model_path, auto_patterns=False)
        anon.add_apply_patterns(["RG"], foreign_rg=False)

        text = "RNE 123456789"
        result = anon.extract_entities(text, return_type="label_detail")
        rgs = [e for e in result if e["label"] == "RG"]
        assert len(rgs) == 0, f"RNE detectado sem foreign_rg=True: {rgs}"

    def test_auto_patterns_detecta_rne_por_padrao(self):
        model_path = os.getenv("SPACY_MODEL_PATH", "X:/sei-anonimizar-bckp/nlp_treinado_v5")
        anon = Anonimizar(model_path=model_path)

        result = anon.extract_entities("RNE 123456789", return_type="label_detail")

        assert any(e["label"] == "RG" for e in result)


# =============================================================================
# PIS
# =============================================================================


class TestPisPatterns:
    """Testes para detecção de PIS via padrões."""

    @pytest.mark.pis
    def test_pis_com_prefixo(self, anonymizer):
        result = anonymizer.extract_entities("PIS: 19020525177", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "PIS"

    @pytest.mark.pis
    def test_pis_pasep(self, anonymizer):
        result = anonymizer.extract_entities("PASEP: 12096379549", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "PIS"

    @pytest.mark.pis
    def test_pis_nit(self, anonymizer):
        result = anonymizer.extract_entities("NIT 13151959279", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "PIS"

    @pytest.mark.pis
    def test_pis_sem_contexto_rejeitado(self, anonymizer):
        """Número com DV de PIS válido, mas sem palavra-chave no entorno, deve ser rejeitado.

        Regressão: o validador aceitava qualquer sequência de 11 dígitos com DV válido
        mesmo sem contexto textual (PIS/PASEP/NIT), gerando falsos positivos em números
        de 11 dígitos soltos (protocolos, processos, etc.).
        """
        result = anonymizer.extract_entities("Processo 19020525177 arquivado", return_type="label_detail")
        pis_results = [r for r in result if r["label"] == "PIS"]
        assert len(pis_results) == 0


# =============================================================================
# CNS
# =============================================================================


class TestCnsPatterns:
    """Testes para detecção de CNS via padrões."""

    @pytest.mark.cns
    def test_cns_com_prefixo(self, anonymizer):
        result = anonymizer.extract_entities("CNS: 898004614875795", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "CNS"

    @pytest.mark.cns
    def test_cns_sem_prefixo(self, anonymizer):
        result = anonymizer.extract_entities("Número do CNS 898001423896809", return_type="label_detail")
        assert len(result) == 1
        assert result[0]["label"] == "CNS"

    @pytest.mark.cns
    def test_cns_protocolo_rejeitado(self, anonymizer):
        result = anonymizer.extract_entities("Protocolo 202305107277221", return_type="label_detail")
        cns_results = [r for r in result if r["label"] == "CNS"]
        assert len(cns_results) == 0


# =============================================================================
# RESERVISTA
# =============================================================================


class TestReservistaPatterns:
    """Testes para detecção de RESERVISTA via padrões."""

    @pytest.mark.reservista
    def test_reservista_formato_hifen(self, anonymizer):
        result = anonymizer.extract_entities(
            "Certificado Reservista: 671737 - 040253111187", return_type="label_detail"
        )
        assert len(result) == 1
        assert result[0]["label"] == "RESERVISTA"

    @pytest.mark.reservista
    def test_reservista_formato_barra(self, anonymizer):
        result = anonymizer.extract_entities(
            "Certificado de Reservista: 040974984169/424213", return_type="label_detail"
        )
        assert len(result) == 1
        assert result[0]["label"] == "RESERVISTA"


# =============================================================================
# Conselhos profissionais como RG
# =============================================================================


class TestConselhoProfissionalAsRg:
    """Testes para detecção de conselhos profissionais como RG."""

    @pytest.mark.rg
    def test_crm_com_uf(self, anonymizer):
        result = anonymizer.extract_entities("CRM-DF 5601", return_type="label_detail")
        rgs = [r for r in result if r["label"] == "RG"]
        assert len(rgs) >= 1

    @pytest.mark.rg
    def test_cro_com_uf(self, anonymizer):
        result = anonymizer.extract_entities("CRO-RR 853PV", return_type="label_detail")
        rgs = [r for r in result if r["label"] == "RG"]
        assert len(rgs) >= 1

    @pytest.mark.rg
    def test_crea_formatado(self, anonymizer):
        result = anonymizer.extract_entities("CREA: 22.713-D CREA/DF", return_type="label_detail")
        rgs = [r for r in result if r["label"] == "RG"]
        assert len(rgs) >= 1

    @pytest.mark.rg
    def test_crp_sem_uf(self, anonymizer):
        result = anonymizer.extract_entities("CRP 12345", return_type="label_detail")
        rgs = [r for r in result if r["label"] == "RG"]
        assert len(rgs) >= 1

    @pytest.mark.rg
    def test_oab_com_uf(self, anonymizer):
        result = anonymizer.extract_entities("OAB/DF 123456", return_type="label_detail")
        rgs = [r for r in result if r["label"] == "RG"]
        assert len(rgs) >= 1

    @pytest.mark.rg
    def test_oab_sem_uf(self, anonymizer):
        result = anonymizer.extract_entities("OAB 123456", return_type="label_detail")
        rgs = [r for r in result if r["label"] == "RG"]
        assert len(rgs) >= 1


# =============================================================================
# B-28 — Forward window: keywords pós-entidade são detectadas
# =============================================================================


class TestB28ForwardWindow:
    """B-28: janela de contexto forward=20 detecta keywords que aparecem após a entidade."""

    def test_cpf_keyword_apos_entidade_aceito(self, model_path):
        anonymizer = Anonimizar(model_path)
        text = "529.982.247-25 (cpf)"
        entities = anonymizer.extract_entities(text)
        cpf_entities = [e for e in entities if e.get("label") == "CPF"]
        assert len(cpf_entities) >= 1

    def test_siape_keyword_apos_entidade_aceito(self, model_path):
        anonymizer = Anonimizar(model_path)
        text = "1234567 siape"
        entities = anonymizer.extract_entities(text)
        siape_entities = [e for e in entities if e.get("label") == "SIAPE"]
        assert len(siape_entities) >= 1


# =============================================================================
# Helpers do pipeline
# =============================================================================


class TestPipelineHelpers:
    """Testes para funções auxiliares do pipeline de extração."""

    def test_limpar_prefixo_rg_com_dois_pontos_direto(self):
        cleaned, delta = _limpar_prefixo_entidade("RG: 19118939 - DF", "RG")
        assert cleaned == "19118939 - DF"
        assert delta == 4


# =============================================================================
# Regressão GEO_COORD — P7 e P8
# =============================================================================


@pytest.mark.geo_coord
class TestGeoCoordP7P8Regressao:
    """Regressão para padrões P7 e P8 de GEO_COORD adicionados na Sprint 78."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Coordenada: 21\u00b0S11'43,0''", "21\u00b0S11'43,0''"),
            ("Ponto: 48\u00b0W47'21,0''", "48\u00b0W47'21,0''"),
            ("Lat: 21\u00b0S08'07,0''", "21\u00b0S08'07,0''"),
            ("Long: 48\u00b0W58'16,0''", "48\u00b0W58'16,0''"),
            ("Loc 21\u00b0S08'17,3''", "21\u00b0S08'17,3''"),
            ("Loc 48\u00b0W58'36,6''", "48\u00b0W58'36,6''"),
        ],
    )
    def test_p7_hemisferio_entre_graus_e_minutos_virgula(self, geo_anonymizer, text, expected):
        result = geo_anonymizer.extract_entities(text, return_type="label_detail")
        textos = [e["text"] for e in result if e["label"] == "GEO_COORD"]
        assert expected in textos, f"Esperado {expected!r} em {textos} para entrada {text!r}"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Lat: 20\u00b0S38'43.4\"", "20\u00b0S38'43.4\""),
            ("Long: 51\u00b0W06'38.9\"", "51\u00b0W06'38.9\""),
        ],
    )
    def test_p7_hemisferio_entre_graus_e_minutos_ponto(self, geo_anonymizer, text, expected):
        result = geo_anonymizer.extract_entities(text, return_type="label_detail")
        textos = [e["text"] for e in result if e["label"] == "GEO_COORD"]
        assert expected in textos, f"Esperado {expected!r} em {textos} para entrada {text!r}"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Latitude: -10.16281\u00b0S", "-10.16281\u00b0S"),
            ("Longitude: -48.87431\u00b0W", "-48.87431\u00b0W"),
            ("Lat -10.175774\u00b0S Long -48.886911\u00b0W", "-10.175774\u00b0S"),
            ("Lat -10.175774\u00b0S Long -48.886911\u00b0W", "-48.886911\u00b0W"),
        ],
    )
    def test_p8_decimal_com_grau_e_hemisferio_sufixo(self, geo_anonymizer, text, expected):
        result = geo_anonymizer.extract_entities(text, return_type="label_detail")
        textos = [e["text"] for e in result if e["label"] == "GEO_COORD"]
        assert expected in textos, f"Esperado {expected!r} em {textos} para entrada {text!r}"

    @pytest.mark.parametrize(
        "text",
        [
            "temperatura de 21C no local",
            "artigo 5o do regulamento",
            "versao 1.234.567 do sistema",
            "CPF 123.456.789-00 do servidor",
            "R$ 10.000,00 de despesa",
            "5\u00b0C de temperatura ambiente",
        ],
    )
    def test_p7_p8_sem_falsos_positivos(self, geo_anonymizer, text):
        result = geo_anonymizer.extract_entities(text, return_type="label_detail")
        geo_entities = [e for e in result if e["label"] == "GEO_COORD"]
        assert geo_entities == [], f"Falso positivo detectado em {text!r}: {geo_entities}"


# =============================================================================
# Regressão GEO_COORD — P9, P10, P11 (formatos compactos, par decimal, sinal separado)
# =============================================================================


@pytest.mark.geo_coord
class TestGeoCoordP9P10P11:
    """Regressão para padrões P9-P11 de GEO_COORD (Sprint 81)."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Coord 02S380200", "02S380200"),
            ("Ponto 47W561530", "47W561530"),
            ("Local 05S100140", "05S100140"),
            ("Ref 47W332120", "47W332120"),
            ("Lat.02S380200", "02S380200"),
        ],
    )
    def test_p9_compacto_sem_prefixo(self, geo_anonymizer, text, expected):
        result = geo_anonymizer.extract_entities(text, return_type="label_detail")
        textos = [e["text"] for e in result if e["label"] == "GEO_COORD"]
        assert expected in textos, f"Esperado {expected!r} em {textos} para entrada {text!r}"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("-10.296940, -48.358915", "-10.296940, -48.358915"),
            ("ponto -23.500000, -46.600000", "-23.500000, -46.600000"),
        ],
    )
    def test_p10_par_decimal_ponto(self, geo_anonymizer, text, expected):
        result = geo_anonymizer.extract_entities(text, return_type="label_detail")
        textos = [e["text"] for e in result if e["label"] == "GEO_COORD"]
        assert expected in textos, f"Esperado {expected!r} em {textos} para entrada {text!r}"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("coordenada - 38.4486", "- 38.4486"),
            ("latitude - 23.5000", "- 23.5000"),
        ],
    )
    def test_p6_espaco_negativo_com_prefixo(self, geo_anonymizer, text, expected):
        """coordenada - NNN.NNNN deve ser capturado por P6 e normalizado."""
        result = geo_anonymizer.extract_entities(text, return_type="label_detail")
        textos = [e["text"] for e in result if e["label"] == "GEO_COORD"]
        assert expected in textos, f"Esperado {expected!r} em {textos} para entrada {text!r}"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Log: -47.64083", "-47.64083"),
            ("Longitude: -51.960028", "-51.960028"),
        ],
    )
    def test_geo_com_prefixo_log(self, geo_anonymizer, text, expected):
        """Log: prefixo deve ser normalizado, texto final sem prefixo."""
        result = geo_anonymizer.extract_entities(text, return_type="label_detail")
        textos = [e["text"] for e in result if e["label"] == "GEO_COORD"]
        assert expected in textos, f"Esperado {expected!r} em {textos} para entrada {text!r}"

    @pytest.mark.parametrize(
        "text",
        [
            "log 123.456 no sistema",
            "temperatura - 38.5 graus",
            "temperatura de 38.5 graus",
        ],
    )
    def test_novos_padroes_sem_falsos_positivos(self, geo_anonymizer, text):
        result = geo_anonymizer.extract_entities(text, return_type="label_detail")
        geo_entities = [e for e in result if e["label"] == "GEO_COORD"]
        assert geo_entities == [], f"Falso positivo detectado em {text!r}: {geo_entities}"
