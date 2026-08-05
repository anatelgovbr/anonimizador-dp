"""Testes para verificar níveis de log do projeto (Estória 855).

Verifica que:
- Logs per-document são DEBUG, não INFO.
- Bug da f-string foi corrigido.
- Logs de inicialização permanecem INFO.
"""

import logging
from unittest.mock import MagicMock, patch


class TestPipelineLoggingLevels:
    """Verifica que logs per-document no pipeline são DEBUG."""

    def test_arquivo_carregado_e_debug(self, tmp_path):
        """Log 'Arquivo carregado' deve ser DEBUG ao processar arquivo .md."""
        from anonimizar._extraction.pipeline import extract_entities

        md_file = tmp_path / "test.md"
        md_file.write_text("Texto simples para teste.", encoding="utf-8")

        mock_logger = MagicMock()
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []

        extract_entities(
            nlp_trained=mock_nlp,
            text_or_path=str(md_file),
            labels=set(),
            patterns=[],
            return_type="label_text",
            verify_fn=lambda _e, _t: True,
            logger=mock_logger,
        )

        # Verifica que "Arquivo carregado" foi chamado via debug, não info
        debug_messages = [str(call) for call in mock_logger.debug.call_args_list]
        info_messages = [str(call) for call in mock_logger.info.call_args_list]

        assert any("Arquivo carregado" in msg for msg in debug_messages), "Esperava 'Arquivo carregado' como DEBUG"
        assert not any("Arquivo carregado" in msg for msg in info_messages), "'Arquivo carregado' não deveria ser INFO"

    def test_processando_texto_direto_e_debug(self):
        """Log 'Processando texto direto' deve ser DEBUG."""
        from anonimizar._extraction.pipeline import extract_entities

        mock_logger = MagicMock()
        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []

        extract_entities(
            nlp_trained=mock_nlp,
            text_or_path="Texto curto de teste",
            labels=set(),
            patterns=[],
            return_type="label_text",
            verify_fn=lambda _e, _t: True,
            logger=mock_logger,
        )

        debug_messages = [str(call) for call in mock_logger.debug.call_args_list]
        info_messages = [str(call) for call in mock_logger.info.call_args_list]

        assert any("Processando texto direto" in msg for msg in debug_messages)
        assert not any("Processando texto direto" in msg for msg in info_messages)


class TestRegistryLoggingLevels:
    """Verifica que logs do registry são DEBUG."""

    def test_aplicando_padroes_e_debug(self):
        """Log 'Aplicando padrões' deve ser DEBUG, não INFO."""
        from anonimizar._patterns.registry import add_apply_patterns

        mock_logger = MagicMock()
        patterns = []

        add_apply_patterns(
            patterns=patterns,
            labels=["CPF"],
            model_labels=[],
            logger=mock_logger,
        )

        debug_messages = [str(call) for call in mock_logger.debug.call_args_list]
        info_messages = [str(call) for call in mock_logger.info.call_args_list]

        assert any("Aplicando" in msg for msg in debug_messages), "Esperava 'Aplicando padrões' como DEBUG"
        assert not any("Aplicando" in msg for msg in info_messages), "'Aplicando padrões' não deveria ser INFO"


class TestAnonymizationLoggingLevels:
    """Verifica que logs de anonimização per-document são DEBUG."""

    def test_anonimizacao_concluida_e_debug(self):
        """Log 'Anonimização concluída' deve ser DEBUG."""
        from anonimizar._anonymization.text import anonymize_text

        mock_logger = MagicMock()

        anonymize_text(
            text="CPF 123.456.789-09 no texto",
            entities_list=[
                {"label": "CPF", "start_position": 4, "end_position": 18},
            ],
            logger=mock_logger,
        )

        debug_messages = [str(call) for call in mock_logger.debug.call_args_list]
        info_messages = [str(call) for call in mock_logger.info.call_args_list]

        assert any("Anonimização concluída" in msg or "substituições aplicadas" in msg for msg in debug_messages), (
            "Esperava 'Anonimização concluída' como DEBUG"
        )
        assert not any("Anonimização concluída" in msg or "substituições aplicadas" in msg for msg in info_messages), (
            "'Anonimização concluída' não deveria ser INFO"
        )


class TestAnonimizarLoggingLevels:
    """Verifica logs de inicialização e bug da f-string."""

    def test_inicializacao_modelo_e_info(self):
        """Log de inicialização do modelo deve ser INFO (evento de ciclo de vida)."""
        # O anonymizer já foi inicializado pela fixture.
        # Verificamos que a mensagem está no logger como INFO.
        # Fazemos isso re-inicializando com mock.
        mock_logger = MagicMock()

        with (
            patch("anonimizar._anonymization.anonymizer.create_default_logger", return_value=mock_logger),
            patch("anonimizar._anonymization.anonymizer.spacy.load") as mock_spacy,
        ):
            mock_model = MagicMock()
            mock_model.pipe_names = ["ner"]
            mock_model.get_pipe.return_value.labels = ("CPF", "EMAIL")
            mock_spacy.return_value = mock_model

            from anonimizar import Anonimizar

            Anonimizar(model_path="fake/path", logger=mock_logger)

        info_messages = [str(call) for call in mock_logger.info.call_args_list]
        assert any("Inicializando" in msg for msg in info_messages), "Esperava 'Inicializando Anonimizar' como INFO"
        assert any("Modelo carregado" in msg for msg in info_messages), (
            "Esperava 'Modelo carregado com sucesso' como INFO"
        )

    def test_fstring_bug_corrigido(self):
        """Verifica que o log de max_length usa %-formatting, não string literal quebrada."""
        mock_logger = MagicMock()

        with patch("anonimizar._anonymization.anonymizer.spacy.load") as mock_spacy:
            mock_model = MagicMock()
            mock_model.pipe_names = ["ner"]
            mock_model.get_pipe.return_value.labels = ("CPF",)
            mock_model.max_length = 3000000
            mock_spacy.return_value = mock_model

            from anonimizar import Anonimizar

            Anonimizar(model_path="fake/path", logger=mock_logger)

        # Nenhuma chamada de debug deve conter a string literal "{self.nlp_trained.max_length}"
        debug_messages = [str(call) for call in mock_logger.debug.call_args_list]
        for msg in debug_messages:
            assert "{self.nlp_trained.max_length}" not in msg, (
                f"Bug da f-string não corrigido! Encontrado literal: {msg}"
            )

        # Verifica que max_length foi passado como argumento numérico
        assert any("Max length" in str(call) for call in mock_logger.debug.call_args_list), (
            "Esperava log de 'Max length' nas chamadas debug"
        )


class TestOverlapLoggingNoNullHandler:
    """Verifica que overlap.py não adiciona NullHandler."""

    def test_sem_null_handler_quando_logger_none(self):
        """Quando logger=None, não deve adicionar NullHandler."""
        from anonimizar._common.overlap import remove_overlap_positions

        entities = [
            {"label": "CPF", "start_position": 0, "end_position": 14, "text": "123.456.789-09"},
        ]

        # Captura o logger usado internamente
        with patch("anonimizar._common.overlap.logging.getLogger") as mock_get_logger:
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance

            remove_overlap_positions(entities, logger=None)

            # Verifica que NÃO chamou addHandler com NullHandler
            for call in mock_logger_instance.addHandler.call_args_list:
                handler = call[0][0]
                assert not isinstance(handler, logging.NullHandler), (
                    "NullHandler não deveria ser adicionado em overlap.py"
                )
