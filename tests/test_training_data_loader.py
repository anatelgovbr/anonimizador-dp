"""Testes para _training/data_loader.py.

Cobre branches nao exercitados:
- transform_standard_format: entities nao-lista ignorada
- transform_row_per_entity_format: NaN, erro de conversao
- val_data_to_evaluation: val_data vazio
- convert_input_data: lista com labels, dict com labels, dict sem campos,
  DataFrame sem fn, string .jsonl sem fn, string nao-.jsonl, tipo invalido
- apply_auto_clean: strict discard, empty discard, conflito discard, entidade reduzida
"""

import logging
import math

import pandas as pd
import pytest

from anonimizar._training.data_loader import (
    apply_auto_clean,
    convert_input_data,
    transform_row_per_entity_format,
    transform_standard_format,
    val_data_to_evaluation,
)


@pytest.fixture
def logger() -> logging.Logger:
    """Logger silencioso para testes."""
    log = logging.getLogger("test_training_data_loader")
    log.setLevel(logging.DEBUG)
    return log


# =============================================================================
# transform_standard_format
# =============================================================================


class TestTransformStandardFormat:
    """Testa branches de transform_standard_format."""

    def test_entities_validas(self, logger) -> None:
        """Linha com entities lista e mantida."""
        _df = pd.DataFrame({"text": ["texto"], "entities": [[(0, 5, "CPF")]]})
        result = transform_standard_format(_df, logger)
        assert len(result) == 1
        assert result[0][1]["entities"] == [(0, 5, "CPF")]

    def test_entities_nao_lista_ignorada(self, logger) -> None:
        """Linhas 73-75, 78: linha com entities nao-lista e descartada."""
        _df = pd.DataFrame({"text": ["texto_invalido", "texto_valido"], "entities": ["nao_lista", [(0, 3, "RG")]]})
        result = transform_standard_format(_df, logger)
        # Apenas a segunda linha deve aparecer
        assert len(result) == 1
        assert result[0][0] == "texto_valido"

    def test_dataframe_vazio(self, logger) -> None:
        """DataFrame vazio retorna lista vazia."""
        _df = pd.DataFrame({"text": [], "entities": []})
        result = transform_standard_format(_df, logger)
        assert result == []


# =============================================================================
# transform_row_per_entity_format
# =============================================================================


class TestTransformRowPerEntityFormat:
    """Testa branches de transform_row_per_entity_format."""

    def test_formato_basico(self, logger) -> None:
        """Formato linha por entidade basico."""
        _df = pd.DataFrame(
            {
                "texto": ["CPF 529.982.247-25", "CPF 529.982.247-25"],
                "start": [4, 4],
                "end": [18, 18],
                "entidade": ["CPF", "CPF"],
            }
        )
        result = transform_row_per_entity_format(_df, logger)
        assert len(result) == 1
        assert result[0][1]["entities"] == [(4, 18, "CPF")]

    def test_nan_ignorado(self, logger) -> None:
        """Linhas 100-101: linha com NaN em start/end/entidade e ignorada."""
        _df = pd.DataFrame(
            {
                "texto": ["texto nan", "texto nan"],
                "start": [float("nan"), 0],
                "end": [float("nan"), 5],
                "entidade": [float("nan"), "CPF"],
            }
        )
        result = transform_row_per_entity_format(_df, logger)
        # A linha com NaN deve ser ignorada; apenas a valida e mantida
        assert len(result) == 1
        ents = result[0][1]["entities"]
        for e in ents:
            assert not any(math.isnan(v) for v in e[:2] if isinstance(v, float))

    def test_deduplicacao_entidades(self, logger) -> None:
        """Entidades duplicadas no mesmo texto sao deduplicadas."""
        _df = pd.DataFrame(
            {
                "texto": ["texto", "texto"],
                "start": [0, 0],
                "end": [5, 5],
                "entidade": ["CPF", "CPF"],
            }
        )
        result = transform_row_per_entity_format(_df, logger)
        assert len(result) == 1
        assert len(result[0][1]["entities"]) == 1


# =============================================================================
# val_data_to_evaluation
# =============================================================================


class TestValDataToEvaluation:
    """Testa val_data_to_evaluation."""

    def test_vazio_lanca_erro(self, logger) -> None:
        """Linhas 177-179: val_data vazio lanca ValueError."""
        with pytest.raises(ValueError, match=r"Não existem dados de validação"):
            val_data_to_evaluation([], logger)

    def test_converte_corretamente(self, logger) -> None:
        """Converte lista de dados para DataFrames."""
        val_data = [
            ("CPF 529.982.247-25", {"entities": [(4, 18, "CPF")]}),
            ("Email joao@test.com aqui", {"entities": [(6, 19, "EMAIL")]}),
        ]
        df_texts, df_gt = val_data_to_evaluation(val_data, logger)
        assert len(df_texts) == 2
        assert len(df_gt) == 2
        assert "id" in df_texts.columns
        assert "text" in df_texts.columns
        assert "tp_entidade" in df_gt.columns

    def test_multiplas_entidades_por_texto(self, logger) -> None:
        """Texto com multiplas entidades gera varias linhas no df_gt."""
        val_data = [
            ("texto com cpf e rg", {"entities": [(10, 13, "CPF"), (16, 18, "RG")]}),
        ]
        df_texts, df_gt = val_data_to_evaluation(val_data, logger)
        assert len(df_texts) == 1
        assert len(df_gt) == 2


# =============================================================================
# convert_input_data
# =============================================================================


class TestConvertInputData:
    """Testa convert_input_data com varios formatos."""

    def test_lista_com_labels(self, logger) -> None:
        """Linhas 230-234: lista com chave 'labels' e convertida."""
        data = [{"text": "CPF 529.982.247-25", "labels": [[4, 18, "CPF"]]}]
        result = convert_input_data(data, logger)
        assert len(result) == 1
        assert result[0][1]["entities"] == [(4, 18, "CPF")]

    def test_lista_com_entities(self, logger) -> None:
        """Linha 236: lista com chave 'entities' e passada diretamente."""
        data = [{"text": "CPF", "entities": [(0, 3, "CPF")]}]
        result = convert_input_data(data, logger)
        assert len(result) == 1
        assert result[0][1]["entities"] == [(0, 3, "CPF")]

    def test_lista_sem_campos_gera_aviso(self, logger) -> None:
        """Linha 238: caso sem 'entities' ou 'labels' e logado como aviso."""
        data = [{"text": "texto", "outro": []}]
        result = convert_input_data(data, logger)
        # Sem entities nem labels -> nao e adicionado
        assert result == []

    def test_dict_com_labels(self, logger) -> None:
        """Linhas 241-246: dict unico com 'labels' e convertido."""
        data = {"text": "CPF 529.982.247-25", "labels": [[4, 18, "CPF"]]}
        result = convert_input_data(data, logger)
        assert len(result) == 1
        assert result[0][1]["entities"] == [(4, 18, "CPF")]

    def test_dict_com_entities(self, logger) -> None:
        """Linha 247-248: dict unico com 'entities'."""
        data = {"text": "CPF", "entities": [(0, 3, "CPF")]}
        result = convert_input_data(data, logger)
        assert len(result) == 1

    def test_dict_sem_campos(self, logger) -> None:
        """Linha 250: dict sem 'entities' nem 'labels'."""
        data = {"text": "texto", "outro": []}
        result = convert_input_data(data, logger)
        assert result == []

    def test_dataframe_sem_fn_lanca_erro(self, logger) -> None:
        """Linhas 253-255: DataFrame sem transform_pandas_fn lanca ValueError."""
        _df = pd.DataFrame({"text": ["texto"], "entities": [[(0, 3, "CPF")]]})
        with pytest.raises(ValueError, match=r"transform_pandas_fn"):
            convert_input_data(_df, logger)

    def test_dataframe_com_fn(self, logger) -> None:
        """DataFrame com transform_pandas_fn e convertido."""
        _df = pd.DataFrame({"text": ["texto"], "entities": [[(0, 3, "CPF")]]})
        fn = lambda _d: [("texto", {"entities": [(0, 3, "CPF")]})]  # noqa: E731
        result = convert_input_data(_df, logger, transform_pandas_fn=fn)
        assert len(result) == 1

    def test_string_jsonl_sem_fn_lanca_erro(self, logger, tmp_path) -> None:
        """Linhas 265-266: string .jsonl sem load_jsonl_fn lanca ValueError."""
        f = tmp_path / "dados.jsonl"
        f.write_text('{"text": "texto", "labels": []}', encoding="utf-8")
        with pytest.raises(ValueError, match=r"load_jsonl_fn"):
            convert_input_data(str(f), logger)

    def test_string_jsonl_com_fn(self, logger, tmp_path) -> None:
        """String .jsonl com load_jsonl_fn carrega dados."""
        f = tmp_path / "dados.jsonl"
        f.write_text('{"text": "texto", "labels": []}', encoding="utf-8")
        fn = lambda _p: [("texto", {"entities": []})]  # noqa: E731
        result = convert_input_data(str(f), logger, load_jsonl_fn=fn)
        assert len(result) == 1

    def test_string_nao_jsonl_raise(self, logger) -> None:
        """Linhas 270-274: string nao-.jsonl com errors='raise' lanca ValueError."""
        with pytest.raises(ValueError, match=r"Formato de arquivo não suportado"):
            convert_input_data("arquivo.csv", logger, errors="raise")

    def test_string_nao_jsonl_coerce_retorna_vazio(self, logger) -> None:
        """errors='coerce' com extensao invalida retorna lista vazia."""
        result = convert_input_data("arquivo.csv", logger, errors="coerce")
        assert result == []

    def test_tipo_invalido_lanca_type_error(self, logger) -> None:
        """Linha 276-278: tipo nao suportado lanca TypeError."""
        with pytest.raises(TypeError, match=r"Dados devem ser"):
            convert_input_data(12345, logger)  # type: ignore[arg-type]


# =============================================================================
# apply_auto_clean
# =============================================================================


class TestApplyAutoClean:
    """Testa apply_auto_clean com varios cenarios."""

    def _make_clean_fn(self, return_val: list):
        """Cria um mock de clean_entities_fn que retorna valor fixo."""

        def fn(text, entities, *, strict=False, resolve_conflicts="coerce", errors="coerce"):  # noqa: ARG001
            return return_val

        return fn

    def test_keep_empty_true_mantem_sem_entidades(self, logger) -> None:
        """keep_empty_entities=True mantem exemplos sem entidades."""
        data = [("texto", {"entities": [(0, 5, "CPF")]})]
        clean_fn = self._make_clean_fn([])
        result = apply_auto_clean(
            data, clean_fn, strict_clean=False, keep_empty_entities=True, resolve_conflicts="coerce", logger=logger
        )
        assert len(result) == 1
        assert result[0][1]["entities"] == []

    def test_keep_empty_false_descarta_sem_entidades(self, logger) -> None:
        """Linhas 331-332: keep_empty_entities=False descarta exemplos sem entidades."""
        data = [("texto", {"entities": [(0, 5, "CPF")]})]
        clean_fn = self._make_clean_fn([])
        result = apply_auto_clean(
            data, clean_fn, strict_clean=False, keep_empty_entities=False, resolve_conflicts="coerce", logger=logger
        )
        assert result == []

    def test_strict_clean_descarta_quando_entidades_removidas(self, logger) -> None:
        """Linhas 324-326: strict=True descarta quando todas entidades sao removidas."""
        data = [("texto", {"entities": [(0, 5, "CPF")]})]
        clean_fn = self._make_clean_fn([])
        result = apply_auto_clean(
            data, clean_fn, strict_clean=True, keep_empty_entities=True, resolve_conflicts="coerce", logger=logger
        )
        assert result == []

    def test_entidade_reduzida_logada(self, logger) -> None:
        """Linhas 337-338: exemplo mantido com menos entidades e logado."""
        data = [("texto", {"entities": [(0, 5, "CPF"), (10, 15, "RG")]})]
        clean_fn = self._make_clean_fn([(0, 5, "CPF")])  # uma entidade removida
        result = apply_auto_clean(
            data, clean_fn, strict_clean=False, keep_empty_entities=True, resolve_conflicts="coerce", logger=logger
        )
        assert len(result) == 1
        assert len(result[0][1]["entities"]) == 1

    def test_callback_recebe_opcoes_de_limpeza_nomeadas(self, logger) -> None:
        """O callback recebe as opções documentadas como argumentos nomeados."""
        received = {}

        def clean_fn(text, entities, *, strict, resolve_conflicts, errors):
            received.update(
                text=text,
                entities=entities,
                strict=strict,
                resolve_conflicts=resolve_conflicts,
                errors=errors,
            )
            return entities

        data = [("texto", {"entities": [(0, 5, "CPF")]})]
        result = apply_auto_clean(
            data,
            clean_fn,
            strict_clean=True,
            keep_empty_entities=True,
            resolve_conflicts="ignore",
            logger=logger,
            errors="raise",
        )

        assert result == data
        assert received == {
            "text": "texto",
            "entities": [(0, 5, "CPF")],
            "strict": True,
            "resolve_conflicts": "ignore",
            "errors": "raise",
        }

    def test_conflito_raise_descartado(self, logger) -> None:
        """Linhas 340-343: clean_fn que lanca ValueError de conflito e descartado."""

        def conflict_fn(text, entities, *, strict=False, resolve_conflicts="coerce", errors="coerce"):  # noqa: ARG001
            msg = "Conflitos de entidades detectados."
            raise ValueError(msg)

        data = [("texto", {"entities": [(0, 5, "CPF"), (3, 8, "RG")]})]
        result = apply_auto_clean(
            data, conflict_fn, strict_clean=False, keep_empty_entities=True, resolve_conflicts="coerce", logger=logger
        )
        assert result == []

    def test_outro_value_error_propagado(self, logger) -> None:
        """Linha 344-345: ValueError nao relacionado a conflito e re-lancado."""

        def error_fn(text, entities, *, strict=False, resolve_conflicts="coerce", errors="coerce"):  # noqa: ARG001
            msg = "Outro erro qualquer"
            raise ValueError(msg)

        data = [("texto", {"entities": [(0, 5, "CPF")]})]
        with pytest.raises(ValueError, match=r"Outro erro qualquer"):
            apply_auto_clean(
                data, error_fn, strict_clean=False, keep_empty_entities=True, resolve_conflicts="coerce", logger=logger
            )
