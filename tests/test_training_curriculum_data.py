"""Testes para o módulo _training/curriculum_data.py.

Cobre a construção de datasets em janelas de dificuldade (w0/w1/w2/full/w00),
os filtros obrigatórios herdados da estória 942 e a persistência em joblib.
"""

import logging

import pandas as pd
import pytest

from anonimizar._training.curriculum_data import (
    build_context_window_dataset,
    build_curriculum_datasets,
    build_full_text_dataset,
    build_pure_entity_dataset,
    load_curriculum_datasets,
    save_curriculum_datasets,
)

__all__ = [
    "TestBuildCurriculumDatasets",
    "TestContextWindowDataset",
    "TestFiltros",
    "TestPureEntityDataset",
    "TestSaveLoad",
]

_LOGGER = logging.getLogger(__name__)

# Textos de exemplo com três parágrafos (separados por quebra de linha simples
# e linha vazia) e entidades em parágrafos distintos.
_TEXTO_1 = "CPF 123.456.789-09\n\ncontexto sem entidade\n\nemail joao@mail.com fim"
_TEXTO_2 = "Segundo doc sem nada\n\nCPF 987.654.321-00 aqui"


def _fixture_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monta DataFrames de textos e entidades de exemplo (formato do pacote)."""
    df_textos = pd.DataFrame({"id": [1, 2], "text": [_TEXTO_1, _TEXTO_2]})
    df_entidades = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "start": [4, 49, 26],
            "end": [18, 62, 40],
            "entidade": ["CPF", "EMAIL", "CPF"],
        }
    )
    return df_textos, df_entidades


class TestContextWindowDataset:
    """Testes do build_context_window_dataset (w0/w1/w2/full)."""

    def test_w0_apenas_paragrafo_com_entidade(self):
        """w0 = só o parágrafo: cada exemplo deve ser um parágrafo com entidades."""
        df_t, df_e = _fixture_dfs()
        exemplos = build_context_window_dataset(df_t, df_e, window=0, logger=_LOGGER)

        # Doc 1: 2 parágrafos com entidade (CPF no 1º e EMAIL no 3º);
        # Doc 2: 1 parágrafo com entidade (CPF no 2º).
        assert len(exemplos) == 3

        primeiro_paragrafo = _TEXTO_1.split("\n")[0]
        texto, annotations = exemplos[0]
        assert texto == primeiro_paragrafo
        start, end, label = annotations["entities"][0]
        assert texto[start:end] == "123.456.789-09"
        assert label == "CPF"

    def test_w1_reposiciona_offsets_no_texto_montado(self):
        """w1 = ±1 parágrafo: offsets devem bater com o texto montado (\"\\n\\n\")."""
        df_t, df_e = _fixture_dfs()
        exemplos = build_context_window_dataset(df_t, df_e, window=1, logger=_LOGGER)

        textos = [texto for texto, _ in exemplos]
        assert any("\n\n" in texto for texto in textos), "Janela w1 deve unir parágrafos"
        for texto, annotations in exemplos:
            for start, end, _label in annotations["entities"]:
                assert texto[start:end] in ("123.456.789-09", "joao@mail.com", "987.654.321-00")
                assert texto.count(texto[start:end]) >= 1

    def test_full_documento_inteiro(self):
        """window='full' retorna o documento inteiro com todas as entidades."""
        df_t, df_e = _fixture_dfs()
        exemplos = build_context_window_dataset(df_t, df_e, window="full", logger=_LOGGER)

        assert len(exemplos) == 2
        texto, annotations = exemplos[0]
        assert texto == _TEXTO_1
        assert len(annotations["entities"]) == 2
        for start, end, _label in annotations["entities"]:
            assert _TEXTO_1[start:end] in ("123.456.789-09", "joao@mail.com")

    def test_full_equivale_a_build_full_text_dataset(self):
        """window='full' deve ser equivalente a build_full_text_dataset."""
        df_t, df_e = _fixture_dfs()
        direto = build_context_window_dataset(df_t, df_e, window="full", logger=_LOGGER)
        wrapper = build_full_text_dataset(df_t, df_e, logger=_LOGGER)
        assert direto == wrapper

    def test_window_invalida_levanta_valueerror(self):
        """window inválida (negativa ou desconhecida) levanta ValueError."""
        df_t, df_e = _fixture_dfs()
        for window in (-1, "w5"):
            with pytest.raises(ValueError, match="window"):
                build_context_window_dataset(df_t, df_e, window=window, logger=_LOGGER)

    def test_window_tipo_invalido_levanta_typeerror(self):
        """window com tipo inválido (ex.: float) levanta TypeError."""
        df_t, df_e = _fixture_dfs()
        with pytest.raises(TypeError, match="window"):
            build_context_window_dataset(df_t, df_e, window=1.5, logger=_LOGGER)

    def test_aceita_colunas_da_942(self):
        """Aceita colunas start_entidade/end_entidade/tp_entidade (formato 942)."""
        df_textos = pd.DataFrame({"id_documento": [1], "text": [_TEXTO_1]})
        df_entidades = pd.DataFrame(
            {
                "id_documento": [1],
                "start_entidade": [48],
                "end_entidade": [59],
                "tp_entidade": ["EMAIL"],
            }
        )
        exemplos = build_context_window_dataset(df_textos, df_entidades, window=0, logger=_LOGGER)
        assert len(exemplos) == 1
        assert exemplos[0][1]["entities"][0][2] == "EMAIL"

    def test_colunas_ausentes_levantam_valueerror(self):
        """Falta de colunas obrigatórias levanta ValueError."""
        df_textos = pd.DataFrame({"id": [1], "texto": [_TEXTO_1]})
        _df_textos, df_entidades = _fixture_dfs()
        with pytest.raises(ValueError, match="df_textos"):
            build_context_window_dataset(df_textos, df_entidades, window=0, logger=_LOGGER)

        df_t, df_entidades_invalidas = _fixture_dfs()
        df_entidades_invalidas = df_entidades_invalidas[["id", "start", "end"]]
        with pytest.raises(ValueError, match="df_entidades"):
            build_context_window_dataset(df_t, df_entidades_invalidas, window=0, logger=_LOGGER)


class TestPureEntityDataset:
    """Testes do build_pure_entity_dataset (w00)."""

    def test_entidade_isolada_de_ponta_a_ponta(self):
        """Cada entidade vira amostra isolada com [(0, len, label)]."""
        df_t, df_e = _fixture_dfs()
        exemplos = build_pure_entity_dataset(df_t, df_e, logger=_LOGGER)

        assert len(exemplos) == 3
        textos = [texto for texto, _ in exemplos]
        assert textos == ["123.456.789-09", "joao@mail.com", "987.654.321-00"]
        for texto, annotations in exemplos:
            start, end, _label = annotations["entities"][0]
            assert (start, end) == (0, len(texto))

    def test_oversample_repetica_fator(self):
        """Oversampling repete amostras do label pelo fator informado."""
        df_t, df_e = _fixture_dfs()
        exemplos = build_pure_entity_dataset(df_t, df_e, oversample={"CPF": 3}, logger=_LOGGER)
        textos = [texto for texto, _ in exemplos]
        assert textos.count("123.456.789-09") == 3
        assert textos.count("987.654.321-00") == 3
        assert textos.count("joao@mail.com") == 1


class TestFiltros:
    """Testes dos filtros obrigatórios da 942."""

    def test_tem_erro_remove_documento_inteiro(self):
        """Documentos com TEM_ERRO=True são removidos por completo."""
        df_t, df_e = _fixture_dfs()
        df_e["TEM_ERRO"] = False
        df_e.loc[df_e["id"] == 2, "TEM_ERRO"] = True
        exemplos = build_context_window_dataset(df_t, df_e, window="full", logger=_LOGGER)
        assert len(exemplos) == 1
        assert exemplos[0][0] == _TEXTO_1

    def test_label_remover_descartada(self):
        """Entidades com label contendo '_remover' são descartadas."""
        df_t, df_e = _fixture_dfs()
        df_e.loc[2, "entidade"] = "CPF_remover"
        exemplos = build_context_window_dataset(df_t, df_e, window="full", logger=_LOGGER)
        labels = [label for _, annotations in exemplos for _, _, label in annotations["entities"]]
        assert "CPF_remover" not in labels
        assert len(labels) == 2, "Apenas as entidades do doc 1 devem permanecer"

    def test_entidades_invalidas_e_duplicadas_removidas(self):
        """Spans com start >= end e entidades duplicadas são removidos."""
        df_t, df_e = _fixture_dfs()
        df_e = pd.concat(
            [df_e, pd.DataFrame({"id": [1], "start": [17], "end": [10], "entidade": ["CPF"]})],
            ignore_index=True,
        )
        df_e = pd.concat(
            [df_e, pd.DataFrame({"id": [1], "start": [4], "end": [18], "entidade": ["CPF"]})],
            ignore_index=True,
        )
        exemplos = build_context_window_dataset(df_t, df_e, window="full", logger=_LOGGER)
        _, annotations = exemplos[0]
        cpf_count = sum(1 for _, _, label in annotations["entities"] if label == "CPF")
        assert cpf_count == 1, "Duplicata de CPF deve ser removida"


class TestBuildCurriculumDatasets:
    """Testes do build_curriculum_datasets."""

    def test_gera_janelas_default_em_conjunto_default(self):
        """Gera w0/w1/w2/full sob o conjunto 'default'."""
        df_t, df_e = _fixture_dfs()
        datasets = build_curriculum_datasets(df_t, df_e, logger=_LOGGER)
        assert set(datasets) == {"default"}
        assert set(datasets["default"]) == {"w0", "w1", "w2", "full"}
        assert all(len(exemplos) > 0 for exemplos in datasets["default"].values())

    def test_include_pure_adiciona_w00(self):
        """include_pure=True gera a janela w00."""
        df_t, df_e = _fixture_dfs()
        datasets = build_curriculum_datasets(df_t, df_e, include_pure=True, logger=_LOGGER)
        assert "w00" in datasets["default"]

    def test_janela_desconhecida_levanta_valueerror(self):
        """Janela fora do mapeamento levanta ValueError."""
        df_t, df_e = _fixture_dfs()
        with pytest.raises(ValueError, match="Janela desconhecida"):
            build_curriculum_datasets(df_t, df_e, windows=("w9",), logger=_LOGGER)


class TestSaveLoad:
    """Testes de persistência/recarga em joblib."""

    def test_roundtrip_tuplas(self, tmp_path):
        """Salvar e carregar datasets de tuplas preserva o conteúdo."""
        df_t, df_e = _fixture_dfs()
        datasets = build_curriculum_datasets(df_t, df_e, windows=("w0", "full"), logger=_LOGGER)
        caminho = tmp_path / "datasets.joblib"

        save_curriculum_datasets(datasets, caminho)
        carregado = load_curriculum_datasets(caminho)

        assert carregado == datasets

    def test_carrega_formato_942_dataframe(self, tmp_path):
        """Carrega joblib no formato da 942 (DataFrames com text/entities)."""
        caminho = tmp_path / "datasets_942.joblib"
        df_sample = pd.DataFrame(
            {
                "text": ["CPF 123.456.789-09", "email joao@mail.com fim"],
                "entities": [[(4, 18, "CPF")], [(6, 19, "EMAIL")]],
            }
        )
        try:
            import joblib
        except ImportError:  # pragma: no cover
            pytest.skip("joblib não instalado")
        joblib.dump({"default": {"w0": df_sample}}, str(caminho))

        carregado = load_curriculum_datasets(caminho)
        amostras = carregado["default"]["w0"]
        assert len(amostras) == 2
        por_texto = dict(amostras)
        assert por_texto["CPF 123.456.789-09"] == {"entities": [(4, 18, "CPF")]}
        assert por_texto["email joao@mail.com fim"] == {"entities": [(6, 19, "EMAIL")]}

    def test_estrutura_invalida_levanta_typeerror(self, tmp_path):
        """Estrutura carregada com tipo inesperado levanta TypeError."""
        caminho = tmp_path / "invalido.joblib"
        try:
            import joblib
        except ImportError:  # pragma: no cover
            pytest.skip("joblib não instalado")
        joblib.dump(["nao", "é", "dict"], str(caminho))

        with pytest.raises(TypeError, match="dict"):
            load_curriculum_datasets(caminho)
