"""Testes para o módulo SeiAnonimizarEvaluation."""

import pandas as pd
import pytest

from anonimizar.sei_anonimizar_evaluation import SeiAnonimizarEvaluation


class TestEvaluationInit:
    """Testes para inicialização do avaliador."""

    def test_init_default(self):
        """Testa inicialização com parâmetros padrão."""
        evaluator = SeiAnonimizarEvaluation()
        assert evaluator.overlap_threshold == 0.8
        assert evaluator.beta == 2.0
        assert evaluator.entity_mapping is not None

    def test_init_custom_params(self):
        """Testa inicialização com parâmetros customizados."""
        evaluator = SeiAnonimizarEvaluation(overlap_threshold=0.7, beta=1.0)
        assert evaluator.overlap_threshold == 0.7
        assert evaluator.beta == 1.0

    def test_init_invalid_threshold(self):
        """Testa inicialização com threshold inválido."""
        with pytest.raises(ValueError, match="overlap_threshold deve estar entre 0.0 e 1.0"):
            SeiAnonimizarEvaluation(overlap_threshold=1.5)

    def test_init_invalid_beta(self):
        """Testa inicialização com beta inválido."""
        with pytest.raises(ValueError, match="beta deve ser maior que 0"):
            SeiAnonimizarEvaluation(beta=-1.0)


class TestPredictionsIO:
    """Testes de entrada e saída de predições."""

    def test_set_predictions_missing_cols(self):
        """Testa erro ao definir predições com colunas ausentes."""
        evaluator = SeiAnonimizarEvaluation()
        preds = pd.DataFrame({"id": [1], "tp_entidade": ["CPF"]})
        with pytest.raises(ValueError, match="Colunas ausentes nas predições"):
            evaluator.set_predictions(preds)

    def test_set_predictions_ok_and_mapping(self):
        """Testa definição de predições válidas e aplicação do mapeamento de entidades."""
        evaluator = SeiAnonimizarEvaluation()
        preds = pd.DataFrame(
            {
                "id": [1],
                "tp_entidade": ["E-mail"],
                "start_entidade": [10],
                "end_entidade": [15],
            }
        )
        evaluator.set_predictions(preds)
        assert evaluator.df_predictions is not None
        assert evaluator.df_predictions.loc[0, "tp_entidade"] == "EMAIL"

    def test_save_predictions_without_data_raises(self, tmp_path):
        """Testa erro ao salvar sem predições definidas."""
        evaluator = SeiAnonimizarEvaluation()
        with pytest.raises(ValueError, match="Nenhuma predição para salvar"):
            evaluator.save_predictions(tmp_path / "p.parquet")

    @pytest.mark.parametrize("fmt,ext", [("parquet", ".parquet"), ("csv", ".csv"), ("json", ".json")])
    def test_save_and_load_predictions_roundtrip(self, tmp_path, fmt, ext):
        """Testa ciclo de salvar e carregar predições em vários formatos."""
        evaluator = SeiAnonimizarEvaluation()
        preds = pd.DataFrame(
            {
                "id": [1, 2],
                "tp_entidade": ["EMAIL", "CPF"],
                "start_entidade": [0, 10],
                "end_entidade": [5, 21],
                "text_entidade": ["a@b.com", "123.456.789-09"],
                "detected_by": ["regex", "modelo"],
            }
        )
        evaluator.set_predictions(preds)
        out = tmp_path / f"preds{ext}"
        evaluator.save_predictions(out, format=fmt)
        assert out.exists()

        # load_predictions
        evaluator2 = SeiAnonimizarEvaluation()
        loaded = evaluator2.load_predictions(out)
        assert isinstance(loaded, pd.DataFrame)
        assert set(["id", "tp_entidade", "start_entidade", "end_entidade"]).issubset(loaded.columns)

    def test_load_predictions_missing_file(self, tmp_path):
        """Testa erro ao carregar arquivo de predições inexistente."""
        evaluator = SeiAnonimizarEvaluation()
        with pytest.raises(FileNotFoundError, match="não encontrado"):
            evaluator.load_predictions(tmp_path / "absent.parquet")

    def test_load_predictions_unsupported_format(self, tmp_path):
        """Testa erro ao carregar predições em formato não suportado."""
        evaluator = SeiAnonimizarEvaluation()
        p = tmp_path / "preds.txt"
        p.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="Formato não suportado"):
            evaluator.load_predictions(p)


class TestDataLoading:
    """Testes para carregamento de dados."""

    def test_load_data_success(self, sample_evaluation_data):
        """Testa carregamento bem-sucedido de dados."""
        texts, ground_truth = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()

        evaluator.load_data(texts, ground_truth)

        assert evaluator.df_texts is not None
        assert evaluator.df_ground_truth is not None
        assert len(evaluator.df_texts) == 3
        assert len(evaluator.df_ground_truth) == 3

    def test_load_data_missing_columns(self):
        """Testa carregamento com colunas faltando."""
        texts = pd.DataFrame({"wrong_column": [1, 2, 3]})
        ground_truth = pd.DataFrame({"id": [1, 2, 3]})

        evaluator = SeiAnonimizarEvaluation()

        with pytest.raises(ValueError, match="Colunas ausentes"):
            evaluator.load_data(texts, ground_truth)


class TestLoadDataFromFiles:
    def test_load_data_from_parquet_and_remove(self, tmp_path):
        texts = pd.DataFrame(
            {
                "id": [1, 2],
                "text": ["Email a@b.com", "CPF 123.456.789-09"],
            }
        )
        gt = pd.DataFrame(
            {
                "id": [1, 2, 2],
                "tp_entidade": ["E-mail", "CPF", "CPF"],
                "start_entidade": [6, 4, 0],
                "end_entidade": [12, 17, 1],
                "remove": [0, 1, 0],
            }
        )
        texts_p = tmp_path / "texts.parquet"
        gt_p = tmp_path / "gt.parquet"
        texts.to_parquet(texts_p, index=False)
        gt.to_parquet(gt_p, index=False)

        evaluator = SeiAnonimizarEvaluation(texts_path=str(texts_p), ground_truth_path=str(gt_p))
        assert evaluator.df_texts is not None
        assert evaluator.df_ground_truth is not None
        assert len(evaluator.df_ground_truth) == 2

    def test_load_data_from_files_missing_cols(self, tmp_path):
        texts = pd.DataFrame({"wrong": [1]})
        gt = pd.DataFrame({"id": [1]})
        texts_p = tmp_path / "texts.parquet"
        gt_p = tmp_path / "gt.parquet"
        texts.to_parquet(texts_p, index=False)
        gt.to_parquet(gt_p, index=False)

        with pytest.raises(ValueError, match="Colunas ausentes"):
            SeiAnonimizarEvaluation(texts_path=str(texts_p), ground_truth_path=str(gt_p))

    def test_load_data_from_files_unsupported_format(self, tmp_path):
        t = tmp_path / "texts.txt"
        g = tmp_path / "gt.txt"
        t.write_text("", encoding="utf-8")
        g.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Formato não suportado"):
            SeiAnonimizarEvaluation(texts_path=str(t), ground_truth_path=str(g))


class TestPredictionExtraction:
    """Testes para extração de predições."""

    def test_extract_predictions(self, sample_evaluation_data, anonymizer):
        """Testa extração de predições."""
        texts, ground_truth = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()
        evaluator.load_data(texts, ground_truth)

        predictions = evaluator.extract_predictions(anonymizer)

        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) >= 0  # Pode ser 0 se nenhuma entidade for detectada

        if len(predictions) > 0:
            required_cols = ["id", "tp_entidade", "start_entidade", "end_entidade"]
            assert all(col in predictions.columns for col in required_cols)

    def test_extract_predictions_no_data(self, anonymizer):
        """Testa extração sem dados carregados."""
        evaluator = SeiAnonimizarEvaluation()

        with pytest.raises(ValueError, match="Dados de texto não carregados"):
            evaluator.extract_predictions(anonymizer)

    def test_save_empty_prediction(self, temp_dir):
        """Testa salvar sem predicao."""
        evaluator = SeiAnonimizarEvaluation()

        with pytest.raises(ValueError, match="Nenhuma predição para salvar"):
            evaluator.save_predictions(output_path=temp_dir)


class TestOverlapCalculation:
    """Testes para cálculo de sobreposição."""

    @pytest.mark.parametrize(
        "start_true,end_true,start_pred,end_pred,expected",
        [
            (0, 10, 0, 10, 1.0),  # Sobreposição completa
            (0, 10, 5, 15, 0.5),  # Sobreposição parcial
            (0, 10, 10, 20, 0.0),  # Sem sobreposição
            (0, 10, 2, 8, 0.6),  # Predição dentro da verdade
            (2, 8, 0, 10, 1.0),  # Verdade dentro da predição
        ],
    )
    def test_calculate_overlap(self, start_true, end_true, start_pred, end_pred, expected):
        """Testa cálculo de sobreposição."""
        evaluator = SeiAnonimizarEvaluation()
        result = evaluator.calculate_overlap(start_true, end_true, start_pred, end_pred)
        assert abs(result - expected) < 0.001

    def test_overlap_with_nans(self):
        evaluator = SeiAnonimizarEvaluation()
        assert evaluator.calculate_overlap(0, 10, float("nan"), 5) == 0.0
        assert evaluator.calculate_overlap(float("nan"), 10, 0, 5) == 0.0

    @pytest.mark.parametrize(
        "args",
        [
            (0, 10, 11, 5),  # start_pred > end_pred
            (10, 5, 0, 3),  # start_true > end_true
        ],
    )
    def test_overlap_invalid_order_raises(self, args):
        evaluator = SeiAnonimizarEvaluation()
        with pytest.raises(ValueError, match="não pode iniciar depois do término"):
            evaluator.calculate_overlap(*args)


class TestModelEvaluation:
    """Testes para avaliação de modelo."""

    def test_evaluate_model_basic(self, sample_evaluation_data, anonymizer):
        """Testa avaliação básica do modelo."""
        texts, ground_truth = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()
        evaluator.load_data(texts, ground_truth)

        results = evaluator.evaluate_model(anonymizer)

        assert isinstance(results, dict)
        # Verifica se tem métricas para entidades ou consolidadas
        assert len(results) > 0

        for entity_type, metrics in results.items():
            assert "fbeta" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "tp" in metrics
            assert "fp" in metrics
            assert "fn" in metrics

    def test_get_summary_report(self, sample_evaluation_data, anonymizer):
        """Testa geração de relatório resumido."""
        texts, ground_truth = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()
        evaluator.load_data(texts, ground_truth)
        evaluator.evaluate_model(anonymizer)

        report = evaluator.get_summary_report()

        assert isinstance(report, str)
        assert "RELATÓRIO DE AVALIAÇÃO" in report

    def test_get_detailed_report(self, sample_evaluation_data, anonymizer):
        """Testa geração de relatório detalhado."""
        texts, ground_truth = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()
        evaluator.load_data(texts, ground_truth)
        evaluator.evaluate_model(anonymizer)

        detailed = evaluator.get_detailed_report()

        assert isinstance(detailed, pd.DataFrame)
        if len(detailed) > 0:
            expected_cols = ["tp_entidade", "fbeta", "precision", "recall"]
            assert all(col in detailed.columns for col in expected_cols)


class TestClassificationCases:
    def test_get_classification_cases_requires_data(self):
        evaluator = SeiAnonimizarEvaluation()
        with pytest.raises(ValueError, match="Predições e ground truth devem estar carregados"):
            evaluator.get_classification_cases()

    def test_get_classification_cases_invalid_case_type(self, sample_evaluation_data):
        texts, gt = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()
        evaluator.load_data(texts, gt)
        preds = pd.DataFrame(
            {
                "id": gt["id"],
                "tp_entidade": gt["tp_entidade"],
                "start_entidade": gt["start_entidade"],
                "end_entidade": gt["end_entidade"],
            }
        )
        evaluator.set_predictions(preds)
        with pytest.raises(ValueError, match="case_type deve ser um de"):
            evaluator.get_classification_cases(case_type="weird")

    @pytest.mark.parametrize("case", ["tp", "fp", "fn", "tn", "all"])
    def test_get_classification_cases_filters(self, sample_evaluation_data, case):
        texts, gt = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()
        evaluator.load_data(texts, gt)

        preds = pd.DataFrame(
            {
                "id": [texts.loc[0, "id"], texts.loc[1, "id"]],
                "tp_entidade": [gt.loc[0, "tp_entidade"], gt.loc[1, "tp_entidade"]],
                "start_entidade": [gt.loc[0, "start_entidade"], gt.loc[1, "start_entidade"] + 100],
                "end_entidade": [gt.loc[0, "end_entidade"], gt.loc[1, "end_entidade"] + 100],
                "text_entidade": ["x", "y"],
            }
        )
        evaluator.set_predictions(preds)
        out = evaluator.get_classification_cases(case_type=case)
        assert isinstance(out, pd.DataFrame)


class TestGenerateComparisonData:
    def test_generate_requires_gt_loaded(self):
        evaluator = SeiAnonimizarEvaluation()
        with pytest.raises(ValueError, match="ground truth não carregados"):
            evaluator.generate_comparison_data(pd.DataFrame())

    def test_generate_missing_cols_in_predictions(self, sample_evaluation_data):
        texts, gt = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()
        evaluator.load_data(texts, gt)
        preds = pd.DataFrame({"id": [1], "tp_entidade": ["EMAIL"]})  # faltam start/end
        with pytest.raises(ValueError, match="Colunas ausentes nas predições"):
            evaluator.generate_comparison_data(preds)

    def test_generate_basic_flow(self, sample_evaluation_data):
        texts, gt = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()
        evaluator.load_data(texts, gt)
        preds = pd.DataFrame(
            {
                "id": texts["id"],
                "tp_entidade": ["EMAIL"] * len(texts),
                "start_entidade": [1] * len(texts),
                "end_entidade": [1] * len(texts),
                "text_entidade": ["x"] * len(texts),
            }
        )
        comp = evaluator.generate_comparison_data(preds, overlap_threshold=0.8)
        assert isinstance(comp, pd.DataFrame)
        assert {"y_true", "y_pred", "overlap"}.issubset(comp.columns)


class TestCompareReports:
    def test_compare_reports_missing_cols_raises(self):
        evaluator = SeiAnonimizarEvaluation()
        current = pd.DataFrame({"tp_entidade": ["CPF"]})
        previous = pd.DataFrame({"tp_entidade": ["CPF"]})
        with pytest.raises(ValueError, match="Colunas ausentes"):
            evaluator.compare_reports(current, previous)

    def test_compare_reports_ok(self):
        evaluator = SeiAnonimizarEvaluation()
        current = pd.DataFrame(
            {
                "tp_entidade": ["CPF"],
                "fbeta": [0.8],
                "precision": [0.75],
                "recall": [0.85],
                "overlap_threshold": [0.8],
                "beta": [2.0],
            }
        )
        previous = pd.DataFrame(
            {
                "tp_entidade": ["CPF"],
                "fbeta": [0.7],
                "precision": [0.70],
                "recall": [0.80],
                "overlap_threshold": [0.8],
                "beta": [2.0],
            }
        )
        comp = evaluator.compare_reports(current, previous)
        assert "fbeta_diff" in comp.columns
        assert bool(comp.loc[0, "fbeta_melhorou"]) is True


class TestEvaluateModelPaths:
    def test_evaluate_with_given_predictions(self, sample_evaluation_data):
        texts, gt = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()
        evaluator.load_data(texts, gt)
        preds = pd.DataFrame(
            {
                "id": gt["id"],
                "tp_entidade": gt["tp_entidade"],
                "start_entidade": gt["start_entidade"],
                "end_entidade": gt["end_entidade"],
            }
        )
        results = evaluator.evaluate_model(predictions=preds)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_evaluate_without_anonymizer_or_predictions_raises(self, sample_evaluation_data):
        texts, gt = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()
        evaluator.load_data(texts, gt)
        with pytest.raises(ValueError, match="Forneça 'anonymizer' para extração ou 'predictions'"):
            evaluator.evaluate_model()


class TestEvaluateMultipleThresholds:
    def test_multiple_thresholds_with_predictions(self, sample_evaluation_data):
        texts, gt = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()
        evaluator.load_data(texts, gt)
        preds = pd.DataFrame(
            {
                "id": gt["id"],
                "tp_entidade": gt["tp_entidade"],
                "start_entidade": gt["start_entidade"],
                "end_entidade": gt["end_entidade"],
            }
        )
        df = evaluator.evaluate_multiple_thresholds(
            predictions=preds, overlap_thresholds=[0.7, 0.9], beta_values=[1, 2]
        )
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert {"tp_entidade", "fbeta", "precision", "recall", "overlap_threshold", "beta"}.issubset(df.columns)

    def test_multiple_thresholds_requires_data(self):
        evaluator = SeiAnonimizarEvaluation()
        with pytest.raises(ValueError, match="Dados não carregados"):
            evaluator.evaluate_multiple_thresholds()


class TestErrorAnalysis:
    def test_error_analysis_requires_data(self):
        evaluator = SeiAnonimizarEvaluation()
        with pytest.raises(ValueError, match="Execute extract_predictions"):
            evaluator.get_error_analysis()

    def test_error_analysis_structure(self, sample_evaluation_data):
        texts, gt = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()
        evaluator.load_data(texts, gt)
        preds = pd.DataFrame(
            {
                "id": gt["id"],
                "tp_entidade": gt["tp_entidade"],
                "start_entidade": gt["start_entidade"],
                "end_entidade": gt["end_entidade"],
            }
        )
        evaluator.set_predictions(preds)
        analysis = evaluator.get_error_analysis()
        assert isinstance(analysis, dict)
        for key in [
            "entity_type",
            "total_cases",
            "summary",
            "tp_examples",
            "fp_examples",
            "fn_examples",
            "overlap_distribution",
        ]:
            assert key in analysis


class TestSaveClassificationCases:
    def test_save_classification_cases_formats(self, sample_evaluation_data, tmp_path):
        texts, gt = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()
        evaluator.load_data(texts, gt)
        preds = pd.DataFrame(
            {
                "id": gt["id"],
                "tp_entidade": gt["tp_entidade"],
                "start_entidade": gt["start_entidade"],
                "end_entidade": gt["end_entidade"],
            }
        )
        evaluator.set_predictions(preds)

        for fmt, ext in [("parquet", ".parquet"), ("csv", ".csv"), ("json", ".json")]:
            out = tmp_path / f"cases{ext}"
            evaluator.save_classification_cases(out, format=fmt)
            assert out.exists()

    def test_save_classification_cases_invalid_format(self, sample_evaluation_data, tmp_path):
        texts, gt = sample_evaluation_data
        evaluator = SeiAnonimizarEvaluation()
        evaluator.load_data(texts, gt)
        preds = pd.DataFrame(
            {
                "id": gt["id"],
                "tp_entidade": gt["tp_entidade"],
                "start_entidade": gt["start_entidade"],
                "end_entidade": gt["end_entidade"],
            }
        )
        evaluator.set_predictions(preds)
        with pytest.raises(ValueError, match="Formato não suportado"):
            evaluator.save_classification_cases(tmp_path / "cases.txt", format="txt")
