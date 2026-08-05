"""Testes para cross-validation do módulo Trainer."""

import json

import pandas as pd
import pytest

from anonimizar import Trainer


class TestCrossValidate:
    """Testes para o método cross_validate."""

    @pytest.fixture
    def sample_entidades(self):
        """Fixture para dados de entidades de exemplo."""
        return pd.DataFrame(
            {
                "id": [1, 1, 2, 3, 4, 4, 4],
                "tp_entidade": ["CPF", "EMAIL", "CPF", "TELEFONE", "CPF", "EMAIL", "TELEFONE"],
                "start_entidade": [4, 19, 4, 5, 4, 19, 44],
                "end_entidade": [18, 33, 18, 19, 18, 33, 57],
            }
        )

    @pytest.fixture
    def sample_textos(self):
        """Fixture para dados de textos de exemplo."""
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "text": [
                    "CPF 123.456.789-09 email@test.com",
                    "CPF 987.654.321-00",
                    "Tel: (11)99999-9999",
                    "CPF 123.456.789-09 email@test.com Telefone: 41 99999 5244",
                ],
            }
        )

    def test_cross_validate_basic(self, sample_entidades, sample_textos, temp_dir):
        """Testa execução básica de cross validation."""
        trainer = Trainer(
            model_name=None, output_dir=str(temp_dir / "custom_model"), labels=["CPF", "EMAIL", "TELEFONE"]
        )

        reports, summaries, results, _ = trainer.cross_validate(
            df_entidades=sample_entidades,
            df_textos=sample_textos,
            n_splits=2,
            stratified=False,
            output_dir=str(temp_dir / "custom_model"),
            n_jobs=1,
            train_params={"n_iter": 5, "drop": 0.1, "batch_size": 2},
            eval_params={"overlap_threshold": 0.8, "beta": 2.0},
            add_data_params={
                "errors": "coerce",
                "auto_clean": True,
                "strict_clean": True,
                "keep_empty_entities": False,
            },
            replace=True,
        )

        assert len(reports) == 2
        assert len(summaries) == 2
        assert len(results) == 2
        assert isinstance(reports[0], pd.DataFrame)
        assert isinstance(summaries[0], dict)

        # CORRIGIDO: Filtrar apenas diretórios
        fold_dirs = [d for d in (temp_dir / "custom_model").glob("fold_*") if d.is_dir()]
        assert len(fold_dirs) == 2

        for fold_dir in fold_dirs:
            model_dir = fold_dir / "model"
            assert model_dir.exists()
            assert any(model_dir.glob("*"))

    def test_cross_validate_stratified(self, sample_entidades, sample_textos, temp_dir):
        """Testa cross validation com estratificação."""
        trainer = Trainer(model_name=None, labels=["CPF", "EMAIL", "TELEFONE"])

        reports, summaries, _, _ = trainer.cross_validate(
            df_entidades=sample_entidades,
            df_textos=sample_textos,
            n_splits=2,
            stratified=True,
            features=["CPF", "EMAIL", "TELEFONE"],
            output_dir=str(temp_dir),
            n_jobs=1,
            train_params={"n_iter": 1},
            eval_params={"overlap_threshold": 0.8, "beta": 2.0},
            add_data_params={
                "errors": "coerce",
                "auto_clean": True,
                "strict_clean": True,
                "keep_empty_entities": False,
            },
            replace=True,
        )

        assert len(reports) == 2
        assert len(summaries) == 2

    def test_cross_validate_with_n_jobs(self, sample_entidades, sample_textos, temp_dir):
        """Testa execução paralela com n_jobs > 1."""
        trainer = Trainer(
            model_name=None, output_dir=str(temp_dir / "custom_model"), labels=["CPF", "EMAIL", "TELEFONE"]
        )

        reports, summaries, results, _ = trainer.cross_validate(
            df_entidades=sample_entidades,
            df_textos=sample_textos,
            n_splits=2,
            stratified=False,
            output_dir=str(temp_dir / "custom_model"),
            n_jobs=2,
            train_params={"n_iter": 5, "drop": 0.1, "batch_size": 2},
            eval_params={"overlap_threshold": 0.8, "beta": 2.0},
            add_data_params={
                "errors": "coerce",
                "auto_clean": True,
                "strict_clean": True,
                "keep_empty_entities": False,
            },
            replace=True,
        )

        assert len(reports) == 2
        assert len(summaries) == 2

    def test_cross_validate_evaluation_integration(self, sample_entidades, sample_textos, temp_dir):
        """Testa integração com avaliação em um fold simples."""
        trainer = Trainer(labels=["CPF", "EMAIL", "TELEFONE"])

        reports, summaries, _, _ = trainer.cross_validate(
            df_entidades=sample_entidades,
            df_textos=sample_textos,
            n_splits=2,
            output_dir=str(temp_dir),
            train_params={"n_iter": 1},
            eval_params={"overlap_threshold": 0.8, "beta": 2.0},
            add_data_params={
                "errors": "coerce",
                "auto_clean": True,
                "strict_clean": True,
                "keep_empty_entities": False,
            },
            replace=True,
        )

        assert len(reports) == 2
        for report in reports:
            assert isinstance(report, pd.DataFrame)
            assert "fold" in report.columns
            assert "fbeta" in report.columns

    def test_cross_validate_saves_fold_ids(self, sample_entidades, sample_textos, temp_dir):
        """Testa se os IDs dos folds são salvos corretamente."""
        trainer = Trainer(model_name=None, labels=["CPF", "EMAIL", "TELEFONE"])

        trainer.cross_validate(
            df_entidades=sample_entidades,
            df_textos=sample_textos,
            n_splits=2,
            stratified=False,
            output_dir=str(temp_dir),
            n_jobs=1,
            train_params={"n_iter": 1},
            replace=True,
        )

        fold_dirs = [d for d in temp_dir.glob("fold_*") if d.is_dir()]
        assert len(fold_dirs) == 2

        for fold_dir in fold_dirs:
            fold_ids_json = fold_dir / "fold_ids.json"
            assert fold_ids_json.exists()

            with fold_ids_json.open(encoding="utf-8") as f:
                fold_info = json.load(f)

            assert "fold" in fold_info
            assert "train_ids" in fold_info
            assert "val_ids" in fold_info
            assert "n_train" in fold_info
            assert "n_val" in fold_info
            assert isinstance(fold_info["train_ids"], list)
            assert isinstance(fold_info["val_ids"], list)
            assert fold_info["n_train"] == len(fold_info["train_ids"])
            assert fold_info["n_val"] == len(fold_info["val_ids"])

            train_csv = fold_dir / "train_ids.csv"
            assert train_csv.exists()
            df_train = pd.read_csv(train_csv)
            assert "id" in df_train.columns
            assert "split" in df_train.columns
            assert (df_train["split"] == "train").all()

            val_csv = fold_dir / "val_ids.csv"
            assert val_csv.exists()
            df_val = pd.read_csv(val_csv)
            assert "id" in df_val.columns
            assert "split" in df_val.columns
            assert (df_val["split"] == "val").all()

    def test_cross_validate_fold_ids_no_overlap(self, sample_entidades, sample_textos, temp_dir):
        """Testa se não há sobreposição entre train e val IDs em cada fold."""
        trainer = Trainer(model_name=None, labels=["CPF", "EMAIL", "TELEFONE"])

        trainer.cross_validate(
            df_entidades=sample_entidades,
            df_textos=sample_textos,
            n_splits=2,
            stratified=False,
            output_dir=str(temp_dir),
            n_jobs=1,
            train_params={"n_iter": 1},
            replace=True,
        )

        fold_dirs = sorted([d for d in temp_dir.glob("fold_*") if d.is_dir()])

        for fold_dir in fold_dirs:
            with (fold_dir / "fold_ids.json").open() as f:
                fold_info = json.load(f)

            train_set = set(fold_info["train_ids"])
            val_set = set(fold_info["val_ids"])

            # Não deve haver sobreposição
            assert len(train_set & val_set) == 0

            # União deve cobrir todos os IDs
            all_ids = set(sample_entidades["id"].unique())
            assert train_set | val_set == all_ids

    def test_cross_validate_with_jsonl_input(self, temp_dir):
        """Testa cross_validate com entrada JSONL."""
        jsonl_path = temp_dir / "data.jsonl"
        data = [
            {"text": "CPF 123.456.789-09", "labels": [[4, 18, "CPF"]]},
            {"text": "CPF 987.654.321-00", "labels": [[4, 18, "CPF"]]},
            {"text": "email@test.com", "labels": [[0, 14, "EMAIL"]]},
            {"text": "outro@email.com", "labels": [[0, 15, "EMAIL"]]},
        ]

        with jsonl_path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        trainer = Trainer(model_name=None, labels=["CPF", "EMAIL"])

        reports, summaries, _, _ = trainer.cross_validate(
            df_entidades=str(jsonl_path),
            df_textos=None,
            n_splits=2,
            stratified=False,
            output_dir=str(temp_dir / "cv_results"),
            n_jobs=1,
            train_params={"n_iter": 1},
            replace=True,
        )

        assert len(reports) == 2
        assert len(summaries) == 2

    def test_cross_validate_fold_ids_json_serialization(self, sample_entidades, sample_textos, temp_dir):
        """Testa se os IDs são corretamente serializados para JSON (sem erro de numpy.int64)."""
        trainer = Trainer(model_name=None, labels=["CPF", "EMAIL", "TELEFONE"])

        trainer.cross_validate(
            df_entidades=sample_entidades,
            df_textos=sample_textos,
            n_splits=2,
            stratified=False,
            output_dir=str(temp_dir),
            n_jobs=1,
            train_params={"n_iter": 1},
            replace=True,
        )

        fold_dirs = [d for d in temp_dir.glob("fold_*") if d.is_dir()]

        for fold_dir in fold_dirs:
            fold_ids_json = fold_dir / "fold_ids.json"

            with fold_ids_json.open(encoding="utf-8") as f:
                fold_info = json.load(f)

            for train_id in fold_info["train_ids"]:
                assert isinstance(train_id, int)
                assert type(train_id).__module__ == "builtins"

            for val_id in fold_info["val_ids"]:
                assert isinstance(val_id, int)
                assert type(val_id).__module__ == "builtins"

    def test_cross_validate_output_structure(self, sample_entidades, sample_textos, temp_dir):
        """Testa a estrutura completa de saída do cross_validate."""
        trainer = Trainer(model_name=None, labels=["CPF", "EMAIL", "TELEFONE"])

        reports, summaries, results, holdout = trainer.cross_validate(
            df_entidades=sample_entidades,
            df_textos=sample_textos,
            n_splits=2,
            output_dir=str(temp_dir),
            n_jobs=1,
            train_params={"n_iter": 1},
            replace=True,
        )

        assert reports is not None
        assert summaries is not None
        assert results is not None
        assert holdout is None or isinstance(holdout, list)

        assert (temp_dir / "all_folds_detailed.parquet").exists()
        assert (temp_dir / "fold_summaries.parquet").exists()

        fold_dirs = [d for d in temp_dir.glob("fold_*") if d.is_dir()]
        for fold_dir in fold_dirs:
            assert (fold_dir / "fold_ids.json").exists()
            assert (fold_dir / "train_ids.csv").exists()
            assert (fold_dir / "val_ids.csv").exists()
            assert (fold_dir / "model").exists()
