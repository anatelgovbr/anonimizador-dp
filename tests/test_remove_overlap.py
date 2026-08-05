from anonimizar import Anonimizar
from anonimizar._common.overlap import remove_overlap_positions
from tests._helpers import _ent, _make_logger


class TestOverlapRemoval:
    """Testes para remoção de sobreposições."""

    def test_remove_overlap_basic(self, anonymizer):
        entities = [
            {"label": "CPF", "start_position": 0, "end_position": 14},
            {"label": "TELEFONE", "start_position": 5, "end_position": 19},
        ]
        result = anonymizer.remove_overlap_positions(entities)
        assert len(result) == 1

    def test_no_overlap(self, anonymizer):
        entities = [
            {"label": "CPF", "start_position": 0, "end_position": 14},
            {"label": "EMAIL", "start_position": 20, "end_position": 35},
        ]
        result = anonymizer.remove_overlap_positions(entities)
        assert len(result) == 2

    def test_priority_when_same_span(self, model_path):
        anonymizer = Anonimizar(model_path)
        ents = [
            {"label": "EMAIL", "start_position": 10, "end_position": 20},
            {"label": "CPF", "start_position": 10, "end_position": 20},
        ]
        out = anonymizer.remove_overlap_positions(ents)
        assert len(out) == 1

    def test_contained_entity_removed(self, model_path):
        anonymizer = Anonimizar(model_path)
        ents = [
            {"label": "CPF", "start_position": 0, "end_position": 30},
            {"label": "EMAIL", "start_position": 10, "end_position": 20},
        ]
        out = anonymizer.remove_overlap_positions(ents)
        assert len(out) == 1
        assert out[0]["label"] == "CPF"

    def test_touching_entities_not_removed(self, model_path):
        anonymizer = Anonimizar(model_path)
        ents = [
            {"label": "CPF", "start_position": 0, "end_position": 10},
            {"label": "EMAIL", "start_position": 10, "end_position": 20},
        ]
        out = anonymizer.remove_overlap_positions(ents)
        assert len(out) == 2

    def test_union_when_new_extends_existing(self, model_path):
        anonymizer = Anonimizar(model_path)
        ents = [
            {"label": "EMAIL", "start_position": 5, "end_position": 10},
            {"label": "CPF", "start_position": 8, "end_position": 15},
        ]
        out = anonymizer.remove_overlap_positions(ents)
        assert len(out) == 1
        assert out[0]["label"] == "CPF"
        assert out[0]["start_position"] == 8
        assert out[0]["end_position"] == 15


class TestB19OverlapSameLabelMerge:
    """B-19: merge de spans sobrepostos só ocorre entre entidades de mesmo label."""

    def test_merge_mesmo_label(self):
        entities = [
            _ent("CPF", 0, 10),
            _ent("CPF", 5, 20),
        ]
        result = remove_overlap_positions(entities, logger=_make_logger())
        assert len(result) == 1
        assert result[0]["label"] == "CPF"

    def test_labels_diferentes_usa_prioridade(self):
        entities = [
            _ent("CPF", 0, 14),
            _ent("SIAPE", 0, 14),
        ]
        result = remove_overlap_positions(entities, logger=_make_logger())
        assert len(result) == 1
        assert result[0]["label"] == "CPF"

    def test_labels_diferentes_sobrepostos_nao_merge(self):
        entities = [
            _ent("CPF", 0, 14),
            _ent("RG", 7, 20),
        ]
        result = remove_overlap_positions(entities, logger=_make_logger())
        assert len(result) == 1

    def test_sem_overlap_mantem_ambas(self):
        entities = [
            _ent("CPF", 0, 14),
            _ent("EMAIL", 20, 35),
        ]
        result = remove_overlap_positions(entities, logger=_make_logger())
        assert len(result) == 2
