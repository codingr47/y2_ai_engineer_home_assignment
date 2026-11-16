"""Selective parsing contract, configuration, and calibration safeguards."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from scripts.calibrate_confidence import (RoutingRow, from_export, operating_point, partition,
                                         select_threshold, wilson_lower)
from src.app import create_app
from src.classifier import Classification
from src.confidence import FEATURE_NAMES, ConfidenceGate, features
from src.extraction.extractor import FieldExtractor
from src.extraction.types import Extraction
from src.local_parser import LocalParser, validate_output
from src.models import ParseResponse
from src.settings import Settings


class FixedConfidence:
    def __init__(self, score):
        self.score = score

    def predict_proba(self, X):
        return np.array([[1-self.score, self.score] for _ in X])


def gate(score=.95, threshold=.95, enabled=True):
    return ConfidenceGate({"acceptance_enabled": enabled, "threshold": threshold, "model": FixedConfidence(score)})


class FixedClassifier:
    def __init__(self, domain="נדל״ן", fields=None):
        self.domain = domain
        self.fields = fields if fields is not None else [domain + "/מס׳_חדרים"]
        self.calls = 0

    def classify(self, query):
        self.calls += 1
        return Classification(self.domain, self.fields, np.array([.98, .01, .01]),
                              np.array([.9, .1]), np.array([True, False]), np.array([.5, .5]),
                              np.array([True, True]))


def local(classifier=None, confidence=None, fallback=None):
    return LocalParser(classifier or FixedClassifier(), FieldExtractor.load(),
                       confidence if confidence is not None else gate(), fallback)


def test_threshold_equality_and_success_cache():
    parser = local()
    with TestClient(create_app(parser=parser)) as client:
        for _ in range(2):
            response = client.post("/parse", json={"q": "דירת 4 חדרים"})
            assert response.status_code == 200
            assert response.json()["params"] == {"מס׳_חדרים": 4}
            assert response.json()["confidence"] == .95
    assert parser.classifier.calls == 1


@pytest.mark.parametrize("confidence,reason", [(gate(.94), "low_confidence"),
                                               (ConfidenceGate(), "confidence_unavailable"),
                                               (gate(.99, enabled=False), "confidence_unavailable")])
def test_fallback_required_not_cached(confidence, reason):
    parser = local(confidence=confidence)
    with TestClient(create_app(parser=parser)) as client:
        for _ in range(2):
            response = client.post("/parse", json={"q": "4 חדרים"})
            assert response.status_code == 503
            assert response.json()["detail"]["code"] == "fallback_required"
            assert response.json()["detail"]["reason"] == reason
            assert "params" not in response.json()["detail"]
    assert parser.classifier.calls == 2


def test_fallback_receives_trimmed_query_and_replaces_local():
    class Fallback:
        calls = []

        async def parse(self, query):
            self.calls.append(query)
            return ParseResponse(category="רכב", params={"יצרן": "טויוטה"}, confidence=.99)

    fallback = Fallback()
    parser = local(confidence=gate(.1), fallback=fallback)
    with TestClient(create_app(parser=parser)) as client:
        response = client.post("/parse", json={"q": "  4 חדרים  "})
        assert response.status_code == 200
        assert response.json()["params"] == {"יצרן": "טויוטה"}
    assert fallback.calls == ["4 חדרים"]


def test_invalid_fallback_is_500_and_not_cached():
    class Fallback:
        async def parse(self, query):
            return ParseResponse(category="רכב", params={"unknown": "bad"}, confidence=.99)
    with TestClient(create_app(parser=local(confidence=gate(.1), fallback=Fallback()))) as client:
        assert client.post("/parse", json={"q": "4 חדרים"}).status_code == 500
        assert len(client.app.state.cache) == 0


@pytest.mark.parametrize("domain,fields,query,expected", [
    ("רכב", ["רכב/יצרנים", "רכב/דגמים"], "טויוטה קורולה", {"יצרן": "טויוטה", "דגם": "קורולה"}),
    ("יד_שנייה", ["יד_שנייה/מחיר"], "עד 2500", {"מחיר": {"max": 2500}}),
    ("נדל״ן", ["נדל״ן/מס׳_חדרים", "נדל״ן/מחיר"], "דירת 4 חדרים עד 2 מיליון",
     {"מס׳_חדרים": 4, "מחיר": {"max": 2000000}}),
])
def test_categories_and_bounds(domain, fields, query, expected):
    with TestClient(create_app(parser=local(FixedClassifier(domain, fields)))) as client:
        response = client.post("/parse", json={"q": query})
        assert response.status_code == 200
        assert response.json()["params"] == expected


@pytest.mark.parametrize("domain,query,reason", [("none", "שלום", "unsupported_category"),
                                               ("נדל״ן", "שלום", "empty_extraction")])
def test_hard_routing_reasons(domain, query, reason):
    with TestClient(create_app(parser=local(FixedClassifier(domain)))) as client:
        assert client.post("/parse", json={"q": query}).json()["detail"]["reason"] == reason


def test_missing_predictions_reach_confidence_features():
    c = FixedClassifier().classify("hello")
    vector = features(c, {c.fields[0]: Extraction(c.fields[0], reason="no_span")})
    assert vector[list(FEATURE_NAMES).index("missing_count")] == 1
    assert vector[list(FEATURE_NAMES).index("max_rejected")] == .1


@pytest.mark.parametrize("params", [{"unknown": 1}, {"מחיר": {"max": True}},
                                   {"מחיר": {"min": 10, "max": 1}}, {"מחיר": float("nan")},
                                   {"מס׳_חדרים": "4"}, {"עיר": "invented-enum"}])
def test_output_validation(params):
    with pytest.raises(ValueError):
        validate_output({"category": "נדל״ן", "params": params}, FieldExtractor.load())


def test_disabled_path_and_explicit_parser_avoid_model_load(monkeypatch):
    def fail(*args):
        raise AssertionError("model loaded")
    monkeypatch.setattr(LocalParser, "load", fail)
    for settings, parser in [(Settings(), None), (Settings(extractor_enabled=True), local())]:
        with TestClient(create_app(settings=settings, parser=parser)) as client:
            assert client.post("/parse", json={"q": "4 חדרים"}).status_code == 200


def test_enabled_startup_and_metrics(monkeypatch):
    calls = []
    def load(settings, fallback, metrics):
        calls.append(1)
        parser = local(confidence=gate(.1))
        parser.metrics = metrics
        return parser
    monkeypatch.setattr(LocalParser, "load", load)
    with TestClient(create_app(settings=Settings(extractor_enabled=True))) as client:
        assert client.post("/parse", json={"q": "4 חדרים"}).status_code == 503
        metrics = client.get("/metrics").text
        assert 'parse_routing_total{destination="fallback",reason="low_confidence"} 1.0' in metrics
        assert "parse_confidence_count 1.0" in metrics
    assert len(calls) == 1


def test_fallback_openapi_contract():
    schema = create_app().openapi()
    response = schema["paths"]["/parse"]["post"]["responses"]["503"]
    assert response["content"]["application/json"]["schema"]["$ref"].endswith("FallbackResponse")
    assert "low_confidence" in schema["components"]["schemas"]["FallbackDetail"]["properties"]["reason"]["enum"]


def test_startup_fails_missing_assets(tmp_path):
    with pytest.raises(ValueError, match="provisioned local encoder"):
        with TestClient(create_app(settings=Settings(extractor_enabled=True, e5_model_path=tmp_path / "absent"))):
            pass


def test_dotenv_and_environment_precedence(tmp_path, monkeypatch):
    from src import settings
    (tmp_path / ".env").write_text("EXTRACTOR_ENABLED=true\nEXTRACTOR_USE_TAGGER=false\n")
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    monkeypatch.setenv("EXTRACTOR_ENABLED", "false")
    monkeypatch.delenv("EXTRACTOR_USE_TAGGER", raising=False)
    assert not Settings.from_env().extractor_enabled
    assert not Settings.from_env().extractor_use_tagger
    # load_dotenv writes os.environ; restore the introduced setting after this test.
    monkeypatch.delenv("EXTRACTOR_USE_TAGGER")


@pytest.mark.parametrize("name,value", [("EXTRACTOR_ENABLED", "maybe"),
                                        ("PARSE_CONFIDENCE_THRESHOLD", "nan"),
                                        ("PARSE_CONFIDENCE_THRESHOLD", "1.1")])
def test_invalid_settings(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings.from_env()


def test_confidence_artifact_bindings_and_disabled_override(tmp_path):
    path = tmp_path / "confidence.joblib"
    artifact = {"schema": "parse-confidence-1", "bindings": {"version": "one"},
                "features": FEATURE_NAMES, "acceptance_enabled": False,
                "threshold": .9, "model": FixedConfidence(.99)}
    joblib.dump(artifact, path)
    assert not ConfidenceGate.load(path, {"version": "one"}, 0).accepts(.99)
    with pytest.raises(ValueError, match="does not match"):
        ConfidenceGate.load(path, {"version": "two"})
    assert not ConfidenceGate.load(tmp_path / "missing", {}, 0).accepts(.99)


def rows(n=20):
    return [RoutingRow(id=str(i), query=f"query {i}", group_id=str(i // 2), domain=None,
                       expected={"params": {}}) for i in range(n)]


def test_data_split_is_reproducible_and_group_isolated():
    split = partition(rows(), [])
    assert split == partition(list(reversed(rows())), []) or [
        {r.id for r in s} for s in split] == [{r.id for r in s} for s in partition(list(reversed(rows())), [])]
    groups = [{r.group_id for r in s} for s in split]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert [len(s) for s in split] == [12, 4, 4]


def test_data_overlap_and_duplicate_rejected():
    with pytest.raises(ValueError, match="overlap"):
        partition(rows(), ["QUERY 0"])
    with pytest.raises(ValueError, match="duplicate"):
        partition(rows() + rows()[:1], [])


def test_threshold_selection_requires_statistical_support():
    ids = np.array([str(i) for i in range(200)])
    assert select_threshold(np.ones(20), np.ones(20, dtype=bool), np.ones(20, dtype=bool), ids[:20], .95) is None
    assert wilson_lower(100, 100) >= .95
    score = np.array([.1] * 100 + [.9] * 100)
    correct = np.array([False] * 100 + [True] * 100)
    assert select_threshold(score, correct, np.ones(200, dtype=bool), ids, .95) == .9
    assert select_threshold(score, correct, np.zeros(200, dtype=bool), ids, .95) is None


def test_reliability_counts_groups_not_rows():
    score, correct, eligible = np.ones(100), np.ones(100, dtype=bool), np.ones(100, dtype=bool)
    independent = operating_point(score, correct, eligible, np.array([str(i) for i in range(100)]), .5)
    paired = operating_point(score, correct, eligible, np.array([str(i // 2) for i in range(100)]), .5)
    assert independent["accuracy_lower_95"] >= .95 > paired["accuracy_lower_95"]
    assert paired["accepted_groups"] == 50 and paired["accuracy_lower_95_rows"] == independent["accuracy_lower_95"]


def test_routing_export_rows_use_unit_groups_and_v2_labels():
    raw = {"id": "rt_1", "query": "מאזדה 3 עד 50 אלף", "domain": "רכב",
           "expected": {"category": "רכב", "params": {"טווח_מחיר": {"max": 50000}}},
           "metadata": {"language": ["typo"], "label_seen_before": True}, "family": "rt_1", "unit": "u7",
           "quality": "gold", "spans": {}}
    row = from_export(raw)
    assert row.group_id == "u7" and row.language == "typo" and row.quality == "gold"
    assert row.label_seen_before == "true" and row.expected == {"params": {"מחיר": {"max": 50000}}}
    assert raw["expected"]["params"] == {"טווח_מחיר": {"max": 50000}}  # source row untouched


def test_exported_artifact_loads_without_training_modules():
    code = "import joblib,sys; joblib.load('src/artifacts/classifier.joblib'); assert not any(n == 'train' or n.startswith(('training', 'shared')) for n in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_inference_runs_off_event_loop():
    import threading
    main_thread = threading.get_ident()
    parser = local()
    original = parser.candidate
    def candidate(query):
        assert threading.get_ident() != main_thread
        return original(query)
    parser.candidate = candidate
    asyncio.run(parser.parse("4 חדרים"))


@pytest.mark.parametrize("audit_passes", [True, False])
def test_calibration_command_publishes_only_audited_acceptance(tmp_path, monkeypatch, audit_passes):
    from scripts import calibrate_confidence as command
    from training.shared import dataset
    from src.classifier import FusionClassifier
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(r.model_dump()) for r in rows(1000)), encoding="utf-8")
    artifact_path, report_path = tmp_path / "confidence.joblib", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["calibrate", "--data", str(data), "--output", str(artifact_path),
                                     "--report", str(report_path)])
    monkeypatch.setattr(dataset, "load_rows", lambda split: [])
    monkeypatch.setattr(FusionClassifier, "load", lambda *args: FixedClassifier())
    monkeypatch.setattr(command, "bindings", lambda *args: {"fixture": True})
    calls = []
    def labeled(parser, partition):
        calls.append(len(partition))
        n = len(partition)
        signal = np.array([0] * (n // 2) + [1] * (n - n // 2))
        y = signal.astype(bool)
        if len(calls) == 3 and not audit_passes:
            y = np.zeros(n, dtype=bool)
        return signal.reshape(-1, 1), y, np.ones(n, dtype=bool), np.array([r.id for r in partition]), 0
    monkeypatch.setattr(command, "labeled_features", labeled)
    assert command.main() == (0 if audit_passes else 2)
    artifact = joblib.load(artifact_path)
    report = json.loads(report_path.read_text())
    assert artifact["acceptance_enabled"] == audit_passes
    assert report["acceptance_enabled"] == audit_passes
    assert calls == [600, 200, 200]
    assert artifact["threshold"] is not None
    assert ConfidenceGate(artifact, threshold=0).accepts(.99) == audit_passes
