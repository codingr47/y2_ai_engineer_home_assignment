"""The fine-tuned tagger as the API's local parser: wiring, gate artifact safeguards, runtime/training separation,
and (when the weights are provisioned) end-to-end parsing."""
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.app import create_app
from src.frozen_fusion.local_parser import LocalParser
from src.config.settings import Settings
from src.tagger.gate import FEATURE_NAMES, SCHEMA, TaggerGate, features
from src.tagger.parser import TaggerParser

ROOT = Path(__file__).resolve().parents[2]
PROVISIONED = (Settings().tagger_model_dir / "member_s42.pt").exists() and Settings().e5_model_path.is_dir()


def test_local_parser_setting_is_validated(monkeypatch):
    assert Settings().local_parser == "frozen_fusion"
    assert Settings(local_parser="tagger").local_parser == "tagger"
    with pytest.raises(ValidationError):
        Settings(local_parser="gpt")
    monkeypatch.setenv("TAGGER_HYBRID_VEHICLE_TYPE", "maybe")
    with pytest.raises(ValidationError):
        Settings.from_env()


def test_tagger_setting_selects_the_tagger_parser(monkeypatch):
    calls = []
    monkeypatch.setattr(LocalParser, "load", lambda *a: (_ for _ in ()).throw(AssertionError("frozen fusion")))
    monkeypatch.setattr(TaggerParser, "load", lambda settings, fallback, metrics: calls.append(1) or object())
    with TestClient(create_app(settings=Settings(extractor_enabled=True, local_parser="tagger"))):
        pass
    assert calls == [1]


class FixedModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p] for _ in X])


def artifact(p=.9, threshold=.8, enabled=True, bindings=None):
    n = len(FEATURE_NAMES)
    return {"schema": SCHEMA, "features": list(FEATURE_NAMES), "model": FixedModel(p), "mu": np.zeros(n),
            "sd": np.ones(n), "threshold": threshold, "acceptance_enabled": enabled, "bindings": bindings or {"b": 1}}


def test_gate_artifact_bindings_and_threshold(tmp_path):
    path = tmp_path / "gate.joblib"
    joblib.dump(artifact(), path)
    gate = TaggerGate.load(path, {"b": 1})
    assert gate.enabled and gate.accepts(gate.score(np.zeros(len(FEATURE_NAMES))))
    with pytest.raises(ValueError, match="does not match"):
        TaggerGate.load(path, {"b": 2})
    joblib.dump({**artifact(), "features": ["other"]}, path)
    with pytest.raises(ValueError, match="feature schema"):
        TaggerGate.load(path, {"b": 1})
    joblib.dump(artifact(p=.5), path)
    assert not TaggerGate.load(path, {"b": 1}).accepts(.5)
    assert TaggerGate.load(path, {"b": 1}, threshold=.4).accepts(.5)          # explicit override
    joblib.dump(artifact(enabled=False), path)
    assert not TaggerGate.load(path, {"b": 1}).accepts(.99)                  # disabled: always route
    assert not TaggerGate.load(tmp_path / "missing.joblib", {"b": 1}).enabled


def test_gate_features_flag_agreement_with_frozen_fusion():
    raw = {"domain": "נדל״ן", "present": {"נדל״ן/מס׳_חדרים": .99}, "spans": [(0, 1, "נדל״ן/מס׳_חדרים", "exact", None)],
           "implied": {}, "conf": {"domain": .99, "token_min": .9, "op_min": .99, "agree": 3, "members": 3}}
    rec = {"raw": raw, "pred": {"מס׳_חדרים": 4}, "pred_domain": "נדל״ן", "query": "4 חדרים"}
    same = features(rec, {"pred": {"category": "נדל״ן", "params": {"מס׳_חדרים": 4}}, "score": .9})
    other = features(rec, {"pred": {"category": "נדל״ן", "params": {"מס׳_חדרים": 5}}, "score": .9})
    i = FEATURE_NAMES.index("ff_same_json")
    assert len(same) == len(FEATURE_NAMES) and same[i] == 1 and other[i] == 0
    assert other[FEATURE_NAMES.index("ff_same_fields")] == 1


def test_runtime_package_does_not_import_training_code():
    code = ("import sys; import src.tagger.parser, src.tagger.gate, src.tagger.normalize, src.tagger.clean; "
            "assert not any(n.startswith(('training', 'shared')) for n in sys.modules), "
            "[n for n in sys.modules if n.startswith(('training', 'shared'))]")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, cwd=ROOT)


def test_committed_artifacts_are_complete():
    d = Settings().tagger_artifact_dir
    for f in ("vocab.json", "members.json", "lexicon_v2.json", "numeric_words.json", "common_words.json",
              "clean_vocab.json", "taxonomy_v2.json", "gate.joblib"):
        assert (d / f).exists(), f


@pytest.fixture(scope="module")
def tagger():
    if not PROVISIONED:
        pytest.skip("tagger weights not provisioned (python -m scripts.provision_tagger)")
    return TaggerParser.load(Settings(extractor_enabled=True, local_parser="tagger"))


@pytest.mark.parametrize("query,category,params", [
    ("דירת 4 חדרים בחיפה עד 2 מיליון", "נדל״ן", {"מס׳_חדרים": 4, "עיר": "חיפה", "מחיר": {"max": 2000000}}),
    ("מאזdה 3 משנת 2019", "רכב", {"יצרנים": "מאזדה"}),                # runtime cleanup fixes «מאזdה»
])
def test_end_to_end_candidate(tagger, query, category, params):
    c = tagger.candidate(query)
    assert c.payload is not None and c.payload["category"] == category
    for k, v in params.items():
        assert c.payload["params"][k] == v
    assert c.vector.shape == (len(FEATURE_NAMES),)


def test_non_search_routes_to_fallback(tagger):
    assert tagger.candidate("איך משיגים תג נכות לרכב").reason == "unsupported_category"


def test_api_serves_tagger_parses(tagger):
    with TestClient(create_app(settings=Settings(extractor_enabled=True, local_parser="tagger"), parser=tagger)) as cl:
        r = cl.post("/parse", json={"q": "דירת 4 חדרים בחיפה עד 2 מיליון"})
        assert r.status_code in (200, 503)   # 503 = the gate routed it and no fallback is configured
        if r.status_code == 200:
            assert r.json()["category"] == "נדל״ן" and r.json()["confidence"] >= tagger.gate.threshold


def test_hybrid_rule_is_applied_only_when_enabled(tagger):
    q = "טויוטה קורולה היברידית"
    off = tagger.candidate(q).payload["params"]
    tagger.hybrid = True
    try:
        on = tagger.candidate(q).payload["params"]
    finally:
        tagger.hybrid = False
    assert "סוגי_רכב" not in off and on.get("סוגי_רכב") == "רכב היברידי/חשמלי"


def test_gate_digest_ignores_line_endings(tmp_path):
    from src.tagger.gate import _digest
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "x.py").write_bytes(b"x = 1\ny = 2\n")
    (tmp_path / "b" / "x.py").write_bytes(b"x = 1\r\ny = 2\r\n")     # what autocrlf produces on Windows
    assert _digest([tmp_path / "a" / "x.py"]) == _digest([tmp_path / "b" / "x.py"])
    (tmp_path / "b" / "x.py").write_bytes(b"x = 1\r\ny = 3\r\n")
    assert _digest([tmp_path / "a" / "x.py"]) != _digest([tmp_path / "b" / "x.py"])


def test_combined_mode_escalates_tagger_rejections_to_openai(monkeypatch):
    """PARSER_MODE=combined with LOCAL_PARSER=tagger: queries the tagger gate rejects go to the OpenAI fallback."""
    import src.api.app as app_module
    from src.parsing.fallback import FallbackRouter
    from src.parsing.parser import FallbackRequired

    class Rejecting:
        async def parse(self, query):
            raise FallbackRequired("low_confidence", .1, .8)

    class FakeOpenAI:
        def __init__(self, client, settings, metrics):
            pass

        async def parse(self, query):
            return {"category": "נדל״ן", "params": {"מס׳_חדרים": 4}, "confidence": 1.0, "notes": []}

    class Client:
        async def close(self):
            pass

    monkeypatch.setattr(TaggerParser, "load", lambda settings, fallback, metrics: Rejecting())
    monkeypatch.setattr(LocalParser, "load", lambda *a: (_ for _ in ()).throw(AssertionError("frozen fusion")))
    monkeypatch.setattr(app_module, "create_client", lambda settings: Client())
    monkeypatch.setattr(app_module, "OpenAIParser", FakeOpenAI)
    settings = Settings(extractor_enabled=True, local_parser="tagger", openai_fallback_enabled=True,
                        openai_model="test-model", openai_api_key="test-key")
    assert settings.active_parser_mode == "combined"
    app = create_app(settings=settings)
    with TestClient(app) as client:
        from src.parsing.empty_params import EmptyParamsParser
        # the tagger, behind the empty-params handling (src/parsing/empty_params.py) and the catalog check
        local = app.state.parser.local
        assert isinstance(app.state.parser, FallbackRouter) and isinstance(local.local, EmptyParamsParser)
        assert isinstance(local.local.local, Rejecting)
        r = client.post("/parse", json={"q": "דירת 4 חדרים"})
        assert r.status_code == 200 and r.json()["params"] == {"מס׳_חדרים": 4}


def test_confident_non_search_is_answered_locally(tagger):
    """A query the tagger reads as not-a-search, with the gate accepting it, gets category null locally (no LLM)."""
    import asyncio
    from src.parsing.parser import FallbackRequired

    q = "איך משיגים תג נכות לרכב"
    c = tagger.candidate(q)
    assert c.reason == "unsupported_category" and c.payload == {"category": None, "params": {}}
    score = tagger.gate.score(c.vector)
    if tagger.gate.accepts(score):
        r = asyncio.run(tagger.parse(q))
        assert r.category is None and r.params == {} and r.confidence == score
    else:
        with pytest.raises(FallbackRequired):
            asyncio.run(tagger.parse(q))


def test_search_parse_reuses_the_computed_candidate(tagger, monkeypatch):
    """parse() runs inference once: LocalParser.parse receives the candidate computed by TaggerParser.parse."""
    import asyncio
    calls = []
    original = tagger._candidate
    monkeypatch.setattr(tagger, "_candidate", lambda q: calls.append(q) or original(q))
    try:
        asyncio.run(tagger.parse("דירת 4 חדרים בחיפה עד 2 מיליון"))
    except Exception:
        pass   # routing outcome is irrelevant here (no fallback configured)
    assert len(calls) == 1 and not tagger._computed
