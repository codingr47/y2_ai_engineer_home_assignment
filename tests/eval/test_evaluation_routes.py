"""Evaluation reports say per request whether the local parser or the LLM fallback answered."""
import pytest

from scripts.evaluate_docker import mount_arguments, runtime_configuration
from scripts.evaluate_parser import route_of, summarize
from scripts.evaluation_html import render_html
from src.settings import Settings


@pytest.mark.parametrize("called,error,route", [
    (False, None, "local"),
    (True, None, "llm_fallback"),
    (True, "http_502", "llm_failed"),
    (False, "http_503", "rejected_no_fallback"),
    (None, None, "unknown"),
])
def test_route_of(called, error, route):
    assert route_of({"provider_called": called, "error": error}) == route


def record(i, route, exact, called):
    return {"id": str(i), "query": "q", "gold": {"category": "רכב", "params": {}}, "pred": None, "route": route,
            "provider_called": called, "category_correct": exact, "field_set_exact": exact, "json_exact": exact,
            "error": None, "latency_seconds": .1 if route == "local" else 2.0, "usage": {}, "language": "standard"}


def test_summary_reports_routes_and_per_route_accuracy():
    recs = [record(0, "local", True, False), record(1, "local", True, False), record(2, "local", False, False),
            record(3, "llm_fallback", True, True)]
    s = summarize(recs)
    assert s["routes"]["counts"] == {"local": 3, "llm_fallback": 1}
    assert s["routes"]["llm_fallback_rate"] == .25
    assert s["slices"]["route"]["local"]["json_exact"] == pytest.approx(2 / 3)
    assert s["slices"]["route"]["llm_fallback"]["latency_p50_seconds"] == 2.0
    assert "X-OpenAI-Usage" in s["routes"]["source"]


def test_summary_flags_missing_route_information():
    s = summarize([record(0, "unknown", True, None)])
    assert s["routes"]["llm_fallback_rate"] == 0 and "unavailable" in s["routes"]["source"]


def test_html_has_route_column_and_filter():
    recs = [record(0, "local", True, False), record(1, "llm_fallback", True, True)]
    s = summarize(recs)
    s["configuration"] = {"mode": "api", "split": "test", "model": "m", "reasoning_effort": "low"}
    html = render_html(s, recs)
    assert 'id="route"' in html and 'data-route="llm_fallback"' in html and "LLM fallback" in html


def test_docker_evaluation_runs_the_tagger(tmp_path):
    (tmp_path / "encoder").mkdir()
    (tmp_path / "encoder" / "config.json").write_text("{}")
    (tmp_path / "member_s42.pt").write_bytes(b"x")
    e5 = tmp_path / "e5"
    e5.mkdir()
    s = Settings(local_parser="tagger", tagger_model_dir=tmp_path, e5_model_path=e5, tagger_hybrid_vehicle_type=True)
    runtime = runtime_configuration(s, "local")
    assert runtime["LOCAL_PARSER"] == "tagger" and runtime["TAGGER_HYBRID_VEHICLE_TYPE"] == "true"
    mounts, paths = mount_arguments(s, "local")
    assert paths["TAGGER_MODEL_DIR"] == "/models/tagger"
    assert any("target=/models/tagger,readonly" in m for m in mounts)
    with pytest.raises(ValueError, match="provisioned weights"):
        mount_arguments(Settings(local_parser="tagger", tagger_model_dir=tmp_path / "missing", e5_model_path=e5), "local")
