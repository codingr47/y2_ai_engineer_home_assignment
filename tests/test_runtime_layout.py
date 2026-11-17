"""Compatibility and artifact integrity checks for runtime package relocation."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pytest

from scripts.migrate_runtime_bindings import verify_source
from src.config.settings import ROOT, Settings
from src.frozen_fusion.confidence import ConfidenceGate, bindings
from src.tagger.gate import TaggerGate, _digest


@pytest.mark.parametrize("legacy,target", [
    ("classifier", "frozen_fusion.classifier"), ("local_parser", "frozen_fusion.local_parser"),
    ("confidence", "frozen_fusion.confidence"), ("settings", "config.settings"),
])
def test_training_compatibility_preserves_module_identity(legacy, target):
    assert importlib.import_module(f"src.{legacy}") is importlib.import_module(f"src.{target}")


def test_legacy_local_parser_monkeypatch_changes_canonical_module(monkeypatch):
    import src.local_parser as legacy
    from src.frozen_fusion import local_parser
    sentinel = object()
    monkeypatch.setattr(legacy, "bindings", sentinel)
    assert local_parser.bindings is sentinel


def test_original_asgi_entrypoint_is_preserved():
    from src import app as legacy
    from src.api import app as canonical
    assert legacy.app is canonical.app
    assert legacy.create_app is canonical.create_app


def test_source_migration_accepts_expected_module_relocation():
    verify_source("from src.classifier import FusionClassifier\nVALUE = 1",
                  "from src.frozen_fusion.classifier import FusionClassifier\nVALUE = 1", "example")


@pytest.mark.parametrize("current", [
    "from src.frozen_fusion.classifier import FusionClassifier\nVALUE = 2",
    "from somewhere_else import FusionClassifier\nVALUE = 1",
])
def test_source_audit_rejects_non_migration_changes(current):
    with pytest.raises(ValueError, match="unapproved executable change"):
        verify_source("from src.classifier import FusionClassifier\nVALUE = 1", current, "example")


@pytest.mark.parametrize("relative,gate_type", [
    ("src/artifacts/confidence.joblib", ConfidenceGate),
    ("src/artifacts/tagger/gate.joblib", TaggerGate),
])
def test_migrated_gate_preserves_fitted_state_and_rejects_binding_mismatch(relative, gate_type):
    report = json.loads((ROOT / "docs/runtime-module-migration.json").read_text())
    key = "frozen_fusion" if gate_type is ConfidenceGate else "tagger"
    baseline = report["checks"][key]
    path = ROOT / relative
    after = joblib.load(path)
    state = lambda artifact: {k: v for k, v in artifact.items() if k not in {"bindings", "code_rebinds"}}
    assert joblib.hash(state(after), hash_name="sha1") == baseline["non_binding_state_joblib_sha1"]
    loaded = gate_type.load(path, after["bindings"])
    width = len(after["features"])
    for index, vector in enumerate((np.zeros(width), np.ones(width), np.linspace(-1, 1, width))):
        assert loaded.score(vector) == baseline["baseline_scores"][index]
        assert loaded.accepts(loaded.score(vector)) == baseline["baseline_decisions"][index]
    with pytest.raises(ValueError, match="does not match"):
        gate_type.load(path, {**after["bindings"], "code": "tampered"})


def test_gate_bindings_cover_relocated_sources_and_nested_dependency():
    settings = Settings()
    ff = joblib.load(settings.confidence_artifact_path)
    tagger = joblib.load(settings.tagger_gate_path)
    code = bindings(settings.classifier_artifact_path, Path("absent-encoder"), True)["code"]
    assert ff["bindings"]["code"] == code
    assert tagger["bindings"]["code"] == _digest([
        *sorted((ROOT / "src/tagger").glob("*.py")), *sorted((ROOT / "src/extraction").glob("*.py"))])
    assert tagger["bindings"]["frozen_fusion"] == {
        "bindings": ff["bindings"], "gate": _digest([settings.confidence_artifact_path])}


def test_runtime_imports_do_not_load_training_packages():
    code = """
import sys
import src.api.app
import src.llm.parser
import src.frozen_fusion.local_parser
import src.tagger.parser
assert not any(n.startswith(('training_', 'shared')) or n == 'train' for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


def test_package_initializers_have_no_executable_wiring():
    import ast
    for name in ("api", "contracts", "config", "parsing", "frozen_fusion", "llm", "infrastructure"):
        assert ast.parse((ROOT / "src" / name / "__init__.py").read_text()).body == []


def test_runtime_sources_have_portable_bytes_for_gate_bindings():
    assert not [path for path in (ROOT / "src").rglob("*.py") if b"\r\n" in path.read_bytes()]
