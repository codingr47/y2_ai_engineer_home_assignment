"""One-time, audited runtime-package migration; never fits or recalibrates models.

Run with --since b1a05cc15b2ac578d6237fb565f540c1133c0b56. The default is
read-only; --apply updates the two gates and writes the migration audit.
"""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

import joblib
import numpy as np

from scripts.rebind_confidence_code import _strip_docstrings
from src.config.settings import ROOT, Settings
from src.frozen_fusion.confidence import ConfidenceGate, bindings as ff_bindings
from src.tagger.gate import TaggerGate, _digest

BASE_REVISION = "b1a05cc15b2ac578d6237fb565f540c1133c0b56"
MOVES = {
    "app": "api/app", "models": "contracts/models", "http_models": "contracts/http_models",
    "settings": "config/settings", "parser": "parsing/parser", "fallback": "parsing/fallback",
    "classifier": "frozen_fusion/classifier", "local_parser": "frozen_fusion/local_parser",
    "confidence": "frozen_fusion/confidence", "openai_parser": "llm/parser",
    "openai_catalog": "llm/catalog", "openai_prompt": "llm/prompt", "few_shots": "llm/few_shots",
    "injection_guard": "llm/injection_guard", "cache": "infrastructure/cache",
    "metrics": "infrastructure/metrics", "request_usage": "infrastructure/request_usage",
}
RESOURCE_EDITS = {
    "settings": [('Path(__file__).resolve().parents[1]', 'Path(__file__).resolve().parents[2]')],
    "openai_catalog": [('Path(__file__).parent / "extraction/artifacts/taxonomy_v2.json"',
                        'Path(__file__).resolve().parents[1] / "extraction/artifacts/taxonomy_v2.json"')],
    "confidence": [('root = Path(__file__).resolve().parent', 'root = Path(__file__).resolve().parents[1]'),
                   ('root / "classifier.py", root / "local_parser.py", root / "confidence.py", root / "models.py"',
                    'root / "frozen_fusion/classifier.py", root / "frozen_fusion/local_parser.py", root / "frozen_fusion/confidence.py", root / "contracts/models.py"')],
}


def git_blob(revision, path):
    return subprocess.run(["git", "show", f"{revision}:{path}"], cwd=ROOT,
                          check=True, capture_output=True).stdout


def expected_source(source, name):
    """Apply only the explicit module rename and resource-location allowlist."""
    for old, new in MOVES.items():
        source = re.sub(r"\bsrc\." + old + r"\b", "src." + new.replace("/", "."), source)
    source = source.replace("from src import request_usage", "from src.infrastructure import request_usage")
    source = source.replace("from src import confidence", "from src.frozen_fusion import confidence")
    for old, new in RESOURCE_EDITS.get(name, ()):
        if old not in source:
            raise ValueError(f"missing expected resource edit: {name}")
        source = source.replace(old, new)
    return source


def verify_source(source, current, name):
    expected = expected_source(source, name)
    if _strip_docstrings(ast.parse(expected)) != _strip_docstrings(ast.parse(current)):
        raise ValueError(f"unapproved executable change: {name}")


def audit_sources(since=BASE_REVISION):
    """Reject executable changes beyond the exact rename/path allowlist."""
    if subprocess.check_output(["git", "rev-parse", since], cwd=ROOT, text=True).strip() != BASE_REVISION:
        raise ValueError("migration must use its reviewed base revision")
    paths = {f"src/{name}.py": f"src/{new}.py" for name, new in MOVES.items()}
    for package in ("extraction", "tagger"):
        old_paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", since,
                                             f"src/{package}"], cwd=ROOT, text=True).splitlines()
        for path in old_paths:
            if path.endswith(".py"):
                paths[path] = path
        if {p.relative_to(ROOT).as_posix() for p in (ROOT / "src" / package).glob("*.py")} != {
                p for p in old_paths if p.endswith(".py")}:
            raise ValueError(f"unexpected source inventory: {package}")
    evidence = []
    for old, new in paths.items():
        source = git_blob(since, old).decode("utf-8")
        current = (ROOT / new).read_text(encoding="utf-8")
        verify_source(source, current, Path(old).stem)
        evidence.append({"old": old, "new": new,
                         "old_git_sha256": hashlib.sha256(source.encode()).hexdigest(),
                         "new_sha256": hashlib.sha256((ROOT / new).read_bytes()).hexdigest()})
    assets = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", since,
                                      "src/extraction/artifacts", "src/artifacts"], cwd=ROOT, text=True).splitlines()
    gate_paths = {"src/artifacts/confidence.joblib", "src/artifacts/tagger/gate.joblib"}
    for path in [*assets, "src/llm/few_shot_examples.json"]:
        if path in gate_paths:
            continue
        old = "src/few_shot_examples.json" if path.endswith("few_shot_examples.json") else path
        baseline, current = git_blob(since, old), (ROOT / path).read_bytes()
        if path.endswith(".json"):
            baseline, current = baseline.replace(b"\r\n", b"\n"), current.replace(b"\r\n", b"\n")
        if baseline != current:
            raise ValueError(f"asset changed: {path}")
    changed_training = subprocess.check_output(["git", "diff", since, "--", "training_bert",
                                               "training_frozen_fusion", "training"], cwd=ROOT)
    if changed_training:
        raise ValueError("training changes are outside this migration")
    return evidence


def baseline_code_digest(since, paths, newline=b"\n"):
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(Path(path).name.encode())
        h.update(git_blob(since, path).replace(b"\r\n", b"\n").replace(b"\n", newline))
    return h.hexdigest()


def model_checks(before, after, gate_type):
    """Compare full non-binding state, fitted models, scores and decisions."""
    state = lambda a: {k: v for k, v in a.items() if k not in {"bindings", "code_rebinds"}}
    if joblib.hash(state(before)) != joblib.hash(state(after)):
        raise ValueError("non-binding gate state changed")
    if joblib.hash(before.get("code_rebinds", [])) != joblib.hash(after.get("code_rebinds", [])[:-1]):
        raise ValueError("existing audit history changed")
    width = len(before["features"])
    vectors = [np.zeros(width), np.ones(width), np.linspace(-1, 1, width)]
    old, new = gate_type(before), gate_type(after)
    baseline_scores = []
    baseline_decisions = []
    for vector in vectors:
        a, b = old.score(vector), new.score(vector)
        if a != b or old.accepts(a) != new.accepts(b):
            raise ValueError("gate score or decision changed")
        baseline_scores.append(a)
        baseline_decisions.append(old.accepts(a))
    return {"non_binding_state_joblib_sha1": joblib.hash(state(before), hash_name="sha1"),
            "feature_vectors_checked": len(vectors), "baseline_scores": baseline_scores,
            "baseline_decisions": baseline_decisions, "scores_and_decisions_identical": True}


def migrate(*, since=BASE_REVISION, apply=False):
    evidence = audit_sources(since)
    settings = Settings()
    ff_path, tag_path = settings.confidence_artifact_path, settings.tagger_gate_path
    old_ff = joblib.load(io.BytesIO(git_blob(since, ff_path.relative_to(ROOT).as_posix())))
    old_tag = joblib.load(io.BytesIO(git_blob(since, tag_path.relative_to(ROOT).as_posix())))
    ff_sources = ["src/classifier.py", "src/local_parser.py", "src/confidence.py", "src/models.py",
                  *[row["old"] for row in evidence if row["old"].startswith("src/extraction/")]]
    tag_sources = [row["old"] for row in evidence if row["old"].startswith(("src/tagger/", "src/extraction/"))]
    if old_ff["bindings"]["code"] not in {baseline_code_digest(since, ff_sources, newline)
                                          for newline in (b"\n", b"\r\n")}:
        raise ValueError("base Frozen Fusion code does not match its audited artifact")
    if old_tag["bindings"]["code"] != baseline_code_digest(since, tag_sources):
        raise ValueError("base tagger code does not match its audited artifact")
    old_gate_bytes = git_blob(since, ff_path.relative_to(ROOT).as_posix())
    if old_tag["bindings"]["frozen_fusion"] != {"bindings": old_ff["bindings"],
            "gate": hashlib.sha256(ff_path.name.encode() + old_gate_bytes).hexdigest()}:
        raise ValueError("base tagger dependency binding does not match the Frozen Fusion gate")
    # Compute only source bindings. Missing encoders must not be replaced with freshly computed empty hashes.
    # Every model/encoder/feature binding is retained from the reviewed artifacts.
    code = ff_bindings(settings.classifier_artifact_path, Path("missing-encoder-for-code-only"),
                       settings.extractor_use_tagger)["code"]
    tag_code = _digest([*sorted((ROOT / "src/tagger").glob("*.py")),
                        *sorted((ROOT / "src/extraction").glob("*.py"))])
    record = {"since": since, "reason": "runtime module relocation; explicit import/resource-path AST audit",
              "old_code": old_ff["bindings"]["code"], "new_code": code}
    ff = {**old_ff, "bindings": {**old_ff["bindings"], "code": code},
          "code_rebinds": [*old_ff.get("code_rebinds", []), record]}
    buffer = io.BytesIO()
    joblib.dump(ff, buffer, compress=3)
    ff_bytes = buffer.getvalue()
    digest = hashlib.sha256(ff_path.name.encode() + ff_bytes).hexdigest()
    tag = {**old_tag, "bindings": {**old_tag["bindings"], "code": tag_code,
           "frozen_fusion": {"bindings": ff["bindings"], "gate": digest}},
           "code_rebinds": [*old_tag.get("code_rebinds", []),
                            {**record, "old_code": old_tag["bindings"]["code"], "new_code": tag_code}]}
    checks = {"frozen_fusion": model_checks(old_ff, ff, ConfidenceGate),
              "tagger": model_checks(old_tag, tag, TaggerGate)}
    report = {"base_revision": since, "source_audit": evidence, "checks": checks,
              "old_bindings": {"frozen_fusion": old_ff["bindings"], "tagger": old_tag["bindings"]},
              "new_bindings": {"frozen_fusion": ff["bindings"], "tagger": tag["bindings"]},
              "limitations": ["Full encoder/model loading was not tested: provisioned weights are absent.",
                              "Encoder bindings are preserved from the base artifacts, not recomputed."]}
    if apply:
        # Refuse to overwrite unrelated artifact edits. Re-running this migration is allowed only if its
        # existing gates have the exact expected content, excluding serialization differences.
        for path, desired in ((ff_path, ff), (tag_path, tag)):
            current = joblib.load(path)
            original = old_ff if path == ff_path else old_tag
            if joblib.hash(current) not in {joblib.hash(original), joblib.hash(desired)}:
                raise ValueError(f"unexpected existing gate changes: {path.name}")
        ff_path.write_bytes(ff_bytes)
        joblib.dump(tag, tag_path, compress=3)
        for filename, artifact, excluded in (("confidence_audit.json", ff, {"model", "features"}),
                                             ("tagger_gate.json", tag, {"model", "mu", "sd"})):
            (ROOT / "docs" / filename).write_text(json.dumps({k: v for k, v in artifact.items() if k not in excluded},
                                                            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (ROOT / "docs/runtime-module-migration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default=BASE_REVISION)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = migrate(since=args.since, apply=args.apply)
    print(f"Audited {len(report['source_audit'])} runtime files; fitted state, scores and decisions unchanged.")
    print("Gate metadata updated." if args.apply else "Read-only audit passed; use --apply to write metadata.")


if __name__ == "__main__":
    main()
