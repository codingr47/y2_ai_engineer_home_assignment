"""Re-stamp the confidence artifact's `code` binding after a change that provably does not change behavior.

    python -m scripts.rebind_confidence_code --since 75a8e0e

The gate artifact hashes the inference code. A docstring edit changes that hash, so `LocalParser.load` refuses the
artifact, even though nothing the gate measured has changed (commit d79bfa6 only rewrote paths in two docstrings).

This command re-stamps only when, for every hashed source file, the code at `--since` (the revision the gate was
audited on) and the working tree have **identical parse trees once docstrings are removed**. Comments and
formatting are invisible to the parse tree as well. Every other binding (classifier, extractor assets, encoder,
features, use_tagger) must already match. The artifact keeps a record of the re-stamp.
"""
import argparse
import ast
import json
import subprocess
from pathlib import Path

import joblib

from src.frozen_fusion.confidence import bindings
from src.config.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def code_files():
    src = ROOT / "src"
    return [src / "frozen_fusion/classifier.py", src / "frozen_fusion/local_parser.py", src / "frozen_fusion/confidence.py", src / "contracts/models.py",
            *sorted((src / "extraction").glob("*.py"))]


def same_behavior(since: str) -> list[str]:
    """Files whose docstring-free parse tree differs from revision `since` (empty = safe to re-stamp)."""
    changed = []
    for p in code_files():
        rel = p.relative_to(ROOT).as_posix()
        try:
            old = subprocess.run(["git", "show", f"{since}:{rel}"], cwd=ROOT, check=True, capture_output=True).stdout
        except subprocess.CalledProcessError:
            changed.append(rel + " (missing at revision)")
            continue
        if _strip_docstrings(ast.parse(old.decode("utf-8"))) != _strip_docstrings(ast.parse(p.read_text(encoding="utf-8"))):
            changed.append(rel)
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", required=True, help="git revision the artifact was audited on")
    ap.add_argument("--artifact", type=Path, default=ROOT / "src/artifacts/confidence.joblib")
    ap.add_argument("--report", type=Path, default=ROOT / "docs/confidence_audit.json")
    a = ap.parse_args()
    s = Settings(extractor_enabled=True)
    art = joblib.load(a.artifact)
    now = bindings(s.classifier_artifact_path, s.e5_model_path, s.extractor_use_tagger)
    other = {k: v for k, v in now.items() if k != "code"} == {k: v for k, v in art["bindings"].items() if k != "code"}
    if not other:
        raise SystemExit("a non-code binding differs; recalibrate instead")
    if art["bindings"]["code"] == now["code"]:
        print("code binding already current")
        return
    changed = same_behavior(a.since)
    if changed:
        raise SystemExit(f"code changed in behavior since {a.since}: {changed}; recalibrate instead")
    art = {**art, "bindings": {**art["bindings"], "code": now["code"]},
           "code_rebinds": [*art.get("code_rebinds", []),
                            {"since": a.since, "reason": "docstring-only changes (parse trees identical without "
                                                         "docstrings)", "files": [p.relative_to(ROOT).as_posix()
                                                                                  for p in code_files()]}]}
    joblib.dump(art, a.artifact, compress=3)
    report = {k: v for k, v in art.items() if k not in ("model", "features")}
    a.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"code binding re-stamped (behavior identical to {a.since})")


if __name__ == "__main__":
    main()
