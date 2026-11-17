"""Re-stamp the tagger gate's bindings after a measured, gate-neutral code change; the fitted gate is not touched.

    python -m scripts.rebind_tagger_gate --reason "..." --evidence training/training_bert/experiment4/results/redecode_report.json

The tagger gate hashes the runtime code (src/tagger, src/extraction). Changing that code changes the hash, and the
gate then refuses to load. That's right for a change that alters the gate's inputs. For a change measured to leave
them essentially unchanged (a decoding fix that changes a handful of val rows, or routing that happens after the
gate), refitting would only add noise. This command keeps the model, scaler, features, threshold and every recorded
measurement exactly as they are, recomputes the bindings, and records the reason and the evidence file in
`code_rebinds`.

It refuses when:
- the feature schema changed
- the member weights, artifacts or encoder changed (only `code` and Frozen Fusion's nested bindings may differ)
- the evidence file is missing
"""
import argparse
import json
from pathlib import Path

import joblib

from src.config.settings import Settings
from src.frozen_fusion.confidence import bindings as ff_bindings
from src.tagger.gate import FEATURE_NAMES, bindings

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--evidence", type=Path, required=True, help="measurement showing the gate inputs are unchanged")
    ap.add_argument("--report", type=Path, default=ROOT / "docs/tagger_gate.json")
    a = ap.parse_args()
    if not (ROOT / a.evidence).is_file():
        raise SystemExit(f"evidence file not found: {a.evidence}")
    s = Settings(extractor_enabled=True, local_parser="tagger")
    art = joblib.load(s.tagger_gate_path)
    if tuple(art["features"]) != FEATURE_NAMES:
        raise SystemExit("feature schema changed: refit the gate (scripts/build_tagger_gate.py)")
    now = bindings(s.tagger_artifact_dir, s.tagger_model_dir,
                   ff_bindings(s.classifier_artifact_path, s.e5_model_path, s.extractor_use_tagger),
                   s.confidence_artifact_path)
    old = art["bindings"]
    changed = sorted(k for k in now if now[k] != old.get(k))
    if any(k not in ("code", "frozen_fusion") for k in changed):
        raise SystemExit(f"bindings beyond code changed: {changed}; refit the gate instead")
    if not changed:
        print("bindings already current")
        return
    art = {**art, "bindings": now,
           "code_rebinds": [*art.get("code_rebinds", []),
                            {"reason": a.reason, "evidence": a.evidence.as_posix(), "changed": changed,
                             "old_code": old.get("code"), "new_code": now["code"]}]}
    joblib.dump(art, s.tagger_gate_path, compress=3)
    report = json.loads(a.report.read_text(encoding="utf-8")) if a.report.exists() else {}
    report.update({k: v for k, v in art.items() if k in ("bindings", "code_rebinds")})
    a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"tagger gate re-stamped ({', '.join(changed)}); fitted state unchanged")


if __name__ == "__main__":
    main()
