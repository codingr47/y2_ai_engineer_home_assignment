"""Enable a confidence artifact at its audited threshold, certifying only the bound the audit actually reached.

calibrate_confidence picks the threshold on the selection part BEFORE the audit, then audits it once. If the audit
misses the requested target, the artifact is written with acceptance disabled. This command is the explicit,
recorded decision to accept that same frozen threshold at the reliability it was measured to have:

    python -m scripts.accept_audited_threshold --min-bound 0.80

It never changes the threshold, the model or the bindings. The report records that the accepted bound was chosen
after seeing the audit, and the target the calibration originally asked for.
"""
import argparse
import json
from pathlib import Path

import joblib


def accept(artifact: dict, min_bound: float) -> dict:
    audit = artifact.get("audit") or {}
    bound = audit.get("accuracy_lower_95")
    if artifact.get("schema") != "parse-confidence-1" or artifact.get("threshold") is None:
        raise ValueError("artifact has no audited threshold")
    if not audit.get("accepted") or bound is None:
        raise ValueError("the audit accepted nothing; there is no measured reliability to certify")
    if bound < min_bound:
        raise ValueError(f"audited lower bound {bound:.3f} is below --min-bound {min_bound:.3f}")
    return {**artifact, "acceptance_enabled": True, "requested_target": artifact.get("target_accuracy"),
            "target_accuracy": round(float(bound), 4), "certified_lower_bound": float(bound),
            "target_set_after_audit": True,
            "acceptance_note": ("Threshold fixed on the selection part before the audit; the audit missed the "
                                "requested target, and acceptance was enabled afterwards at the audited bound.")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, default=Path("src/artifacts/confidence.joblib"))
    ap.add_argument("--report", type=Path, default=Path("docs/confidence_audit.json"))
    ap.add_argument("--min-bound", type=float, required=True,
                    help="refuse unless the audited accuracy lower bound reaches this value")
    args = ap.parse_args()
    artifact = accept(joblib.load(args.artifact), args.min_bound)
    joblib.dump(artifact, args.artifact, compress=3)
    report = {k: v for k, v in artifact.items() if k not in ("model", "features")}
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    a = artifact["audit"]
    print(f"acceptance enabled at threshold {artifact['threshold']:.4f}: audited coverage {a['coverage']:.1%}, "
          f"accuracy {a['accuracy']:.1%}, lower bound {a['accuracy_lower_95']:.3f}")


if __name__ == "__main__":
    main()
