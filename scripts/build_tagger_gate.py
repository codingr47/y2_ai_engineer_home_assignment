"""Assemble src/artifacts/tagger: the tagger parser's versioned artifacts and its confidence gate.

    python -m scripts.build_tagger_gate            # after: python -m scripts.provision_tagger

1. Copies the decoding artifacts the experiments used (training_bert/artifacts: lexicon, numeric words, common
   words, cleaner vocabulary, taxonomy) and the ensemble's vocab.json, and writes members.json: the E6 checkpoints
   (seeds 42, 13, 1337) with their SHA-256 and the decoding settings.
2. Fits the gate exactly as experiment 2 did (training_bert/experiment2/gate_ff.py, cleaned queries, tagger +
   Frozen Fusion features). Logistic regression on val_eval; threshold = the largest coverage on val_calib half A
   whose one-sided 95% Wilson lower bound reaches 0.96 (target 0.95 + 0.01). It reproduces the experiment's
   threshold, asserted. The audit (val_calib half B), test and routing-audit numbers are recorded in the artifact.
3. Binds the artifact to the members, the artifacts, the encoder, the runtime code and Frozen Fusion's artifacts.

Acceptance is enabled with the measured numbers recorded. The 0.95 target is NOT certified: the audit lower bound is
0.943, and test measured 92.9% at 84.5% coverage. Set acceptance_enabled False in the artifact (or raise
PARSE_CONFIDENCE_THRESHOLD) to be stricter.
"""
import json
import random
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.confidence import bindings as ff_bindings
from src.settings import Settings
from src.tagger.gate import FEATURE_NAMES, SCHEMA, bindings
from src.tagger.parser import sha256

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "training_bert/experiment1/runs"
E2 = ROOT / "training_bert/experiment2/results"
SEEDS = (42, 13, 1337)
TARGET = 0.95


def copy_artifacts(dst: Path):
    from training_bert.common.clean import QueryCleaner
    from training_bert.common.normalize import SpanNormalizer
    SpanNormalizer.load()                   # builds missing training artifacts (lexicon, numeric/common words)
    QueryCleaner.load()                     # builds clean_vocab.json if missing
    src = ROOT / "training_bert/artifacts"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("lexicon_v2.json", "numeric_words.json", "common_words.json", "clean_vocab.json", "taxonomy_v2.json"):
        shutil.copyfile(src / f, dst / f)
    shutil.copyfile(RUNS / f"e6_lr5e-5_s{SEEDS[0]}" / "vocab.json", dst / "vocab.json")
    vocabs = {(RUNS / f"e6_lr5e-5_s{s}" / "vocab.json").read_text(encoding="utf-8") for s in SEEDS}
    assert len(vocabs) == 1, "ensemble members must share one vocabulary"
    spec = {"encoder": "dicta-il/dictabert", "max_len": 128,
            "decode": {"policy": "spans", "presence_threshold": 0.5, "fallback": "presence"},
            "members": [{"run": f"e6_lr5e-5_s{s}", "file": f"member_s{s}.pt",
                         "sha256": sha256(RUNS / f"e6_lr5e-5_s{s}" / "model.pt")} for s in SEEDS],
            "source": "training_bert/experiment1 (E6: lr 5e-5, 8 epochs, batch 32, bf16 autocast)"}
    (dst / "members.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")


def fit_gate():
    from training_bert.common import data as D
    from training_bert.experiment2 import gate_ff
    gate_ff.CLEAN = True
    _, bf, ef, yf = gate_ff.load("val_eval")
    calib, bc, ec, yc = gate_ff.load("val_calib")
    Xf, Xc = np.hstack([bf, ef]), np.hstack([bc, ec])
    fam = {r["id"]: r["family"] for r in D.load("val_calib")}
    fams = sorted({fam[r["id"]] for r in calib})
    random.Random(42).shuffle(fams)
    ia = np.array([fam[r["id"]] in set(fams[: len(fams) // 2]) for r in calib])
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    model = LogisticRegression(C=1.0, max_iter=4000).fit((Xf - mu) / sd, yf)
    score = lambda X: model.predict_proba((X - mu) / sd)[:, 1]
    op = gate_ff.select(score(Xc[ia]), yc[ia], TARGET + 0.01)
    audit = gate_ff.operating_point(score(Xc[~ia]), yc[~ia], op["threshold"])
    return model, mu, sd, op, audit


def main():
    s = Settings(extractor_enabled=True, local_parser="tagger")
    copy_artifacts(s.tagger_artifact_dir)
    model, mu, sd, op, audit = fit_gate()
    report = json.loads((E2 / "gate_ff_report_clean_with_test.json").read_text(encoding="utf-8"))
    e2 = report["gates"]["tagger+ff"][f"{TARGET:.2f}"]
    assert abs(op["threshold"] - e2["select"]["threshold"]) < 1e-9, "threshold differs from experiment 2"
    routing = json.loads((ROOT / "training_bert/experiment3/results/calibration_report.json").read_text(encoding="utf-8"))
    bound = bindings(s.tagger_artifact_dir, s.tagger_model_dir,
                     ff_bindings(s.classifier_artifact_path, s.e5_model_path, s.extractor_use_tagger),
                     s.confidence_artifact_path)
    artifact = {"schema": SCHEMA, "features": list(FEATURE_NAMES), "model": model, "mu": mu, "sd": sd,
                "threshold": float(op["threshold"]), "target_accuracy": TARGET, "acceptance_enabled": True,
                "certified": audit["accuracy_lower_95"] >= TARGET,
                "selection": op, "audit": audit, "test": e2["test"],
                "routing_audit": routing["expt2_val_gate_on_routing_audit"],
                "fit": "val_eval (cleaned queries)", "threshold_from": "val_calib half A (families, seed 42)",
                "audit_on": "val_calib half B",
                "note": ("Acceptance enabled at the experiment-2 threshold. The 0.95 target is not certified: "
                         "audit lower bound 0.943; test 92.9% at 84.5% coverage; routing audit 95.5% at 84.3%."),
                "bindings": bound}
    joblib.dump(artifact, s.tagger_gate_path, compress=3)
    rep = {k: v for k, v in artifact.items() if k not in ("model", "mu", "sd")}
    (ROOT / "docs/tagger_gate.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=float),
                                                encoding="utf-8")
    print(f"gate: threshold {op['threshold']:.4f} | audit cov {audit['coverage']:.3f} acc {audit['accuracy']:.3f} "
          f"lb {audit['accuracy_lower_95']:.3f} | test cov {e2['test']['coverage']:.3f} acc {e2['test']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
