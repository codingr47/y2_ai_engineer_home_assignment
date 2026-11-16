"""Joint comparison of Exp1, Exp1-slots, Exp2, Exp3 (frozen configs; nothing is fit or tuned here).

    python training_frozen_fusion/comparison/compare.py

val_eval numbers are DEVELOPMENT metrics (val_eval selected C and w). Test numbers come from the frozen
configurations. Exp1's test results had already been inspected before these experiments were designed, and no
choice here is tuned against test.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from shared import paths  # noqa: E402
from shared.features import E5Features  # noqa: E402
from shared.runner import Bundle, base_split, resolver_diagnostics, score  # noqa: E402
from shared.slots import fields_to_slots, slot_vocab  # noqa: E402
from shared.taxonomy_rules import Resolver  # noqa: E402

from train import load_vocab  # noqa: E402

RESULTS = HERE / "results"
N_BOOT = 2000
SEED = 0
MODELS = ("exp1", "exp1-slots", "exp2", "exp3")
LABEL = {"exp1": "Exp1 · E5 · 77 fields", "exp1-slots": "Exp1-slots · E5", "exp2": "Exp2 · TF-IDF",
         "exp3": "Exp3 · E5 ⊕ TF-IDF"}
PAIRS = [("exp1-slots", "exp1", "label space (slots vs 77 fields), E5 features"),
         ("exp2", "exp1-slots", "TF-IDF vs E5"),
         ("exp3", "exp1-slots", "fusion vs E5"),
         ("exp3", "exp2", "fusion vs TF-IDF")]
PROTECTED = [paths.EXP1 / "artifacts" / "models" / "heads.joblib", paths.EXP1 / "artifacts" / "models" / "thresholds.json",
             paths.DATA_DIR / "train.jsonl", paths.DATA_DIR / "val.jsonl", paths.DATA_DIR / "test.jsonl"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def load_bundles() -> dict[str, Bundle]:
    vocab77 = load_vocab()
    heads = joblib.load(paths.EXP1 / "artifacts" / "models" / "heads.joblib")
    resolver = Resolver.from_train(base_split("train").fields, slot_vocab(vocab77))
    out = {"exp1": Bundle("exp1", "e5", "fields", heads.C, None, E5Features(), heads, resolver, vocab77)}
    for name, folder in (("exp1-slots", "experiment_1_slots"), ("exp2", "experiment_2"), ("exp3", "experiment_3")):
        out[name] = joblib.load(paths.TRAINING / folder / "artifacts" / "models" / "bundle.joblib")
    return out


# ------------------------------------------------------------------ statistics
def family_bootstrap(recs_a, recs_b, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap over whole families (a source and its rephrasings are resampled together)."""
    assert [r["id"] for r in recs_a] == [r["id"] for r in recs_b]
    a = np.array([r["slot_exact_match"] for r in recs_a], float)
    b = np.array([r["slot_exact_match"] for r in recs_b], float)
    fam = defaultdict(list)
    for i, r in enumerate(recs_a):
        fam[r["family"] or r["id"]].append(i)
    groups = [np.array(v) for v in fam.values()]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        diffs[k] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": round(float(a.mean() - b.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_le_0": round(float((diffs <= 0).mean()), 4), "families": len(groups), "replicates": n_boot,
            "wins": int(((a == 1) & (b == 0)).sum()), "losses": int(((a == 0) & (b == 1)).sum())}


def slices(recs):
    toks = np.array([r["n_tokens"] for r in recs])
    t1, t2 = np.percentile(toks, [33.3, 66.7])

    def length(r):
        return "short" if r["n_tokens"] <= t1 else ("medium" if r["n_tokens"] <= t2 else "long")

    def op(r):
        ops = set(r["operator"]) - {"none"}
        return "range" if "range" in ops else ("min/max" if ops & {"min", "max"} else ("exact" if ops else "none"))

    keys = {
        "language": lambda r: r["language"],
        "operator": op,
        "length": length,
        "rephrasing": lambda r: "rephrasing" if r["is_rephrasing"] else "source",
        "n_gold_fields": lambda r: str(min(len(r["gold_fields"]), 5)) + ("+" if len(r["gold_fields"]) >= 5 else ""),
        "generator": lambda r: (r["generator"] or "?").split("/")[0],
        "complexity": lambda r: r["complexity"],
        "quality": lambda r: r["quality"],
    }
    out = {}
    for name, f in keys.items():
        g = defaultdict(list)
        for r in recs:
            g[f(r)].append(r["slot_exact_match"])
        out[name] = {k: {"n": len(v), "slot_exact_set": round(float(np.mean(v)), 4)} for k, v in sorted(g.items())}
    out["_length_cutoffs_tokens"] = [float(t1), float(t2)]
    return out


def per_slot_delta(res_a, res_b, k=12):
    fa = {d["field"]: d for d in res_a["per_slot"]}
    fb = {d["field"]: d for d in res_b["per_slot"]}
    rows = [{"slot": s, "f1_a": round(fa[s]["f1"], 3), "f1_b": round(fb[s]["f1"], 3),
             "delta": round(fa[s]["f1"] - fb[s]["f1"], 3), "support": fa[s]["support"]} for s in fa if s in fb]
    rows.sort(key=lambda r: r["delta"])
    return {"improved": rows[::-1][:k], "regressed": [r for r in rows if r["delta"] < 0][:k],
            "n_improved": sum(r["delta"] > 0 for r in rows), "n_regressed": sum(r["delta"] < 0 for r in rows)}


def annotated_cases(recs_by_model):
    path = paths.EXP1 / "experiment_analysis" / "root_cause_annotations.jsonl"
    if not path.exists():
        return {"note": f"{path} missing - run experiment_1/experiment_analysis/annotate_sample.py"}
    cases = [json.loads(line) for line in open(path, encoding="utf-8")]
    by_id = {m: {r["id"]: r for r in recs} for m, recs in recs_by_model.items()}
    table = defaultdict(lambda: {"n": 0, **{m: 0 for m in recs_by_model}})
    for c in cases:
        row = table[c["primary_cause"]]
        row["n"] += 1
        for m in recs_by_model:
            row[m] += bool(by_id[m][c["id"]]["slot_exact_match"])
    return {"slot_exact_correct_by_primary_cause": dict(sorted(table.items(), key=lambda kv: -kv[1]["n"])),
            "note": "100 stratified cases from the Exp1 error review (80 errors + 20 controls); counts are cases whose "
                    "full slot set is correct. Exp1 is scored at slot level, so its twin-only errors count as correct."}


# ------------------------------------------------------------------ latency / size
def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def latency(bundles, texts, n_single=200):
    import torch

    from embed import get_encoder

    t0 = time.perf_counter()
    get_encoder()
    e5_load = time.perf_counter() - t0
    out = {"e5_model_load_seconds": round(e5_load, 2), "torch_threads": torch.get_num_threads(),
           "cpu": platform.processor() or platform.machine(), "n_single": n_single, "models": {}}
    for name, b in bundles.items():
        for t in texts[:5]:  # warm-up
            b.postprocess(b.predict_X(b.featurizer.transform_texts([t]))[2][0])
        parts = defaultdict(list)
        totals = []
        for t in texts[:n_single]:
            s0 = time.perf_counter()
            X = b.featurizer.transform_texts([t])
            s1 = time.perf_counter()
            _, _, pred = b.predict_X(X)
            s2 = time.perf_counter()
            b.postprocess(pred[0])
            s3 = time.perf_counter()
            parts["featurize"].append(s1 - s0)
            parts["classify"].append(s2 - s1)
            parts["postprocess"].append(s3 - s2)
            totals.append(s3 - s0)
        s0 = time.perf_counter()
        for i in range(0, len(texts), 32):
            X = b.featurizer.transform_texts(texts[i:i + 32])
            _, _, pred = b.predict_X(X)
            for row in pred:
                b.postprocess(row)
        batch_per_query = (time.perf_counter() - s0) / len(texts)
        ms = lambda v: round(1000 * float(v), 3)  # noqa: E731
        out["models"][name] = {
            "batch1_ms_p50": ms(np.median(totals)), "batch1_ms_p95": ms(np.percentile(totals, 95)),
            "batch1_ms_p50_by_stage": {k: ms(np.median(v)) for k, v in parts.items()},
            "batch32_ms_per_query": ms(batch_per_query),
        }
    return out


def sizes(bundles):
    hf = Path.home() / ".cache" / "huggingface" / "hub" / "models--intfloat--multilingual-e5-small"
    out = {"e5_weights_on_disk_mb": round(dir_size(hf) / 1e6, 1), "models": {}}
    for name, b in bundles.items():
        heads_b = len(joblib_bytes(b.heads))
        feat_b = len(joblib_bytes(b.featurizer))
        out["models"][name] = {"heads_mb": round(heads_b / 1e6, 2), "featurizer_mb": round(feat_b / 1e6, 2),
                               "needs_e5": b.kind in ("e5", "e5+tfidf"),
                               "n_features": getattr(b.featurizer, "n_features", None)
                               or (384 + b.featurizer.tfidf.n_features if b.kind == "e5+tfidf" else 384)}
    return out


def joblib_bytes(obj) -> bytes:
    import io

    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()


# ------------------------------------------------------------------ main
def main():
    before = {p.name: sha(p) for p in PROTECTED}
    bundles = load_bundles()

    res = {split: {} for split in ("val_eval", "test")}
    recs = {split: {} for split in ("val_eval", "test")}
    for split in res:
        for m in MODELS:
            r = score(bundles[m], split, records=True)
            recs[split][m] = r.pop("records")
            res[split][m] = r
            print(f"{split} {m}: slot_exact={r['slot_exact_set']:.3f} strict={r['strict_exact_set']['deployable']:.3f}",
                  flush=True)

    out = {
        "notes": ["val_eval = development (used to select C and w); test = frozen configs, scored once per experiment",
                  "Exp1 test results were inspected before this comparison was designed; nothing is tuned on test",
                  "slot metrics use the 70-slot space; Exp1 (77 fields) is mapped to slots for comparison",
                  "strict = original 77 taxonomy names: deployable (no operator, default name), provisional "
                  "(span-local detector on GOLD spans - diagnostic), oracle (gold operator - diagnostic)"],
        "headline": {split: {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                 if k not in ("per_slot", "confidence")}
                             for m, r in res[split].items()} for split in res},
        "bootstrap": {split: {f"{a} vs {b}": {"question": q, **family_bootstrap(recs[split][a], recs[split][b])}
                              for a, b, q in PAIRS} for split in res},
        "per_slot_delta": {f"{a} vs {b}": per_slot_delta(res["val_eval"][a], res["val_eval"][b])
                           for a, b, _ in PAIRS[1:]},
        "slices": {split: {m: slices(recs[split][m]) for m in MODELS} for split in res},
        "annotated_cases": annotated_cases(recs["val_eval"]),
        "resolver": {**{split: resolver_diagnostics(bundles["exp2"], split) for split in res},
                     "test_gold": resolver_diagnostics(bundles["exp2"], "test", "gold"),
                     "test_silver": resolver_diagnostics(bundles["exp2"], "test", "silver")},
        "confidence": {split: {m: res[split][m]["confidence"] for m in MODELS} for split in res},
        "learning_curves": {m: json.loads((paths.TRAINING / f / "results" / "learning_curve.json")
                                          .read_text(encoding="utf-8"))
                            for m, f in (("exp1-slots", "experiment_1_slots"), ("exp2", "experiment_2"),
                                         ("exp3", "experiment_3"))},
        "selected": {m: json.loads((paths.TRAINING / f / "results" / "selected.json").read_text(encoding="utf-8"))
                     for m, f in (("exp1-slots", "experiment_1_slots"), ("exp2", "experiment_2"),
                                  ("exp3", "experiment_3"))},
    }
    texts = [m["query"] for m in base_split("val_eval").meta]
    out["latency"] = latency(bundles, texts)
    out["size"] = sizes(bundles)

    after = {p.name: sha(p) for p in PROTECTED}
    assert before == after, "protected artifacts or data changed"
    out["protected_sha256_16"] = after

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "comparison.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                             encoding="utf-8")
    for split in recs:  # per-query records of all models, for the future routing calibration
        with open(RESULTS / f"records_{split}.jsonl", "w", encoding="utf-8") as f:
            for i in range(len(recs[split]["exp1"])):
                row = {"id": recs[split]["exp1"][i]["id"]}
                for m in MODELS:
                    r = recs[split][m][i]
                    row[m] = {k: r[k] for k in ("predicted_slots", "slot_exact_match", "domain_confidence",
                                                "min_selected_probability", "max_rejected_probability",
                                                "prediction_margin", "prediction_margin_ratio")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"latency": out["latency"]["models"], "size": out["size"]}, ensure_ascii=False, indent=1))


def main_v2():
    """Same comparison under label schema v2 (training_frozen_fusion/shared/normalize.py, applied at load time).

    Exp1-slots/2/3 use their v2 bundles (same features and frozen configs, labels normalized). Exp1's 77-field
    predictions are mapped through normalize_fields. An equivalence check compares each v2 bundle with its slot
    bundle scored against the same v2 gold: they may only disagree on rows touching the vehicle year (the one
    non-bijective change: שנות_ייצור -> שנה)."""
    from shared.normalize import RULES_SHA, SCHEMA

    before = {p.name: sha(p) for p in PROTECTED}
    slot_bundles = load_bundles()
    bundles = {"exp1": slot_bundles["exp1"]}
    for name, folder in (("exp1-slots", "experiment_1_slots"), ("exp2", "experiment_2"), ("exp3", "experiment_3")):
        bundles[name] = joblib.load(paths.TRAINING / folder / "artifacts" / "models_v2" / "bundle.joblib")

    res = {s: {} for s in ("val_eval", "test")}
    recs = {s: {} for s in ("val_eval", "test")}
    equivalence = {s: {} for s in ("val_eval", "test")}
    for split in res:
        for m in MODELS:
            r = score(bundles[m], split, records=True, eval_schema="v2")
            recs[split][m] = r.pop("records")
            res[split][m] = r
            print(f"[v2] {split} {m}: field_exact={r['slot_exact_set']:.3f}", flush=True)
        for m in ("exp1-slots", "exp2", "exp3"):
            old = score(slot_bundles[m], split, records=True, eval_schema="v2")["records"]
            diff = [(a["id"], a["gold_fields"], a["predicted_slots"], b["predicted_slots"])
                    for a, b in zip(old, recs[split][m]) if set(a["predicted_slots"]) != set(b["predicted_slots"])]
            year = [d for d in diff if any(f == "רכב/שנה" for f in d[1] + d[2] + d[3])]
            equivalence[split][m] = {
                "rows_with_different_predictions": len(diff), "of_which_touch_vehicle_year": len(year),
                "slot_bundle_exact": round(float(np.mean([r["slot_exact_match"] for r in old])), 4),
                "v2_bundle_exact": round(res[split][m]["slot_exact_set"], 4)}

    out = {
        "schema": SCHEMA, "rules_sha": RULES_SHA,
        "notes": ["labels normalized at load time; data files untouched (hashes checked)",
                  "metric names keep 'slot_' for continuity; in v2 they are field metrics over 69 v2 fields, and "
                  "strict == field exact-set (no point/range resolution left)",
                  "v2 bundles reuse each experiment's frozen selected config (no re-selection); test is a "
                  "frozen-config re-labelled evaluation",
                  "val_eval = development"],
        "headline": {s: {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                             if k not in ("per_slot", "confidence", "strict_exact_set", "strict_micro_f1")}
                         for m, r in res[s].items()} for s in res},
        "bootstrap": {s: {f"{a} vs {b}": {"question": q, **family_bootstrap(recs[s][a], recs[s][b])}
                          for a, b, q in PAIRS} for s in res},
        "equivalence_vs_slot_bundles": equivalence,
        "slices": {s: {m: slices(recs[s][m]) for m in MODELS} for s in res},
        "confidence": {s: {m: res[s][m]["confidence"] for m in MODELS} for s in res},
    }
    after = {p.name: sha(p) for p in PROTECTED}
    assert before == after, "protected artifacts or data changed"
    out["protected_sha256_16"] = after
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "comparison_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                                encoding="utf-8")
    print(json.dumps({"equivalence": equivalence, "bootstrap_test": out["bootstrap"]["test"]}, ensure_ascii=False,
                     indent=1))


if __name__ == "__main__":
    if "--schema" in sys.argv and sys.argv[sys.argv.index("--schema") + 1] == "v2":
        main_v2()
    else:
        main()
