"""Collect every experiment1 number from the run files into results/summary.json (and print the tables).

    python -m training_bert.experiment1.summarize

Nothing is recomputed: each value is read from runs/<run>/metrics.json, final*.json, decode*.json or
e12_gate/gate_report.json, so every figure in README.md can be traced to a file.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = HERE / "results" / "summary.json"


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def best_of(run: str) -> dict:
    """Best val_eval epoch of a training run (as selected during training) and its main metrics."""
    m = jload(RUNS / run / "metrics.json")
    b = m["best"]
    ev = m["history"][b["epoch"] - 1]["val_eval"][b["policy"]]
    last = m["history"][-1]
    return {"run": run, "epoch": b["epoch"], "epochs": len(m["history"]), "policy": b["policy"],
            "train_rows": m["config"]["train_rows"], "lr": m["config"]["lr"], "seed": m["config"]["seed"],
            "encoder": m["config"]["encoder"], "json_exact": ev["json_exact"],
            "field_set_exact": ev["field_set_exact"], "domain_acc": ev["domain_acc"], "span_f1": ev["span_f1"],
            "op_acc": ev["op_acc"], "train_seconds_per_epoch": round(st.mean(h["train_seconds"] for h in m["history"]), 1),
            "per_epoch_json": [round(max(v["json_exact"] for v in h["val_eval"].values()), 4) for h in m["history"]],
            "last_epoch_json": max(v["json_exact"] for v in last["val_eval"].values())}


def mean_sd(xs):
    return {"mean": st.mean(xs), "sd": st.stdev(xs) if len(xs) > 1 else 0.0, "n": len(xs), "values": xs}


def main():
    s = {}
    s["frozen_fusion_val_eval"] = jload(HERE / "results" / "frozen_fusion_val_eval.json")
    s["e2_lr_sweep"] = [best_of(f"e2_lr{lr}_s42") for lr in ("1e-5", "2e-5", "3e-5", "5e-5", "8e-5")]
    seeds = [best_of(r) for r in ("e2_lr5e-5_s42", "e3_lr5e-5_s13", "e3_lr5e-5_s1337")]
    s["e3_seeds"] = {"runs": seeds, "json_exact": mean_sd([r["json_exact"] for r in seeds]),
                     "field_set_exact": mean_sd([r["field_set_exact"] for r in seeds])}
    curve = []
    for n in (2000, 4000, 8000, 12000):
        rs = [best_of(f"e4_n{n}_sub0_s42"), best_of(f"e4_n{n}_sub1_s13")]
        curve.append({"target_rows": n, "rows": [r["train_rows"] for r in rs], "runs": rs,
                      "json_exact": mean_sd([r["json_exact"] for r in rs])})
    curve.append({"target_rows": 17272, "rows": [r["train_rows"] for r in seeds], "runs": seeds,
                  "json_exact": mean_sd([r["json_exact"] for r in seeds])})
    s["e4_learning_curve"] = curve
    silver = [best_of("e4_silver_s42"), best_of("e4_silver_s13")]
    s["e4_silver_only"] = {"runs": silver, "json_exact": mean_sd([r["json_exact"] for r in silver])}

    e6 = {}
    for seed in (42, 13, 1337):
        d = RUNS / f"e6_lr5e-5_s{seed}"
        old, new = jload(d / "final.json"), jload(d / "final_norm2.json")
        dec = {sp: jload(d / f"decode_norm2_{sp}.json") for sp in ("val_eval", "val_calib")}
        e6[seed] = {"old_normalizer": {sp: old[sp]["json_exact"] for sp in ("val_eval", "val_calib")},
                    "new_normalizer": {sp: new[sp]["json_exact"] for sp in ("val_eval", "val_calib")},
                    "fallback": {sp: {k: v["json_exact"] for k, v in dec[sp].items()} for sp in dec},
                    "field_set_new": {sp: new[sp]["field_set_exact"] for sp in ("val_eval", "val_calib")}}
    s["e6_final_config_seeds"] = e6
    s["e6_means"] = {
        name: {sp: mean_sd([pick(e6[x], sp) for x in e6]) for sp in ("val_eval", "val_calib")}
        for name, pick in (("old_normalizer", lambda r, sp: r["old_normalizer"][sp]),
                           ("new_normalizer", lambda r, sp: r["new_normalizer"][sp]),
                           ("new_normalizer+fallback_span", lambda r, sp: r["fallback"][sp]["span@0.5"]),
                           ("new_normalizer+fallback_presence@0.5", lambda r, sp: r["fallback"][sp]["presence@0.5"]))}

    vh = jload(RUNS / "e7_vh_lr5e-5_s42" / "metrics.json")
    s["e7_value_head_s42_epoch8"] = {k: v["json_exact"] for k, v in vh["history"][-1]["val_eval"].items()}
    s["e8_large"] = [best_of("e8_large_lr3e-5_s42"), best_of("e8_large_lr2e-5_s42")]
    ens = RUNS / "e10_ensemble_e6"
    s["e10_ensemble"] = {sp: {k: v["json_exact"] for k, v in jload(ens / f"decode_norm2_{sp}.json").items()}
                         for sp in ("val_eval", "val_calib")}
    fin = jload(RUNS / "e11_final_ensemble" / "final_final.json")
    s["e11_final"] = {sp: {k: fin[sp][k] for k in ("json_exact", "field_set_exact", "domain_acc", "span_f1",
                                                   "span_overlap_r", "op_acc", "waterfall", "slices")}
                      for sp in ("val_eval", "val_calib", "test")}
    s["e12_gate"] = jload(RUNS / "e12_gate" / "gate_report.json")
    s["errors_e6_s42_val_eval"] = {k: v for k, v in jload(RUNS / "e6_lr5e-5_s42" / "errors_norm2_val_eval.json").items()
                                   if k != "examples"}
    OUT.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

    pct = lambda x: f"{100 * x:.1f}%"
    print("LR sweep:", [(r["lr"], pct(r["json_exact"]), r["epoch"]) for r in s["e2_lr_sweep"]])
    print("seeds:", pct(s["e3_seeds"]["json_exact"]["mean"]), "+-", pct(s["e3_seeds"]["json_exact"]["sd"]))
    print("curve:", [(c["rows"], pct(c["json_exact"]["mean"])) for c in curve], "silver", pct(s["e4_silver_only"]["json_exact"]["mean"]))
    for k, v in s["e6_means"].items():
        print("e6", k, {sp: f"{pct(x['mean'])}+-{pct(x['sd'])}" for sp, x in v.items()})
    print("value head:", {k: pct(v) for k, v in s["e7_value_head_s42_epoch8"].items()})
    print("large:", [(r["lr"], pct(r["json_exact"]), r["epoch"], r["train_seconds_per_epoch"]) for r in s["e8_large"]])
    print("ensemble:", {sp: {k: pct(v) for k, v in d.items()} for sp, d in s["e10_ensemble"].items()})
    print("final:", {sp: pct(v["json_exact"]) for sp, v in s["e11_final"].items()})
    print("wrote", OUT.relative_to(HERE.parents[1]))


if __name__ == "__main__":
    main()
