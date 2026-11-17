"""Inference, decoding to {category, params}, and metrics.

Decoding policy: params = fields of predicted spans (+ implied values for present span-less fields).
("filtered", keeping a span only if the presence head agrees, lost in every E2 epoch and was dropped.)
Value policies for categorical spans (lexicon / head / hybrid / confident, see SpanNormalizer.enum_value) are all
scored every epoch on val_eval; one is selected there.
Both run the same SpanNormalizer, then the taxonomy rules.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import torch

from src.extraction.types import values_equal

from src.tagger.predict import predict as _predict, to_json  # noqa: F401 (to_json re-exported)

from .data import explicit_spans, params
from .normalize import json_equal

POLICIES = ("spans",)   # "filtered" (span kept only if the presence head agrees) lost in every E2 epoch; dropped
VALUE_POLICIES = ("lexicon", "head", "hybrid", "confident")


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    tags = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        att[i, :k] = torch.tensor(b["attention_mask"])
        tags[i, :k] = torch.tensor(b["tags"])
    return {"input_ids": ids, "attention_mask": att, "tags": tags,
            "domain": torch.tensor([b["domain"] for b in batch]),
            "present": torch.tensor([b["present"] for b in batch]),
            "spans": [(i, s, e) for i, b in enumerate(batch) for s, e, _, _ in b["spans"]],
            "ops": torch.tensor([op for b in batch for _, _, op, _ in b["spans"]], dtype=torch.long),
            "values": torch.tensor([v for b in batch for _, _, _, v in b["spans"]], dtype=torch.long),
            "implied": [b["implied"] for b in batch]}


def predict(model, tok, rows: list[dict], device, batch_size: int = 64, amp: bool = False) -> list[dict]:
    """Raw per-row predictions (see src.tagger.predict.predict); rows carry "id" and "query"."""
    return _predict(model, tok, [r["query"] for r in rows], device, batch_size, amp, ids=[r["id"] for r in rows])


def _language(r):
    return "+".join((r.get("metadata") or {}).get("language") or ["unknown"])


def score(rows: list[dict], preds: list[dict], norm, policy: str = "spans", threshold: float = 0.5,
          records: bool = False, value_policy: str = "lexicon", fallback: str = "none") -> dict:
    """All metrics for one decoding policy. Span metrics count explicit gold spans only."""
    by_id = {p["id"]: p for p in preds}
    agg = Counter()
    slices = defaultdict(Counter)
    recs = []
    for r in rows:
        p = by_id[r["id"]]
        gd = r["domain"] or "none"
        gold = r["expected"] if r["domain"] else None
        pj = to_json(p, r["query"], norm, policy, threshold, value_policy, fallback)
        exact = json_equal(pj, gold)
        gfields = set(params(r)) if r["domain"] else set()
        pfields = set((pj or {}).get("params") or {})
        dom_ok = p["domain"] == gd
        fs_ok = dom_ok and gfields == pfields
        stage = "ok" if exact else "domain" if not dom_ok else "field_set" if not fs_ok else "values"
        present = {f.split("/", 1)[1] for f, q in p["present"].items() if q >= threshold}
        agg["n"] += 1
        agg["domain_ok"] += dom_ok
        agg["presence_set_ok"] += dom_ok and present == gfields
        agg["field_set_ok"] += fs_ok
        agg["json_ok"] += exact
        agg["stage_" + stage] += 1
        # spans
        gsp = {(s, e, f) for s, e, f, _ in explicit_spans(r)} if r["domain"] else set()
        gop = {(s, e, f): op for s, e, f, op in explicit_spans(r)} if r["domain"] else {}
        psp = {(s, e, f): op for s, e, f, op, *_ in p["spans"]}
        agg["span_gold"] += len(gsp)
        agg["span_pred"] += len(psp)
        agg["span_tp"] += len(gsp & set(psp))
        for s, e, f in gsp:
            hit = [k for k in psp if k[2] == f and k[0] < e and k[1] > s]
            agg["span_overlap_tp"] += bool(hit)
            if (s, e, f) in psp:
                agg["op_n"] += 1
                agg["op_ok"] += psp[(s, e, f)] == gop[(s, e, f)]
        # field values, for fields present in both
        if r["domain"] and pj and pj["category"] == r["domain"]:
            for k, v in params(r).items():
                if k in pj["params"]:
                    agg["value_n"] += 1
                    agg["value_ok"] += values_equal(pj["params"][k], v)
        for key, val in (("domain", gd), ("quality", r.get("quality") or "?"), ("language", _language(r))):
            slices[f"{key}={val}"]["n"] += 1
            slices[f"{key}={val}"]["json_ok"] += exact
            slices[f"{key}={val}"]["field_set_ok"] += fs_ok
        if records:
            recs.append({"id": r["id"], "query": r["query"], "gold": gold and gold["params"], "domain": r["domain"],
                         "pred_domain": p["domain"], "pred": pj and pj["params"], "exact": exact, "stage": stage,
                         "spans": [(r["query"][s:e], f, op, *rest) for s, e, f, op, *rest in p["spans"]]})
    n = agg["n"]
    prec = agg["span_tp"] / max(1, agg["span_pred"])
    rec = agg["span_tp"] / max(1, agg["span_gold"])
    out = {"policy": policy, "value_policy": value_policy, "fallback": fallback, "threshold": threshold, "n": n,
           "domain_acc": agg["domain_ok"] / n,
           "presence_field_set_exact": agg["presence_set_ok"] / n,
           "field_set_exact": agg["field_set_ok"] / n,
           "json_exact": agg["json_ok"] / n,
           "span_p": prec, "span_r": rec, "span_f1": 2 * prec * rec / max(1e-9, prec + rec),
           "span_overlap_r": agg["span_overlap_tp"] / max(1, agg["span_gold"]),
           "op_acc": agg["op_ok"] / max(1, agg["op_n"]),
           "value_acc_on_shared_fields": agg["value_ok"] / max(1, agg["value_n"]),
           "waterfall": {s: agg["stage_" + s] / n for s in ("ok", "domain", "field_set", "values")},
           "slices": {k: {"n": c["n"], "json_exact": c["json_ok"] / c["n"], "field_set_exact": c["field_set_ok"] / c["n"]}
                      for k, c in sorted(slices.items())}}
    if records:
        out["records"] = recs
    return out
