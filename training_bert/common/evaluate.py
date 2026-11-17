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

from .data import DOMAINS, OPS, decode_tags, encode, explicit_spans, params
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


@torch.no_grad()
def predict(model, tok, rows: list[dict], device, batch_size: int = 64, amp: bool = False) -> list[dict]:
    """Raw per-row predictions: domain, presence probabilities, decoded spans with operators, implied values.
    `model` may be a list of models sharing one vocabulary: an ensemble that averages every head's probabilities
    (domain, presence, BIO tags, the operator of each decoded span, implied values)."""
    models = model if isinstance(model, (list, tuple)) else [model]
    for m in models:
        m.eval()
    model = models[0]
    vocab = model.vocab
    assert all(m.vocab.to_json() == vocab.to_json() for m in models), "ensemble members must share a vocabulary"
    fields = vocab.fields
    dom_of = [f.split("/", 1)[0] for f in fields]
    # tag mask per domain: only that domain's B-/I- tags (and O) are allowed
    tag_masks = {}
    for d in DOMAINS[1:]:
        m = torch.full((len(vocab.tags),), float("-inf"))
        m[0] = 0
        for j, f in enumerate(fields):
            if dom_of[j] == d:
                m[1 + 2 * j] = m[2 + 2 * j] = 0
        tag_masks[d] = m.to(device)
    out = []
    for k in range(0, len(rows), batch_size):
        chunk = rows[k:k + batch_size]
        enc = [encode(r, tok, vocab) for r in chunk]
        b = collate(enc, tok.pad_token_id)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            outs = [m(b["input_ids"].to(device), b["attention_mask"].to(device)) for m in models]
        o = outs[0]
        mean = lambda key, fn: torch.stack([fn(x[key].float()) for x in outs]).mean(0)
        dom_prob = mean("domain", lambda t: t.softmax(-1))
        dom = dom_prob.argmax(-1).tolist()
        dom_p = dom_prob.max(-1).values.tolist()
        pres = mean("present", torch.sigmoid).cpu()
        tag_prob = mean("tags", lambda t: t.softmax(-1))
        tag_logits = tag_prob.clamp_min(1e-9).log()
        rows_spans, flat = [], []
        tok_conf = [1.0] * len(chunk)
        agree = [len(models)] * len(chunk)
        for i, r in enumerate(chunk):
            d = DOMAINS[dom[i]]
            if d == "none":
                rows_spans.append([])
                continue
            masked = (tag_logits[i] + tag_masks[d]).softmax(-1)
            tags = masked.argmax(-1).tolist()
            sp = decode_tags(tags, enc[i]["offsets"], vocab)
            rows_spans.append(sp)
            # confidence: lowest per-token tag probability over the row's real tokens (spans and O alike)
            real = [q for q, (a, b2) in enumerate(enc[i]["offsets"]) if a != b2]
            tok_conf[i] = float(masked[real].max(-1).values.min()) if real else 1.0
            if len(models) > 1:   # members that decode the same domain and the same spans as the ensemble
                key = {(a, b2, f) for a, b2, f, _, _ in sp}
                agree[i] = 0
                for x in outs:
                    if DOMAINS[int(x["domain"][i].float().argmax())] != d:
                        continue
                    mt = (x["tags"][i].float() + tag_masks[d]).argmax(-1).tolist()
                    agree[i] += {(a, b2, f) for a, b2, f, _, _ in decode_tags(mt, enc[i]["offsets"], vocab)} == key
            flat.extend((i, ts, te) for _, _, _, ts, te in sp)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            span_outs = [m.span_logits(x["hidden"], flat) for m, x in zip(models, outs)]
        op_l, val_l = span_outs[0]
        if flat and len(models) > 1:
            op_l = torch.stack([so[0].float().softmax(-1) for so in span_outs]).mean(0)
        op_prob = op_l.float().softmax(-1) if flat and len(models) == 1 else op_l.float() if flat else None
        op_ids = op_prob.argmax(-1).tolist() if flat else []
        op_p = op_prob.max(-1).values.tolist() if flat else []
        heads = [None] * len(flat)
        if flat and val_l is not None:
            fields_flat = [f for i in range(len(chunk)) for _, _, f, _, _ in rows_spans[i]]
            cat = [q for q, f in enumerate(fields_flat) if f in vocab.values]
            if cat:
                probs = torch.softmax(val_l[cat].float() + model.value_mask_for([fields_flat[q] for q in cat]), -1)
                p, ix = probs.max(-1)
                for n, q in enumerate(cat):
                    heads[q] = (vocab.value_index[int(ix[n])][1], float(p[n]))
        imp = [torch.stack([x["implied"][n].float().softmax(-1) for x in outs]).mean(0).argmax(-1).tolist()
               for n in range(len(model.implied_fields))]
        j = 0
        for i, r in enumerate(chunk):
            spans, op_conf = [], 1.0
            for cs, ce, f, _, _ in rows_spans[i]:
                spans.append((cs, ce, f, OPS[op_ids[j]], heads[j]))
                op_conf = min(op_conf, op_p[j])
                j += 1
            implied = {f: vocab.implied[f][imp[n][i]] for n, f in enumerate(model.implied_fields)}
            out.append({"id": r["id"], "domain": DOMAINS[dom[i]],
                        "present": {f: float(pres[i, q]) for q, f in enumerate(fields)
                                    if pres[i, q] >= 0.05 and dom_of[q] == DOMAINS[dom[i]]},
                        "spans": spans, "implied": implied,
                        "conf": {"domain": dom_p[i], "token_min": tok_conf[i], "op_min": op_conf,
                                 "agree": agree[i], "members": len(models)}})
    return out


def to_json(pred: dict, query: str, norm, policy: str, threshold: float = 0.5,
            value_policy: str = "lexicon", fallback: str = "none") -> dict | None:
    d = pred["domain"]
    if d == "none":
        return None
    present = {f for f, p in pred["present"].items() if p >= threshold}
    spans = [(f, query[s:e], op, h, s, e) for s, e, f, op, *rest in pred["spans"] for h in [rest[0] if rest else None]
             if policy == "spans" or f in present]
    span_fields = {sp[0] for sp in spans}
    implied = {f: v for f, v in pred["implied"].items() if f in present and f not in span_fields}
    return norm.assemble(d, spans, implied, value_policy=value_policy, query=query, fallback=fallback,
                         present_without_span=present - span_fields - set(implied))


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
