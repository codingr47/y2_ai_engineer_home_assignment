"""Inference for the fine-tuned tagger: ensemble prediction and decoding to {category, params}.

Shared by training (training_bert.common.evaluate wraps `predict`) and the runtime parser (src.tagger.parser).
"""
from __future__ import annotations

import torch

from .vocab import DOMAINS, OPS, decode_tags, encode_inputs


def collate_inputs(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        att[i, :k] = torch.tensor(b["attention_mask"])
    return {"input_ids": ids, "attention_mask": att}


@torch.no_grad()
def predict(model, tok, queries: list[str], device, batch_size: int = 64, amp: bool = False,
            ids: list | None = None) -> list[dict]:
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
    ids = list(ids) if ids is not None else list(range(len(queries)))
    for k in range(0, len(queries), batch_size):
        chunk = queries[k:k + batch_size]
        enc = [encode_inputs(q, tok) for q in chunk]
        b = collate_inputs(enc, tok.pad_token_id)
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
            out.append({"id": ids[k + i], "domain": DOMAINS[dom[i]],
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
