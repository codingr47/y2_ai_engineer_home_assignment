"""Fine-tune the multi-task tagger and score it on val_eval every epoch.

    python -m training.training_bert.train --out runs/e1_lr3e-5_s42 --lr 3e-5 --seed 42 --epochs 8
    python -m training.training_bert.train --out runs/smoke --train-size 200 --epochs 1 --eval-size 200 --cpu

Writes <out>/metrics.json (config, per-epoch val_eval metrics for each decoding policy, best epoch) and, with --save,
<out>/model.pt (best epoch, bf16 weights) + vocab.json. Test is never touched here (see final_test.py).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .common import data as D
from .common.evaluate import POLICIES, VALUE_POLICIES, collate, predict, score
from .common.model import TaggerModel
from .common.normalize import SpanNormalizer, dump

ENCODER = "dicta-il/dictabert"


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def losses(model, out, b, device):
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    l_dom = ce(out["domain"].float(), b["domain"].to(device))
    l_pres = nn.functional.binary_cross_entropy_with_logits(out["present"].float(), b["present"].to(device))
    l_tag = ce(out["tags"].float().flatten(0, 1), b["tags"].to(device).flatten())
    l_op = l_val = 0.0
    if b["spans"]:
        op_l, val_l = model.span_logits(out["hidden"], b["spans"])
        l_op = ce(op_l.float(), b["ops"].to(device))
        y = b["values"].to(device)
        keep = (y >= 0).nonzero().flatten()
        if val_l is not None and len(keep):
            fields = [model.vocab.value_index[int(t)][0] for t in y[keep]]
            l_val = ce(val_l[keep].float() + model.value_mask_for(fields), y[keep])
    l_imp = 0.0
    for n, f in enumerate(model.implied_fields):
        y = torch.tensor([d.get(f, -100) for d in b["implied"]], device=device)
        if (y >= 0).any():
            l_imp = l_imp + ce(out["implied"][n].float(), y)
    total = l_dom + l_pres * 10 + l_tag + l_op + l_imp + l_val   # BCE over 69 mostly-zero labels is ~10x smaller
    f = lambda x: float(x.detach()) if torch.is_tensor(x) else float(x)
    return total, {"domain": f(l_dom), "present": f(l_pres), "tags": f(l_tag), "op": f(l_op),
                   "implied": f(l_imp), "value": f(l_val)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-size", type=int, default=None, help="family-grouped nested subset of train")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--quality", default=None, help="train only on rows of this quality (e.g. silver)")
    ap.add_argument("--eval-size", type=int, default=None, help="smoke runs only: first N val_eval rows")
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--save", action="store_true", help="keep the best epoch's weights")
    ap.add_argument("--value-head", action="store_true", help="learned value head for categorical spans")
    a = ap.parse_args()

    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    amp = device.type == "cuda" and not a.no_bf16
    set_seed(a.seed)
    all_train = D.load("train")
    norm = SpanNormalizer.load()
    enum_fields = {f for f, fi in norm.ex.model.fields.items() if fi.type == "enum"} if a.value_head else None
    # vocabulary from all of train, so learning-curve runs share one label space
    vocab = D.Vocab.build(all_train, enum_fields)
    train = D.subset(all_train, a.train_size, a.subset_seed, a.quality)
    val = D.load("val_eval")[: a.eval_size] if a.eval_size else D.load("val_eval")
    tok = AutoTokenizer.from_pretrained(a.encoder)
    enc = [D.encode(r, tok, vocab) for r in train]

    model = TaggerModel(a.encoder, vocab).to(device)
    head = set(map(id, model.head_parameters()))
    enc_params = [p for p in model.parameters() if id(p) not in head]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr},
                             {"params": model.head_parameters(), "lr": a.lr * a.head_lr_mult}], weight_decay=0.01)
    steps = a.epochs * math.ceil(len(enc) / a.batch)
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * steps), steps)

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    cfg.update(train_rows=len(train), train_families=len({r["family"] for r in train}), val_rows=len(val),
               device=device.type, gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               bf16_autocast=amp, fields=len(vocab.fields), tags=len(vocab.tags), values=len(vocab.value_index))
    print(json.dumps(cfg, ensure_ascii=False))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "vocab.json").write_text(json.dumps(vocab.to_json(), ensure_ascii=False), encoding="utf-8")
    history, best = [], None
    rng = random.Random(a.seed)
    for ep in range(1, a.epochs + 1):
        model.train()
        order = list(range(len(enc)))
        rng.shuffle(order)
        t0, run = time.time(), []
        for k in range(0, len(order), a.batch):
            b = collate([enc[i] for i in order[k:k + a.batch]], tok.pad_token_id)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(b["input_ids"].to(device), b["attention_mask"].to(device))
                loss, parts = losses(model, out, b, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run.append(parts | {"total": float(loss)})
        train_s = time.time() - t0
        preds = predict(model, tok, val, device, amp=amp)
        vps = VALUE_POLICIES if vocab.values else ("lexicon",)
        evals = {f"{p}/{v}": score(val, preds, norm, p, value_policy=v) for p in POLICIES for v in vps}
        mean = {k: float(np.mean([x[k] for x in run])) for k in run[0]}
        rec = {"epoch": ep, "train_loss": mean, "train_seconds": round(train_s, 1),
               "eval_seconds": round(time.time() - t0 - train_s, 1), "val_eval": evals}
        history.append(rec)
        top = max(evals, key=lambda p: evals[p]["json_exact"])
        e = evals[top]
        print(f"epoch {ep}: loss {mean['total']:.4f} | val_eval[{top}] JSON {e['json_exact']:.4f} "
              f"fields {e['field_set_exact']:.4f} domain {e['domain_acc']:.4f} span F1 {e['span_f1']:.4f} "
              f"op {e['op_acc']:.4f} | {train_s:.0f}s train", flush=True)
        if best is None or e["json_exact"] > best["json_exact"]:
            best = {"epoch": ep, "policy": top, "json_exact": e["json_exact"]}
            if a.save:
                torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v
                            for k, v in model.state_dict().items()}, a.out / "model.pt")
        dump({"config": cfg, "best": best, "history": history}, a.out / "metrics.json")
    print("best", json.dumps(best))


if __name__ == "__main__":
    main()
