"""Score saved tagger runs on val_eval / val_calib, and (only with --with-test) once on the held-out test split.

    python -m training.training_bert.final_test training/training_bert/experiment1/runs/e6_lr5e-5_s42 [more run dirs ...]          # val only
    python -m training.training_bert.final_test --with-test training/training_bert/experiment1/runs/... # the one final check, last model only
    python -m training.training_bert.final_test --tag=norm2 training/training_bert/experiment1/runs/...  # re-score with a newer normalizer
    python -m training.training_bert.final_test --ensemble=training/training_bert/experiment1/runs/ens_e6 [--fallback=span] runs/a runs/b runs/c

Everything about the model was fixed on val_eval before this runs (LR 5e-5, 8 epochs, policy "spans", the epoch
with the best val_eval JSON). Test is read here and nowhere else; its numbers must not feed back into any choice.
Per run writes <run>/final.json (metrics per split) and <run>/predictions_<split>.jsonl (per-row raw predictions,
kept for error analysis and a later confidence gate calibrated on val_calib).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .common import data as D
from .common.evaluate import predict, score
from .common.model import TaggerModel
from .common.normalize import SpanNormalizer, dump


def load_run(rd: Path, device):
    m = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
    vocab = D.Vocab.from_json(json.loads((rd / "vocab.json").read_text(encoding="utf-8")))
    model = TaggerModel(m["config"]["encoder"], vocab)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v
                           for k, v in torch.load(rd / "model.pt", map_location="cpu").items()})
    return model.to(device), m


def ensemble(run_dirs: list[str], out_dir: str, with_test: bool = False, tag: str = "", fallback: str = "none"):
    """Average the saved runs (same encoder and vocabulary) into one predictor and score it like a single run."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = SpanNormalizer.load()
    loaded = [load_run(Path(rd), device) for rd in run_dirs]
    models = [m for m, _ in loaded]
    cfg = loaded[0][1]["config"]
    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    sfx = f"_{tag}" if tag else ""
    out = {"members": [str(r) for r in run_dirs], "fallback": fallback, "config": cfg}
    for split in ("val_eval", "val_calib") + (("test",) if with_test else ()):
        rows = D.load(split)
        preds = predict(models, tok, rows, device, amp=device.type == "cuda")
        s = score(rows, preds, norm, "spans", records=True, fallback=fallback)
        recs = s.pop("records")
        out[split] = s
        with open(od / f"predictions{sfx}_{split}.jsonl", "w", encoding="utf-8") as f:
            for p, r in zip(preds, recs):
                f.write(json.dumps({**r, "raw": p}, ensure_ascii=False) + "\n")
        print(f"{od.name} {split}: JSON {s['json_exact']:.4f} fields {s['field_set_exact']:.4f} "
              f"domain {s['domain_acc']:.4f} span F1 {s['span_f1']:.4f}", flush=True)
    dump(out, od / f"final{sfx}.json")


def main(run_dirs: list[str], with_test: bool = False, tag: str = ""):
    """tag: suffix for the output files, so a re-score with a newer normalizer never overwrites an earlier one."""
    sfx = f"_{tag}" if tag else ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    norm = SpanNormalizer.load()
    for rd in map(Path, run_dirs):
        m = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
        cfg, best = m["config"], m["best"]
        vocab = D.Vocab.from_json(json.loads((rd / "vocab.json").read_text(encoding="utf-8")))
        tok = AutoTokenizer.from_pretrained(cfg["encoder"])
        model = TaggerModel(cfg["encoder"], vocab)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v
                               for k, v in torch.load(rd / "model.pt", map_location="cpu").items()})
        model.to(device)
        out = {"run": rd.name, "epoch": best["epoch"], "policy": best["policy"], "config": cfg}
        for split in ("val_eval", "val_calib") + (("test",) if with_test else ()):
            rows = D.load(split)
            preds = predict(model, tok, rows, device, amp=amp)
            pol, _, vpol = best["policy"].partition("/")
            s = score(rows, preds, norm, pol, records=True, value_policy=vpol or "lexicon")
            recs = s.pop("records")
            out[split] = s
            with open(rd / f"predictions{sfx}_{split}.jsonl", "w", encoding="utf-8") as f:
                for p, r in zip(preds, recs):
                    f.write(json.dumps({**r, "raw": p}, ensure_ascii=False) + "\n")
            print(f"{rd.name} {split}: JSON {s['json_exact']:.4f} fields {s['field_set_exact']:.4f} "
                  f"domain {s['domain_acc']:.4f} span F1 {s['span_f1']:.4f}", flush=True)
        dump(out, rd / f"final{sfx}.json")


if __name__ == "__main__":
    args = sys.argv[1:]
    opt = lambda name, default="": next((a.split("=", 1)[1] for a in args if a.startswith(f"--{name}=")), default)
    dirs = [a for a in args if not a.startswith("--")]
    if opt("ensemble"):   # --ensemble=<out dir>: average the given runs
        ensemble(dirs, opt("ensemble"), "--with-test" in args, opt("tag"), opt("fallback", "none"))
    else:
        main(dirs, with_test="--with-test" in args, tag=opt("tag"))
