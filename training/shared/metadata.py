"""Raw rows (spans, metadata) and generation-batch tags, loaded lazily per split.

The test file is read only when a caller asks for split 'test' (the --final-test path).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from .paths import REPO

GENERATOR_FILE = REPO.parent / "y2_ai_synthetic_generator" / "data" / "synthetic.jsonl"
def raw_rows(split_name: str, schema: str = "raw") -> dict:
    """id -> row. schema='raw' keeps the stored labels (slot / 77-field runs need the original span keys);
    schema='v2' applies the load-time normalization (shared/normalize.py). Delegates to shared/dataset.py."""
    from .dataset import rows_by_id

    return rows_by_id(split_name, schema)


@lru_cache(maxsize=1)
def generation_batches() -> dict:
    """id -> {batch, generator, rephrase_of} from the generator's notes tag (read-only; {} if unavailable)."""
    if not GENERATOR_FILE.exists():
        return {}
    out = {}
    with open(GENERATOR_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m = re.match(r"\[([^\]]*)\]", r.get("notes") or "")
            toks = m.group(1).split()[2:] if m else []
            kinds = [t for t in toks if not t.startswith(("gen:", "rephrase_of:"))]
            kind = next((t for t in kinds if t.startswith("variation:")), kinds[0] if kinds else "seed")
            out[r["id"]] = {"batch": kind, "generator": next((t[4:] for t in toks if t.startswith("gen:")), None),
                            "rephrase_of": next((t.split(":", 1)[1] for t in toks if t.startswith("rephrase_of:")),
                                                None)}
    return out
