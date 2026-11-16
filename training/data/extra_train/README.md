# Y2 extra training batch (2026-11-16)

**Training only. Never use it for calibration, threshold selection or evaluation.**

3,188 silver rows in 2,132 units (at most 2 rows each: a query and its rephrasing), `tr_` ids.
Generated with the same pipeline, generator rotation and blind extractor as the routing batch, and approved the
same way (the blind LLM extraction agreed exactly; nothing checked by a person, no hand-written rows).

No row matches any existing row (every split, `synthetic*.jsonl`, annotations, the routing batch, the parser
repo's `training/data/split1` and `fresh_routing_calibration`, the earlier exports) by the same normalized
query, near-duplicate (character 4-gram Jaccard >= 0.6) or contained query: see report.md. Identical labels are
allowed; 323 rows have one, tagged `metadata.label_seen_before`. Because no query is shared, adding these
rows to training does not leak into the routing / calibration set or the test splits.

## Files
- `train_extra.jsonl`: one row per line, same schema as the dataset export's `train.jsonl` (+ label_seen_before).
- `report.md`: distributions and the leakage audit.
- `manifest.json`: seed, config, run costs, ids, sha256 of train_extra.jsonl, every file deduplicated against.
