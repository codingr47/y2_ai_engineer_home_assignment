# Fine-Tuned Tagger — Experiment 4: decoding fixes and local non-search answers (2026-11-17)

**Method.** No model was retrained, and test was not used. The final ensemble's saved outputs on val
(`experiment2/results/predictions_clean_*.jsonl`, cleaned queries) are re-decoded with the fixed normalizer
(`redecode.py`). Each fix is switched off in turn.

| variant | val_eval JSON | val_calib JSON |
|---|---|---|
| all fixes off (= experiment 2) | 91.65% | 90.49% |
| F1 off | 91.71% | 90.49% |
| F2 off | 91.71% | 90.49% |
| F3 off | 91.65% | 90.49% |
| **all on** | **91.71%** | **90.49%** |

**The fixes** (`src/tagger/normalize.py`):
- **F1:** storage units glued to their number («1TB», «512GB») are split before reading. **0 rows changed.**
- **F2:** canonical names up to hyphens, spaces or a glued one-letter prefix («נאות-כרמים» = «נאות כרמים»). When the
  labels use both spellings, the more frequent one wins. **0 rows changed.**
- **F3:** one property type, the most specific («דירת גג» → גג/טיפוס, not [דירה, גג/טיפוס]). **+1 row** on val_eval.

**Why so little:** in the cases that motivated F1 and F2, the tagger produces **no span or a partial one**:
- storage: «ו-1TB» gives no span, or «ו-» + «TB» as fragments
- neighbourhoods: names never seen in training («עין גנים», «מעונות ים»), or partial spans («סם בוי»)

The normalizer can only fix what the tagger hands it. These errors need training data (bucket B5 and the place
names of B1 in `docs/data_requests/y2_train_v3.md`), not decoding rules. The fixes stay: they're correct, and F1/F2
will apply once the tagger finds those spans.

**Local non-search answers** (`src/tagger/parser.py`): when the tagger reads a query as not a search *and* the gate
accepts it, the API answers `category: null` locally instead of calling the LLM. In the live test run, all 100
non-search queries were already right, and the 93 `unsupported_category` escalations each cost an LLM call. This
is routing after the gate: no gate input changes.

**Gate:**
- **Not refitted.** A refit on this decoding reproduced the same threshold (0.8178), with the audit within one row
  (95.4% at 86.1%, lower bound 0.942).
- **Code binding re-stamped** with `scripts/rebind_tagger_gate.py`, with this report as recorded evidence. The fitted
  state is unchanged (`tests/test_runtime_layout.py`).

```bash
python -m training.training_bert.experiment4.redecode      # writes results/ (ablation + re-decoded predictions)
```
