# Prompt-injection protection at the LLM gate

`OpenAIParser` checks input before any provider call, in both OpenAI-only and
combined fallback modes. High-confidence instruction overrides, forged roles,
prompt/secret extraction requests, forced output, and encoded command requests
return HTTP **400** with:

```json
{"detail":{"code":"prompt_injection_detected"}}
```

Rejected requests are not cached and make no OpenAI call. Logs record only a
bounded reason, never the query. The counter
`llm_security_rejections_total{reason}` records these decisions separately from
provider calls. Local parsing and calibrated artifacts are unchanged.

Detection inspects normalized text, including common Unicode, invisible-character,
HTML, percent-encoding, and literal Unicode-escape disguises. It never rewrites
the original query sent to the model. These checks are heuristics; novel attacks
can bypass them, and quoted attack text can be rejected.

A concise mandatory trusted `SECURITY` policy tells the model to treat the query
as search data, ignore embedded instructions, and exclude instruction text from
filters. It remains enabled when optional extraction guidelines are disabled.
The query stays JSON-encoded in a separate user message. Existing strict output
schema and catalog validation remain in place. This is layered protection, not
a guarantee against every injection (see the [OpenAI safety guidance](https://developers.openai.com/api/docs/guides/agent-builder-safety)).

## Verification

Offline tests cover English and Hebrew attacks, obfuscation, benign search
corrections/negations, unchanged query serialization, mandatory policy, metrics,
and HTTP rejection without caching, query logging, or provider calls:

```powershell
python -m pytest tests/api/test_injection_guard.py -q
```

The security evaluation builds the current Docker image, starts an isolated API
with cache disabled, and sends real `POST /parse` requests. Recognized attacks
must return 400 without provider calls. Subtler instructions must either be
blocked or preserve the expected clean search. Normal queries must parse correctly.

```powershell
python -m scripts.evaluate_security --model gpt-6.1-terra --allow-paid-calls
```

This small suite uses low reasoning, twelve reviewed examples, and extraction
guidelines. It writes `summary.json`, `records.jsonl`, and standalone `REPORT.html`
with per-request usage, estimated cost, and responses. The first live run passed
19/19 cases: thirteen blocked attacks, three subtler attacks handled by the model,
and three benign controls. Estimated total cost was $0.00956725. Its report is
[`security_20261117T125937Z_c1121a25`](../eval/results/security_20261117T125937Z_c1121a25/REPORT.html).
Passing this finite suite does not establish resistance to unseen attacks.

An offline scan of the existing validation and test datasets flagged six explicit
injection queries in each split and no other rows. Their existing extraction
labels still expect HTTP 200; the unchanged accuracy evaluator will count the new
400 responses as failures. Labels and historical results were not altered.
Security-policy and guard versions are recorded in prompt metadata; the mandatory
policy changes the effective prompt hash, so historical accuracy is not a measured
accuracy baseline for this version.
