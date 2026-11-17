"""Open-loop throughput / latency benchmark of the LOCAL parsing path (no LLM calls, no cost).

    # server: PARSER_MODE=local, LOCAL_PARSER=tagger, CACHE_MAX_SIZE=0
    python -m scripts.benchmark_local --url http://127.0.0.1:8011 --rates 8 12 16 --duration 30 --output docs/benchmark_local.json

- **Queries:** unique validation queries (never test), sent on a fixed schedule (open loop). The latency of each
  request is measured from its scheduled time, so server queueing shows up instead of hiding in a slow sender.
- **What counts as a completed local parse:** 200 (answered locally) and 503 `fallback_required` (the gate
  rejected it). Both ran the full local inference. Anything else counts as an error.
- **What this measures:** the capacity of one container's local path. The LLM path is network-bound and measured
  by the evaluation reports.
"""
import argparse
import asyncio
import json
import platform
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "training/training_bert/data/val.jsonl"


def percentile(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else None


async def run_rate(client, queries, rate, duration):
    n = int(rate * duration)
    latencies, statuses, errors = [], {}, 0
    started = time.perf_counter() + 0.2

    async def send(i):
        nonlocal errors
        scheduled = started + i / rate
        await asyncio.sleep(max(0, scheduled - time.perf_counter()))
        try:
            r = await client.post("/parse", json={"q": queries[i % len(queries)]})
            statuses[r.status_code] = statuses.get(r.status_code, 0) + 1
            if r.status_code in (200, 503):
                latencies.append(time.perf_counter() - scheduled)
            else:
                errors += 1
        except httpx.HTTPError:
            errors += 1

    await asyncio.gather(*(send(i) for i in range(n)))
    elapsed = time.perf_counter() - started
    return {"offered_qps": rate, "duration_s": round(elapsed, 1), "completed": len(latencies), "errors": errors,
            "statuses": statuses, "achieved_qps": round(len(latencies) / elapsed, 2),
            "p50_ms": round(1000 * percentile(latencies, .5), 1), "p95_ms": round(1000 * percentile(latencies, .95), 1),
            "p99_ms": round(1000 * percentile(latencies, .99), 1)}


async def main_async(a):
    queries = list(dict.fromkeys(json.loads(l)["query"].strip() for l in VAL.read_text(encoding="utf-8").splitlines()
                                 if l.strip()))
    async with httpx.AsyncClient(base_url=a.url.rstrip("/"), timeout=60, trust_env=False) as client:
        (await client.get("/health")).raise_for_status()
        for q in queries[:5]:                                   # warm-up
            await client.post("/parse", json={"q": q})
        results, offset = [], 5
        for rate in a.rates:
            chunk = queries[offset:] + queries[:offset]
            results.append(await run_rate(client, chunk, rate, a.duration))
            offset += int(rate * a.duration)
            r = results[-1]
            print(f"{rate:5.1f} QPS offered -> {r['achieved_qps']:5.2f} achieved | p50 {r['p50_ms']} ms p95 {r['p95_ms']} ms "
                  f"p99 {r['p99_ms']} ms | {r['statuses']} errors {r['errors']}", flush=True)
    report = {"what": "local path (PARSER_MODE=local, LOCAL_PARSER=tagger), cache off, unique validation queries",
              "client": {"platform": platform.platform(), "python": platform.python_version()},
              "server_note": a.server_note, "results": results}
    if a.output:
        Path(a.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--rates", type=float, nargs="+", default=[8, 12, 16])
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--server-note", default="", help="server hardware / container limits, recorded in the report")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
