"""Open-loop HTTP benchmark: python -m scripts.load_test --url http://127.0.0.1:8000."""

import argparse
import asyncio
import json
import math
import platform
import sys
import time
import uuid
from importlib.metadata import version
from pathlib import Path

import httpx
from prometheus_client.parser import text_string_to_metric_families

from src.models import ParseResponse


async def cache_metrics(client):
    response = await client.get("/metrics")
    response.raise_for_status()
    return {
        sample.name: sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name in {"parse_cache_hits_total", "parse_cache_misses_total",
                           "parse_cache_evictions_total", "parse_cache_entries"}
    }


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


async def workload(client, name, rate, duration):
    run_id = uuid.uuid4().hex
    query = f"דירה {run_id}"
    for i in range(20):
        response = await client.post("/parse", json={"q": f"חימום {run_id} {i}"})
        response.raise_for_status()
    # Seed the repeated workload so all measured calls are cache hits.
    if name == "repeated":
        (await client.post("/parse", json={"q": query})).raise_for_status()
    before = await cache_metrics(client)
    latencies, errors = [], []
    started = time.perf_counter()

    async def send(number, scheduled):
        try:
            response = await client.post("/parse", json={"q": query if name == "repeated" else f"{query} {number}"})
            response.raise_for_status()
            ParseResponse.model_validate(response.json())
        except Exception as exc:
            errors.append(type(exc).__name__)
        else:
            # Include dispatch delay, preventing a stalled generator from hiding latency.
            latencies.append(time.perf_counter() - scheduled)

    tasks = []
    count = int(rate * duration)
    for number in range(count):
        scheduled = started + number / rate
        await asyncio.sleep(max(0, scheduled - time.perf_counter()))
        tasks.append(asyncio.create_task(send(number, scheduled)))
    await asyncio.gather(*tasks)
    elapsed = max(duration, time.perf_counter() - started)
    after = await cache_metrics(client)
    p95 = percentile(latencies, .95)
    achieved = len(latencies) / elapsed
    return {
        "workload": name, "offered_qps": rate, "duration_seconds": elapsed,
        "successful_requests": len(latencies), "failed_requests": len(errors),
        "error_types": sorted(set(errors)), "successful_qps": achieved,
        "p50_ms": percentile(latencies, .5) * 1000 if latencies else None,
        "p95_ms": p95 * 1000 if p95 is not None else None,
        "cache_counter_deltas": {key: after[key] - before.get(key, 0) for key in after if key.endswith("_total")},
        "cache_entries_after": after.get("parse_cache_entries"),
        "passed": achieved >= 12 and not errors and p95 is not None and p95 <= .15,
    }


async def benchmark(args):
    async with httpx.AsyncClient(base_url=args.url.rstrip("/"), timeout=5, trust_env=False) as client:
        (await client.get("/health")).raise_for_status()
        results = []
        for name in ("repeated", "unique"):
            results.append(await workload(client, name, args.rate, args.duration))
    report = {
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "httpx": version("httpx"), "target": args.url,
                        "note": "Client environment; record server hardware and runtime separately."},
        "results": results, "passed": all(result["passed"] for result in results),
    }
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--rate", type=float, default=20)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    if not math.isfinite(args.rate) or not math.isfinite(args.duration) or args.rate <= 0 or args.duration <= 0 or args.rate * args.duration < 1:
        parser.error("rate and duration must be finite positive numbers producing at least one request")
    sys.exit(0 if asyncio.run(benchmark(args)) else 1)


if __name__ == "__main__":
    main()
