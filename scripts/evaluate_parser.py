"""Report actual parser outcomes; live OpenAI runs require --allow-paid-calls.

The existing local candidate/gate audit in tests/eval remains a separate baseline.
"""
import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.parsing.fallback import FallbackRouter
from src.contracts.http_models import SearchResponse
from src.infrastructure.metrics import Metrics
from src.llm.parser import OpenAIParser, ProviderFailure, create_client
from src.parsing.parser import FallbackRequired
from src.config.settings import ROOT, Settings


class HTTPParser:
    """Evaluation adapter: all parsing goes through the public HTTP endpoint."""
    def __init__(self, client):
        self.client = client
        self.last_status = None
        self.last_usage = {}
        self.provider_called = None

    async def parse(self, query):
        import httpx
        self.last_status = None
        self.last_usage = {}
        self.provider_called = None
        try:
            response = await self.client.post("/parse", json={"q": query})
        except httpx.TimeoutException:
            raise ProviderFailure("http_timeout", 504) from None
        except httpx.RequestError:
            raise ProviderFailure("http_connection_error", 503) from None
        self.last_status = response.status_code
        try:
            usage = json.loads(response.headers["X-OpenAI-Usage"])
            if type(usage["provider_called"]) is not bool:
                raise ValueError("invalid usage metadata")
            tokens = usage["tokens"]
            if tokens is not None and not isinstance(tokens, dict):
                raise ValueError("invalid token metadata")
            self.last_usage = tokens or {}
            self.provider_called = usage["provider_called"]
        except (KeyError, ValueError, TypeError):
            pass
        if response.status_code != 200:
            # Report status, without copying potentially sensitive server messages.
            raise ProviderFailure(f"http_{response.status_code}", response.status_code)
        return SearchResponse.model_validate(response.json())


def load_rows(split):
    if split == "val":
        from training.training_frozen_fusion.shared.dataset import load_rows as load
        return load("val")
    from training.training_frozen_fusion.shared.normalize import normalize_row
    path = ROOT / "eval/dataset/data/test.jsonl"
    return [normalize_row({**r, "domain": r["expected"]["category"]})
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip() for r in [json.loads(line)]]


def public_payload(category, params):
    keys = {"יצרנים": "יצרן", "דגמים": "דגם"} if category == "רכב" else {}
    return {"category": category, "params": {keys.get(k, k): v for k, v in params.items()}}


async def evaluate(parser, rows, provider=None, on_record=None):
    """Sequential, uncached calls keep latency and per-row token attribution comparable."""
    from src.extraction.types import values_equal
    records = []
    for row in rows:
        gold = public_payload(row["domain"], row["expected"]["params"])
        pred, error = None, None
        if provider:
            provider.last_usage = {}
        started = perf_counter()
        try:
            result = await parser.parse(row["query"].strip())
            result = SearchResponse.model_validate(result.model_dump())
            pred = {"category": result.category, "params": result.params}
        except ProviderFailure as exc:
            error = exc.code
        except FallbackRequired as exc:
            error = "fallback_required:" + exc.detail["reason"]
        except Exception:
            error = "parser_error"
        category_correct = pred is not None and pred["category"] == gold["category"]
        fields_correct = category_correct and set(pred["params"]) == set(gold["params"])
        exact = fields_correct and all(values_equal(pred["params"][k], v) for k, v in gold["params"].items())
        if gold["category"] is None:
            exact = category_correct and not pred["params"]
        metadata = row.get("metadata") or {}
        records.append({"id": row["id"], "query": row["query"], "gold": gold, "pred": pred,
                        "category_correct": bool(category_correct), "field_set_exact": bool(fields_correct),
                        "json_exact": bool(exact), "error": error, "latency_seconds": perf_counter() - started,
                        "usage": dict(provider.last_usage) if provider else dict(getattr(parser, "last_usage", {})),
                        "provider_called": getattr(parser, "provider_called", None),
                        "http_status": getattr(parser, "last_status", None),
                        "language": "+".join(metadata.get("language") or ["unknown"])})
        if on_record is not None:
            on_record(records[-1])
    return records


def summarize(records):
    def percentile(values, fraction):
        ordered = sorted(values)
        if not ordered:
            return None
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    def block(rows):
        n = len(rows)
        latencies = [r["latency_seconds"] for r in rows]
        usage = Counter()
        for row in rows:
            usage.update(row["usage"])
        return {"rows": n, **{key: sum(r[key] for r in rows) / n if n else None
                              for key in ("category_correct", "field_set_exact", "json_exact")},
                "failures": dict(Counter(r["error"] for r in rows if r["error"])),
                "latency_p50_seconds": percentile(latencies, .5),
                "latency_p95_seconds": percentile(latencies, .95),
                "successful_latency_p95_seconds": percentile([r["latency_seconds"] for r in rows if not r["error"]], .95),
                "tokens": dict(usage)}

    slices = {}
    for key in ("category", "language"):
        groups = defaultdict(list)
        for row in records:
            groups[str(row["gold"]["category"] if key == "category" else row[key])].append(row)
        slices[key] = {name: block(rows) for name, rows in groups.items()}
    return {"overall": block(records), "slices": slices}


def write_reports(directory, summary, records):
    from scripts.evaluation_cost import estimate_cost, request_cost
    for record in records:
        record.update(request_cost(summary["configuration"].get("model"), record))
    priced = [record for record in records if record["estimated_cost_usd"] is not None]
    summary["request_cost_coverage"] = {"priced_requests": len(priced), "unavailable_requests": len(records) - len(priced),
                                         "sum_estimated_cost_usd": sum(record["estimated_cost_usd"] for record in priced)}
    summary["cost"] = estimate_cost(summary)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    config, overall = summary["configuration"], summary["overall"]
    lines = ["# Parser evaluation", "", f"Mode: {config['mode']}; split: {config['split']}; model: {config['model']}; effort: {config['reasoning_effort']}",
             "", "| Metric | Value |", "| --- | --- |"]
    lines += [f"| {key} | {json.dumps(value, ensure_ascii=False)} |" for key, value in overall.items()]
    for key, groups in summary["slices"].items():
        lines += ["", f"## By {key}", "", "| Slice | Rows | Category accuracy | Field set exact | JSON exact |", "| --- | --- | --- | --- | --- |"]
        lines += [f"| {name} | {b['rows']} | {b['category_correct']:.4f} | {b['field_set_exact']:.4f} | {b['json_exact']:.4f} |"
                  for name, b in groups.items()]
    lines += ["", "Estimated token cost: `" + json.dumps(summary["cost"]) + "`", "",
              "Routing: `" + json.dumps(summary["routing"]) + "`", "",
              "Cache counter changes: `" + json.dumps(summary.get("cache", {})) + "`", "",
              "Deployment: `" + json.dumps(summary.get("deployment", {})) + "`", "",
              "Latency scope: " + config.get("latency_scope", "direct parser invocation") + ".",
              "Model confidence is an uncalibrated self-rating. Failed rows count as incorrect.",
              "Token breakdowns overlap: cached input is included in input; reasoning is included in output.",
              "Per-query records contain dataset query text and predictions."]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    from scripts.evaluation_html import render_html
    (directory / "REPORT.html").write_text(render_html(summary, records), encoding="utf-8")


async def run_evaluation(*, mode, split="val", settings=None, limit=None, output=None, allow_paid_calls=False, url=None):
    if mode not in {"local", "openai", "combined", "api"} or split not in {"val", "test"}:
        raise ValueError("invalid evaluation mode or split")
    if mode != "local" and not allow_paid_calls:
        raise ValueError("OpenAI evaluation requires explicit paid-call opt-in")
    settings = settings or Settings.from_env()
    if mode == "api":
        if not url:
            raise ValueError("API evaluation requires --url")
        return await run_api_evaluation(url=url, split=split, settings=settings, limit=limit, output=output)
    if mode != "local":
        settings = Settings.model_validate({**settings.model_dump(), "openai_fallback_enabled": True})
    rows = load_rows(split)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    metrics, client, provider = Metrics(), None, None
    try:
        if mode != "openai":
            import anyio
            from src.frozen_fusion.local_parser import LocalParser
            parser = await anyio.to_thread.run_sync(lambda: LocalParser.load(settings, metrics=metrics))
        if mode != "local":
            client = create_client(settings)
            provider = OpenAIParser(client, settings, metrics)
            parser = provider if mode == "openai" else FallbackRouter(parser, provider)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = Path(output or ROOT / "eval/results") / f"{mode}_{split}_{stamp}_{uuid4().hex[:8]}"
        directory.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = directory.with_suffix(".partial.jsonl")
        print(f"Evaluating {len(rows)} rows; mode={mode}; split={split}; model={settings.openai_model if provider else None}; effort={settings.openai_reasoning_effort if provider else None}", flush=True)
        print(f"Checkpoint: {checkpoint}", flush=True)
        completed, correct, failed = 0, 0, 0
        evaluation_started = perf_counter()
        with checkpoint.open("x", encoding="utf-8") as stream:
            def record_progress(record):
                nonlocal completed, correct, failed
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                completed += 1
                correct += int(record["json_exact"])
                failed += int(record["error"] is not None)
                if completed == 1 or completed % 10 == 0 or completed == len(rows):
                    print(f"{completed}/{len(rows)} rows; JSON exact={correct / completed:.1%}; failures={failed}; elapsed={perf_counter() - evaluation_started:.0f}s", flush=True)

            records = await evaluate(parser, rows, provider, on_record=record_progress)
        summary = summarize(records)
        summary["configuration"] = {"mode": mode, "split": split, "model": settings.openai_model if provider else None,
                                    "reasoning_effort": settings.openai_reasoning_effort if provider else None,
                                    "timeout_seconds": settings.openai_timeout_seconds if provider else None,
                                    "max_output_tokens": settings.openai_max_output_tokens if provider else None,
                                    "limit": limit,
                                    "dataset_sha256": hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                                    "prompt_sha256": hashlib.sha256(provider.system_prompt.encode()).hexdigest() if provider else None,
                                    "taxonomy_sha256": hashlib.sha256(provider.catalog.source).hexdigest() if provider else None}
        if provider and hasattr(provider, "prompt_metadata"):
            summary["configuration"].update(provider.prompt_metadata)
        summary["routing"] = {json.dumps(sample.labels, sort_keys=True): sample.value
                              for family in metrics.routing.collect() for sample in family.samples
                              if sample.name == "parse_routing_total"}
        summary["provider_outcomes"] = {sample.labels["outcome"]: sample.value
                                       for family in metrics.provider_calls.collect() for sample in family.samples
                                       if sample.name == "openai_calls_total"}
        write_reports(directory, summary, records)
        checkpoint.unlink()
        return directory, summary
    finally:
        if client is not None:
            await client.close()


async def run_api_evaluation(*, url, split, settings, limit=None, output=None, deployment=None, concurrency=1):
    """Measure HTTP responses, including transport, validation, middleware and caching."""
    import httpx
    from prometheus_client.parser import text_string_to_metric_families
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    async def snapshot(client):
        response = await client.get("/metrics")
        response.raise_for_status()
        return {(sample.name, tuple(sorted(sample.labels.items()))): sample.value
                for family in text_string_to_metric_families(response.text) for sample in family.samples
                if sample.name.endswith("_total")}

    rows = load_rows(split)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(output or ROOT / "eval/results") / f"api_{split}_{stamp}_{uuid4().hex[:8]}"
    directory.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = directory.with_suffix(".partial.jsonl")
    async with httpx.AsyncClient(base_url=url.rstrip("/"), timeout=settings.openai_timeout_seconds + 10,
                                 trust_env=False) as client:
        (await client.get("/health")).raise_for_status()
        before = await snapshot(client)
        completed, correct, failed = 0, 0, 0
        started = perf_counter()
        print(f"HTTP evaluation: {len(rows)} dataset rows; endpoint={url.rstrip('/')}/parse; split={split}; concurrency={concurrency}", flush=True)
        print(f"Checkpoint: {checkpoint}", flush=True)
        with checkpoint.open("x", encoding="utf-8") as stream:
            def progress(record):
                nonlocal completed, correct, failed
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                completed += 1
                correct += int(record["json_exact"])
                failed += int(record["error"] is not None)
                if completed == 1 or completed % 10 == 0 or completed == len(rows):
                    print(f"{completed}/{len(rows)} HTTP requests; JSON exact={correct / completed:.1%}; failures={failed}; elapsed={perf_counter() - started:.0f}s", flush=True)

            if concurrency == 1:
                records = await evaluate(HTTPParser(client), rows, on_record=progress)
            else:
                semaphore = asyncio.Semaphore(concurrency)

                async def send(row):
                    async with semaphore:
                        return (await evaluate(HTTPParser(client), [row], on_record=progress))[0]

                records = await asyncio.gather(*(send(row) for row in rows))
        after = await snapshot(client)
    deltas = {key: value - before.get(key, 0) for key, value in after.items()}
    summary = summarize(records)
    summary["configuration"] = {"mode": "api", "split": split, "url": url, "limit": limit,
                                "model": settings.openai_model, "reasoning_effort": settings.openai_reasoning_effort,
                                "model_source": "client .env; server deployment must match",
                                "concurrency": concurrency,
                                "latency_scope": f"client-observed POST /parse over HTTP, {concurrency} concurrent clients; excludes client queue wait",
                                "dataset_path": str(ROOT / "eval/dataset/data/test.jsonl") if split == "test" else "training/training_frozen_fusion/data/split1/val.jsonl",
                                "dataset_sha256": hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    summary["routing"] = {json.dumps(dict(labels), sort_keys=True): value
                          for (name, labels), value in deltas.items() if name == "parse_routing_total"}
    summary["provider_outcomes"] = {dict(labels)["outcome"]: value
                                   for (name, labels), value in deltas.items() if name == "openai_calls_total"}
    summary["cache"] = {name: value for (name, labels), value in deltas.items()
                        if name in {"parse_cache_hits_total", "parse_cache_misses_total"}}
    summary["overall"]["tokens"] = {dict(labels)["kind"]: value
                                     for (name, labels), value in deltas.items() if name == "openai_tokens_total"}
    summary["http_statuses"] = dict(Counter(str(record["http_status"]) for record in records))
    if deployment is not None:
        summary["deployment"] = deployment
        summary["configuration"]["model_source"] = "Docker runtime environment recorded in deployment"
        from src.llm.few_shots import prompt_configuration
        from src.llm.catalog import Catalog
        _, metadata = prompt_configuration(Catalog(), settings.openai_few_shot_count,
                                            guidelines_enabled=settings.openai_guidelines_enabled)
        summary["configuration"].update(metadata)
    write_reports(directory, summary, records)
    checkpoint.unlink()
    return directory, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["local", "openai", "combined", "api"], required=True)
    ap.add_argument("--url", help="Base URL of the running API; required for --mode api")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--model")
    ap.add_argument("--reasoning-effort")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--allow-paid-calls", action="store_true")
    args = vars(ap.parse_args())
    settings = Settings.from_env()
    model, effort = args.pop("model"), args.pop("reasoning_effort")
    overrides = {k: v for k, v in {"openai_model": model, "openai_reasoning_effort": effort}.items() if v is not None}
    settings = Settings.model_validate({**settings.model_dump(), **overrides})
    directory, summary = asyncio.run(run_evaluation(settings=settings, **args))
    print(f"Report: {directory / 'REPORT.md'}")
    return int(bool(summary["overall"]["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
