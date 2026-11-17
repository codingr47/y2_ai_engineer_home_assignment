"""Build the serving image, launch an isolated API, and evaluate POST /parse."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx

from scripts.evaluate_parser import run_api_evaluation
from src.config.settings import ROOT, Settings


async def docker(*args, env=None):
    process = await asyncio.create_subprocess_exec("docker", *args, cwd=ROOT, env=env,
                                                   stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    if process.returncode:
        # Runtime values, including credentials, never appear in command arguments.
        raise RuntimeError(f"Docker {args[0]} failed: {stderr.decode(errors='replace')[-3000:]}")
    return stdout.decode(errors="replace").strip()


def runtime_configuration(settings, mode, cache_size=0):
    """Explicit configuration isolates each run from host-only paths and flags."""
    values = {"PARSER_MODE": mode, "CACHE_MAX_SIZE": str(cache_size),
              "EXPOSE_TOKEN_USAGE": "true",
              "EXTRACTOR_ENABLED": "false", "OPENAI_FALLBACK_ENABLED": "false",
              "EXTRACTOR_USE_TAGGER": str(settings.extractor_use_tagger).lower(),
              "OPENAI_TIMEOUT_SECONDS": str(settings.openai_timeout_seconds),
              "OPENAI_FEW_SHOT_COUNT": str(settings.openai_few_shot_count),
              "OPENAI_GUIDELINES_ENABLED": str(settings.openai_guidelines_enabled).lower(),
              "OPENAI_MAX_OUTPUT_TOKENS": str(settings.openai_max_output_tokens)}
    if mode in {"openai", "combined"}:
        values["OPENAI_MODEL"] = settings.openai_model
        if settings.openai_reasoning_effort is not None:
            values["OPENAI_REASONING_EFFORT"] = settings.openai_reasoning_effort
    if settings.parse_confidence_threshold is not None:
        values["PARSE_CONFIDENCE_THRESHOLD"] = str(settings.parse_confidence_threshold)
    return values


def mount_arguments(settings, mode):
    if mode not in {"local", "combined"}:
        return [], {}
    encoder = settings.e5_model_path.resolve()
    if not encoder.is_dir():
        raise ValueError("Local/combined Docker evaluation needs a provisioned E5_MODEL_PATH")
    source, target = encoder, "/models/e5"
    if encoder.parent.name == "snapshots" and (encoder.parent.parent / "blobs").is_dir():
        source = encoder.parent.parent
        target += "/snapshots/" + encoder.name
    mounts = ["--mount", f"type=bind,source={source},target=/models/e5,readonly"]
    paths = {"E5_MODEL_PATH": target}
    for setting, default, destination in [
        ("classifier_artifact_path", ROOT / "src/artifacts/classifier.joblib", "/app/src/artifacts/classifier.joblib"),
        ("confidence_artifact_path", ROOT / "src/artifacts/confidence.joblib", "/app/src/artifacts/confidence.joblib"),
    ]:
        value = getattr(settings, setting).resolve()
        if value != default.resolve():
            if not value.is_file():
                raise ValueError(f"Custom {setting} must point to a provisioned file")
            mounts += ["--mount", f"type=bind,source={value},target={destination},readonly"]
        paths[setting.upper()] = destination
    if any("," in str(path) for path in [source, settings.classifier_artifact_path, settings.confidence_artifact_path]):
        raise ValueError("Docker mount paths cannot contain commas")
    return mounts, paths


async def wait_ready(url, container, timeout):
    deadline = perf_counter() + timeout
    async with httpx.AsyncClient(base_url=url, timeout=2, trust_env=False) as client:
        while perf_counter() < deadline:
            try:
                response = await client.get("/health")
                if response.status_code == 200 and response.json() == {"status": "ok"}:
                    return
            except (httpx.RequestError, ValueError):
                pass
            if await docker("inspect", "--format", "{{.State.Running}}", container) != "true":
                raise RuntimeError("Evaluation container exited before readiness; inspect its Docker logs")
            await asyncio.sleep(.5)
    raise RuntimeError("Evaluation API did not become healthy before the startup timeout")


async def run_docker_evaluation(*, mode="openai", split="test", settings=None, limit=None,
                                output=None, allow_paid_calls=False, startup_timeout=180, cache_size=0, concurrency=1,
                                http_evaluator=None):
    if mode not in {"stub", "local", "openai", "combined"} or split not in {"val", "test"}:
        raise ValueError("invalid Docker evaluation mode or split")
    if mode in {"openai", "combined"} and not allow_paid_calls:
        raise ValueError("OpenAI evaluation requires explicit paid-call opt-in")
    if startup_timeout <= 0 or cache_size < 0 or concurrency < 1 or (limit is not None and limit < 1):
        raise ValueError("invalid startup timeout, cache size, or limit")
    settings = settings or Settings.from_env()
    settings = Settings.model_validate({**settings.model_dump(), "parser_mode": mode})
    mounts, paths = mount_arguments(settings, mode)
    runtime = {**runtime_configuration(settings, mode, cache_size), **paths}
    run_id = uuid4().hex[:12]
    image, container = f"yad2-eval:{run_id}", f"yad2-eval-{run_id}"
    print(f"Building {image} for parser mode {mode}", flush=True)
    await docker("build", "--tag", image, ".")
    image_id = await docker("image", "inspect", "--format", "{{.Id}}", image)
    # Secrets go through inherited environment by variable name, never command text.
    environment = os.environ.copy()
    environment.update(runtime)
    if mode in {"openai", "combined"}:
        environment["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()
    env_names = list(runtime)
    if mode in {"openai", "combined"}:
        env_names.append("OPENAI_API_KEY")
        env_names.extend(name for name in ("OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID",
                                          "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY") if name in environment)
    arguments = ["run", "--detach", "--name", container, "--publish", "127.0.0.1::8000"]
    for name in env_names:
        arguments += ["--env", name]
    arguments += mounts + [image]
    created = False
    try:
        await docker(*arguments, env=environment)
        created = True
        binding = await docker("port", container, "8000/tcp")
        url = "http://" + binding.splitlines()[0]
        await wait_ready(url, container, startup_timeout)
        print(f"Container ready: {url}; cache capacity={cache_size}", flush=True)
        evaluator = http_evaluator or run_api_evaluation
        return await evaluator(url=url, split=split, settings=settings, limit=limit, output=output, concurrency=concurrency,
                                        deployment={"image": image, "image_id": image_id, "container": container,
                                                    "runtime": runtime, "cache_capacity": cache_size,
                                                    "transport": "HTTP POST /parse to Docker published localhost port"})
    finally:
        if created:
            # Remove only this run's uniquely named container, including on failure.
            await docker("rm", "--force", container)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["stub", "local", "openai", "combined"], default="openai")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--model")
    parser.add_argument("--few-shot-count", type=int, choices=[0, 6, 12])
    parser.add_argument("--guidelines", choices=["on", "off"], help="Override the concise extraction guidelines")
    parser.add_argument("--reasoning-effort", help="Effort, or 'default' to omit the parameter")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--cache-size", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--allow-paid-calls", action="store_true")
    arguments = vars(parser.parse_args())
    settings = Settings.from_env()
    model, effort = arguments.pop("model"), arguments.pop("reasoning_effort")
    count = arguments.pop("few_shot_count")
    guidelines = arguments.pop("guidelines")
    overrides = {}
    if model is not None:
        overrides["openai_model"] = model
    if count is not None:
        overrides["openai_few_shot_count"] = count
    if guidelines is not None:
        overrides["openai_guidelines_enabled"] = guidelines == "on"
    if effort is not None:
        overrides["openai_reasoning_effort"] = None if effort == "default" else effort
    settings = Settings.model_validate({**settings.model_dump(), **overrides})
    directory, summary = asyncio.run(run_docker_evaluation(settings=settings, **arguments))
    print(f"HTML report: {directory / 'REPORT.html'}", flush=True)
    return int(bool(summary["overall"]["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
