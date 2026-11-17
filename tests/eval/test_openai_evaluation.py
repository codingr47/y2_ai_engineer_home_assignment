"""Paid model comparisons through the evaluation suite; normal tests skip these."""
import asyncio
import json
import os

import pytest

from scripts.evaluate_docker import run_docker_evaluation
from src.config.settings import Settings

pytestmark = pytest.mark.skipif(os.environ.get("RUN_OPENAI_EVAL") != "1",
                                reason="set RUN_OPENAI_EVAL=1 to authorize paid evaluation")


@pytest.mark.parametrize("configuration", json.loads(os.environ.get("EVAL_CONFIGURATIONS", "[{}]")))
def test_model_configuration(configuration):
    settings = Settings.from_env()
    settings = Settings.model_validate({**settings.model_dump(), **configuration})
    _, summary = asyncio.run(run_docker_evaluation(mode=os.environ.get("EVAL_MODE", "openai"),
                                            split=os.environ.get("EVAL_SPLIT", "test"), settings=settings,
                                            limit=int(os.environ["EVAL_LIMIT"]) if os.environ.get("EVAL_LIMIT") else None,
                                            concurrency=int(os.environ.get("EVAL_CONCURRENCY", "1")),
                                            allow_paid_calls=True))
    assert not summary["overall"]["failures"], summary["overall"]["failures"]
