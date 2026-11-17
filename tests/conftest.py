import pytest


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch):
    """Tests must not depend on a developer's local .env (environment variables take precedence over .env).

    Default to the stub parser; tests that exercise the local parser pass explicit Settings or set the variable.
    """
    monkeypatch.setenv("EXTRACTOR_ENABLED", "false")
    monkeypatch.setenv("OPENAI_FALLBACK_ENABLED", "false")
