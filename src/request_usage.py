"""Request-scoped provider usage shared with HTTP middleware across child tasks."""
from contextvars import ContextVar

_usage = ContextVar("openai_request_usage", default=None)


def begin():
    holder = {"provider_called": False, "tokens": None}
    return holder, _usage.set(holder)


def reset(token):
    _usage.reset(token)


def provider_started():
    holder = _usage.get()
    if holder is not None:
        holder["provider_called"] = True


def record_tokens(tokens):
    holder = _usage.get()
    if holder is not None:
        holder["tokens"] = dict(tokens)
