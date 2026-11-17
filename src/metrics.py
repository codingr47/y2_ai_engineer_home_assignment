"""A dedicated Prometheus registry per application instance."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "http_requests_total", "Completed HTTP requests", ["route", "method", "status"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "http_request_duration_seconds", "HTTP response latency", ["route", "method"],
            buckets=(.005, .01, .025, .05, .1, .15, .3, .6, 1, 2, 5), registry=self.registry,
        )
        self.parses = Counter(
            "parse_requests_total", "Successful parses", ["category"], registry=self.registry,
        )
        self.hits = Counter("parse_cache_hits_total", "Cache hits", registry=self.registry)
        self.misses = Counter("parse_cache_misses_total", "Cache misses", registry=self.registry)
        self.evictions = Counter("parse_cache_evictions_total", "LRU evictions", registry=self.registry)
        self.entries = Gauge("parse_cache_entries", "Current cache size", registry=self.registry)
        self.routing = Counter("parse_routing_total", "Parser routing decisions on cache misses",
                               ["destination", "reason"], registry=self.registry)
        self.confidence = Histogram("parse_confidence", "Local whole-parse confidence",
                                    buckets=(0, .25, .5, .75, .9, .95, .99, 1), registry=self.registry)
        self.provider_calls = Counter("openai_calls_total", "OpenAI calls by outcome", ["outcome"], registry=self.registry)
        self.provider_duration = Histogram("openai_duration_seconds", "OpenAI call and validation latency",
                                           buckets=(.1, .3, .6, 1, 2, 5, 10, 30, 60), registry=self.registry)
        self.provider_tokens = Counter("openai_tokens_total", "OpenAI token usage", ["kind"], registry=self.registry)
        self.security_rejections = Counter("llm_security_rejections_total", "LLM input guard rejections",
                                           ["reason"], registry=self.registry)
