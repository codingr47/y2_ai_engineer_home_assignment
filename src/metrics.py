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
