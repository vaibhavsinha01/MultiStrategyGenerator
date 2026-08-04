"""Application metrics.  Imports remain safe when Prometheus is disabled."""
from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram, make_asgi_app
    REQUESTS = Counter("msg_http_requests_total", "HTTP requests", ["method", "path", "status"])
    REQUEST_LATENCY = Histogram("msg_http_request_duration_seconds", "HTTP request latency", ["method", "path"])
    STRATEGIES_GENERATED = Counter("msg_strategies_generated_total", "User strategies generated")
    OPTIMIZATIONS = Counter("msg_strategy_optimizations_total", "Strategy optimizations", ["status"])
    CACHE = Counter("msg_redis_cache_total", "Redis cache lookups", ["result", "kind"])
    LLM_REQUESTS = Counter("msg_llm_requests_total", "LLM requests")
    LLM_LATENCY = Histogram("msg_llm_response_duration_seconds", "LLM response time")
    prometheus_app = make_asgi_app()
except ImportError:  # pragma: no cover
    class _Noop:
        def labels(self, *args, **kwargs): return self
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    REQUESTS = REQUEST_LATENCY = STRATEGIES_GENERATED = OPTIMIZATIONS = CACHE = LLM_REQUESTS = LLM_LATENCY = _Noop()
    prometheus_app = None
