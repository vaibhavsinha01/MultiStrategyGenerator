# MultiStrategyGenerator

MultiStrategyGenerator creates, evaluates, documents, and exports algorithmic trading strategies. The FastAPI dashboard is authenticated and supports both generated CSV report strategies and saved user strategies.

## Run locally

1. Create a virtual environment and install dependencies: `pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` and set `DATABASE_URL`, `SECRET_KEY`, OAuth/payment credentials, and any market-data credentials required by your workflow.
3. Start PostgreSQL, then run `python web_dashboard.py` (or `uvicorn web_dashboard:app --host 127.0.0.1 --port 10000`).
4. Sign in at `http://127.0.0.1:10000/app`.

The first application start creates the database tables. Keep secrets out of source control; `.env.example` contains placeholders only.

## Saved and public strategies

Authenticated users can create, list, update, delete, and download only strategies they own. Developer-created public strategies are visible and downloadable to every authenticated user but are read-only. `GET /api/signals` returns the registry from `signals.py`; its `defaultSelected` list contains every available signal, so clients can remove or add IDs without copying signal definitions.

Saved-strategy API:

- `GET /api/signals` — signal metadata and default selection.
- `GET` / `POST /api/strategies` — list/create strategies.
- `PUT` / `DELETE /api/strategies/{id}` — owner-only update/delete.
- `GET /api/strategies/{id}/download?format=code|pdf` — permitted user/public download.

The dashboard includes a saved-strategy panel with create, download, delete, and optimize actions. Integrations can use the APIs to provide a richer signal-picker UI.

## Strategy Factory and signal selection

Open `/strategies` from the dashboard to run the full existing generation pipeline for 50, 100, or 500 candidates. The page lists every signal from `signals.py` in a checkbox grid; all are checked initially. Unchecking signals removes them from the generator pool, signal validation, NumPy prefilter, train backtest, regime-aware validation, and evaluator run. For example, selecting only EMA, MACD, SMC, and Range Filter signals means every generated candidate is composed solely from that selection.

The generator retains its current 2/3/4/5 signal-count weights and compatibility/contradiction-group checks, so it never creates a 174-indicator strategy. Start a job through `POST /api/strategy-generation` with `count` (50, 100, or 500), `signals`, `symbol`, and `timeframe`; poll `GET /api/strategy-generation/{id}`. Evaluated results are saved to the requesting user's **My strategies** list.

## Manual walk-forward optimization

Start a job with `POST /api/strategies/{id}/optimize` and JSON such as `{"symbol":"ethusd","timeframe":"15m"}`. It uses Backtesting.py SAMBO optimization (`method="sambo"`) on a 70/30 walk-forward split and optimizes TP/SL. Only one job runs at once; a concurrent request returns HTTP 409. Poll `GET /api/optimizations/{id}` for `running`, `completed`, or `failed` plus the train/test report, then download the report through `GET /api/optimizations/{id}/document`. Optimization is manual and never runs as part of generation.

## Redis cache

Redis is optional. Start it in WSL with `/usr/bin/redis-server` and set `REDIS_URL=redis://localhost:6379/0`. The app pings Redis during startup and silently falls back to normal generation if unavailable. It caches strategy reports (60 seconds), generated PDFs (24 hours), and Gemini LLM narratives (24 hours). Report/PDF/LLM cache hits and misses are included in Prometheus metrics.

## Prometheus and Grafana

The dashboard exposes Prometheus-format metrics at `http://127.0.0.1:10000/metrics`: request totals/latency, strategies generated, optimization statuses, Redis cache hits/misses, and LLM requests/latency. Run Prometheus with:

`C:\Prometheus\prometheus-3.13.2.windows-amd64\prometheus.exe --config.file=prometheus.yml`

`prometheus.yml` scrapes the local application through `host.docker.internal:10000`; change the target to `127.0.0.1:10000` if both processes run directly on Windows. Start Grafana with `C:\Program Files\GrafanaLabs\grafana\bin\grafana.exe`, add Prometheus (`http://localhost:9090`) as a data source, and graph the `msg_*` metrics.

## Docker and CI

`docker build -t multi-strategy-generator .` builds the dashboard image, which exposes port 10000 and starts Uvicorn. `.dockerignore` excludes data, results, local logs, secrets, notebooks, and generated PDFs while retaining build files. GitHub Actions installs the project and checks that all Python files compile.

## Testing

- Run `python -m compileall -q .` for a fast syntax check.
- Sign in as two users; create a strategy as user A, then verify user B cannot update/delete it but can read/download a developer public strategy.
- Call `/api/signals`, remove signal IDs from `defaultSelected`, create with the resulting list, and retrieve it to verify the exact selection persists.
- Start one optimization and immediately start another; expect 202 then 409. Poll its status until complete.
- Stop Redis and request a report/PDF: it should still succeed. Start Redis and repeat requests, then inspect `msg_redis_cache_total` at `/metrics`.
- Start Prometheus and Grafana, issue dashboard/API traffic, and confirm the `msg_http_requests_total` and latency series appear.
