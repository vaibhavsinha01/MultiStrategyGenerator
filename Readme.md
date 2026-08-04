# MultiStrategyGenerator

MultiStrategyGenerator creates, evaluates, documents, and exports algorithmic trading strategies. The FastAPI dashboard is authenticated and supports both generated strategies and saved user strategies.

---

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy the environment file and fill in credentials:

```bash
cp .env.example .env
```

Set in `.env`:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/multi_strategy_generator
SECRET_KEY=your-secret-key
# + OAuth / payment credentials
# + market-data credentials
```

Keep secrets out of source control — `.env.example` only contains placeholders.

---

## 2. Run

Start PostgreSQL, then run the app:

```bash
python web_dashboard.py
```

or

```bash
uvicorn web_dashboard:app --host 127.0.0.1 --port 10000
```

The first start creates the database tables automatically.

| Service   | URL                                                        |
| --------- | ----------------------------------------------------------- |
| App       | http://127.0.0.1:10000                                      |
| Dashboard | http://127.0.0.1:10000/app                                  |
| Swagger   | http://127.0.0.1:10000/docs                                 |
| Metrics   | http://127.0.0.1:10000/metrics                              |

Sign in at `http://127.0.0.1:10000/app`.

---

## 3. Saved & Public Strategies

Authenticated users can create, list, update, delete, and download only strategies they own. Developer-created public strategies are visible/downloadable to every user but are read-only.

| Endpoint | Description |
|---|---|
| `GET /api/signals` | Signal metadata + default selection (from `signals.py`) |
| `GET /api/strategies` | List strategies |
| `POST /api/strategies` | Create strategy |
| `PUT /api/strategies/{id}` | Update (owner-only) |
| `DELETE /api/strategies/{id}` | Delete (owner-only) |
| `GET /api/strategies/{id}/download?format=code\|pdf` | Download (permitted user/public) |

`GET /api/signals` returns `defaultSelected` containing every available signal ID — clients add/remove IDs without redefining signals.

---

## 4. Strategy Generation

Open `/strategies` in the dashboard to run the generation pipeline for **50, 100, or 500** candidates.

- All signals from `signals.py` are checked by default in a checkbox grid.
- Unchecking a signal removes it from: the generator pool, signal validation, NumPy prefilter, train backtest, regime-aware validation, and evaluator.
- Example: selecting only EMA, MACD, SMC, and Range Filter restricts every generated candidate to that set.
- The generator keeps its 2/3/4/5 signal-count weighting and compatibility/contradiction-group checks (no oversized/contradictory strategies).

**API:**

```bash
POST /api/strategy-generation
{ "count": 100, "signals": [...], "symbol": "ethusd", "timeframe": "15m" }
```

Poll:

```bash
GET /api/strategy-generation/{id}
```

Evaluated results are saved to the requesting user's **My strategies** list.

---

## 5. Manual Walk-Forward Optimization

```bash
POST /api/strategies/{id}/optimize
{ "symbol": "ethusd", "timeframe": "15m" }
```

- Uses Backtesting.py SAMBO optimization (`method="sambo"`) on a 70/30 walk-forward split, optimizing TP/SL.
- Only one optimization job runs at a time — a concurrent request returns **HTTP 409**.

Poll:

```bash
GET /api/optimizations/{id}
```

Returns `running`, `completed`, or `failed` plus the train/test report.

Download the report:

```bash
GET /api/optimizations/{id}/document
```

Optimization is manual only — it never runs automatically as part of generation.

---

## 6. Redis Cache (Optional)

```bash
/usr/bin/redis-server        # WSL
redis-cli ping                # → PONG
```

```env
REDIS_URL=redis://localhost:6379/0
```

The app pings Redis on startup and silently falls back to normal execution if it's unavailable.

Caches:

| Item | TTL |
|---|---|
| Strategy reports | 60 seconds |
| Generated PDFs | 24 hours |
| Gemini LLM narratives | 24 hours |

Cache hits/misses are exported in Prometheus metrics.

---

## 7. Prometheus & Grafana

Metrics are exposed at `http://127.0.0.1:10000/metrics` (request totals/latency, strategies generated, optimization statuses, Redis hits/misses, LLM requests/latency).

**Prometheus:**

```cmd
C:\Prometheus\prometheus-3.13.2.windows-amd64\prometheus.exe --config.file=prometheus.yml
```

`prometheus.yml` scrapes via `host.docker.internal:10000` by default — change the target to `127.0.0.1:10000` if both processes run directly on Windows (not in Docker).

Open: `http://localhost:9090` (targets: `http://localhost:9090/targets`)

**Grafana:**

```cmd
C:\Program Files\GrafanaLabs\grafana\bin\grafana.exe
```

Open `http://localhost:3000` → add Prometheus (`http://localhost:9090`) as a data source → graph `msg_*` metrics.

---

## 8. Docker

```bash
docker build -t multi-strategy-generator .
```

Exposes port `10000`, starts Uvicorn. `.dockerignore` excludes data, results, local logs, secrets, notebooks, and generated PDFs while keeping build files.

---

## Startup Order

```
PostgreSQL → Redis (optional) → venv → App → Prometheus → Grafana
```

## Shutdown Order

```
Grafana → Prometheus → App → Redis → PostgreSQL
```