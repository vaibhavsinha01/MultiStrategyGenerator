"""
web_dashboard.py
────────────────
Local strategy report website.

Run:
    python web_dashboard.py

Then open:
    http://127.0.0.1:10000

Auth has been removed — the dashboard is publicly accessible.
"""

from __future__ import annotations

import math 
import csv
import json
import re
from pathlib import Path
import time
import hashlib
import threading
from datetime import datetime, timezone

from llm_generator import build_document_payload, generate_strategy_document
from tools_llm_strategy import generate_strategy_main_py_source

from auth_db import (
    authenticate_user,
    create_access_token,
    create_or_get_google_user,
    create_user,
    deduct_credits,
    ensure_admin_user,
    get_current_user,
    get_user_credits,
    init_db,
    log_event,
    optional_user,
    record_subscription,
    save_prediction,
    set_user_premium,
    user_has_credits,
    user_has_premium,
)
from google_auth import build_google_auth_url, exchange_code_for_user, new_oauth_state
from payments import capture_order, create_checkout_order
from ml_pipeline import predict_probabilities
from risk_validation import latest_validation_summary
from cache import cache
from metrics import CACHE, OPTIMIZATIONS, REQUESTS, REQUEST_LATENCY, STRATEGIES_GENERATED, prometheus_app
from optimization import optimize_walk_forward, release, try_acquire
from signals import SIGNALS
from strategy_store import (create_optimization, create_strategy, delete_strategy,
                            finish_optimization, get_optimization, get_strategy,
                            init_strategy_store, list_strategies, update_strategy,
                            create_generation_job, finish_generation_job, get_generation_job)

from fastapi import Depends, FastAPI, Form, HTTPException, Request
# from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    PlainTextResponse,
)


ROOT          = Path(__file__).resolve().parent
DATA_DIR      = ROOT / "data"
RESULTS_DIR   = ROOT / "results"
DASHBOARD_DIR = ROOT / "dashboard"
IMAGES_DIR    = ROOT / "images"
PORT          = 10000
# HOST          = "0.0.0.0"
HOST          = "127.0.0.1"

# ── credit costs ───────────────────────────────────────────────────────────────
CREDIT_COST_CODE     = 4   # /api/strategy-code
CREDIT_COST_DOCUMENT = 2   # /api/document
_generation_lock = threading.Lock()

DATA_RE = re.compile(r"(?P<symbol>[a-z]+usd[tc]?)_(?P<timeframe>[^.]+)\.csv$", re.IGNORECASE)
REGIMES = ("chop", "trendy", "volatile")
HIDDEN_TIMEFRAMES = {"1m", "5m"}

# ── normalisation helpers ─────────────────────────────────────────────────────

def _normalize_symbol(value: str) -> str:
    value = value.lower().strip()
    if value == "eth": return "ethusd"
    if value == "btc": return "btcusd"
    return value


def _display_timeframe(value: str) -> str:
    value = value.strip()
    return "1D" if value.upper() in {"D", "1D"} else value


def _file_timeframe(value: str) -> str:
    value = value.strip()
    return "D" if value.upper() == "1D" else value


def _json_safe(value):
    """Recursively replace NaN/Infinity floats with 0.0.

    Postgres's json/jsonb columns reject the NaN/Infinity tokens that
    json.dumps emits by default for these values, even though they're
    valid Python floats (e.g. from a Sharpe ratio divide-by-zero when a
    walk-forward split has too few trades to have return variance).
    """
    if isinstance(value, float):
        return 0.0 if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


# ── HTTP response helpers ─────────────────────────────────────────────────────


# ── HTTP response helpers ─────────────────────────────────────────────────────

def _json(payload: dict, status: int = 200) -> JSONResponse:
    return JSONResponse(content=payload, status_code=status)


# ── scan / file helpers ───────────────────────────────────────────────────────

def _scan_options() -> dict:
    symbols: set[str] = set()
    timeframes_by_symbol: dict[str, set[str]] = {}

    unified = RESULTS_DIR / "strategy_results_unified.csv"
    if unified.exists():
        try:
            with unified.open("r", newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    if row.get("result_type") not in {"train_top500", "validated"}:
                        continue
                    symbol = _normalize_symbol(row.get("symbol", ""))
                    timeframe = _display_timeframe(row.get("timeframe", ""))
                    if not symbol or not timeframe or timeframe.lower() in HIDDEN_TIMEFRAMES:
                        continue
                    symbols.add(symbol)
                    timeframes_by_symbol.setdefault(symbol, set()).add(timeframe)
        except Exception:
            pass

    return {
        "symbols": sorted(symbols),
        "timeframesBySymbol": {
            sym: sorted(vals, key=_timeframe_sort_key)
            for sym, vals in timeframes_by_symbol.items()
        },
        "reports": {},  # no longer needed
    }


def _parse_result_name(filename: str) -> tuple[str, str, str] | None:
    if not filename.lower().startswith("strategy_results_") or not filename.lower().endswith(".csv"):
        return None
    stem = filename[:-4]
    kind = ""
    if stem.lower().endswith("_validated"):
        kind = "validated"; stem = stem[:-10]
    elif stem.lower().endswith("_train_top500"):
        kind = "train"; stem = stem[:-13]
    else:
        return None
    parts = stem[len("strategy_results_"):].split("_")
    if len(parts) < 2:
        return None
    first, second = parts[0], parts[1]
    if first[-1:].lower() in {"m", "h", "d"}:
        return _normalize_symbol(second), _display_timeframe(first), kind
    return _normalize_symbol(first), _display_timeframe(second), kind


def _timeframe_sort_key(value: str) -> tuple[int, int | str]:
    suffix = value[-1].lower()
    number = value[:-1]
    if value.upper() == "1D":    return (4, 1)
    if suffix == "m" and number.isdigit(): return (1, int(number))
    if suffix == "h" and number.isdigit(): return (2, int(number))
    return (9, value)


def _candidate_report(symbol: str, timeframe: str, dataset: str) -> Path | None:
    unified = RESULTS_DIR / "strategy_results_unified.csv"
    return unified if unified.exists() else None

def _num(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        return float(value) if value not in ("", None) else default
    except (TypeError, ValueError):
        return default


def _display_sharpe(row: dict, prefix: str) -> float:
    sharpe = _num(row, f"{prefix}_sharpe", _num(row, f"{prefix}_sharp"))
    if abs(sharpe) > 1e-12:
        return sharpe
    ret = _num(row, f"{prefix}_return")
    dd = abs(_num(row, f"{prefix}_drawdown"))
    trades = max(_num(row, f"{prefix}_trades"), 1.0)
    if dd > 0 and abs(ret) > 0:
        return round((ret / dd) * min((trades ** 0.5) / 4.0, 2.0), 4)
    return 0.0


# ── strategy payload helpers ──────────────────────────────────────────────────

def _strategy_payload(row: dict, rank: int) -> dict:
    regimes = {}
    for split in ("train", "test"):
        regimes[split] = {}
        for regime in REGIMES:
            prefix = f"{split}_{regime}"
            regimes[split][regime] = {
                "trades":              int(_num(row, f"{prefix}_trades")),
                "share":               _num(row, f"{prefix}_trade_share"),
                "returnPct":           _num(row, f"{prefix}_return_pct"),
                "winRate":             _num(row, f"{prefix}_win_rate"),
                "avgTradeReturnPct":   _num(row, f"{prefix}_avg_trade_return_pct"),
                "pnl":                 _num(row, f"{prefix}_pnl"),
            }
    return {
        "rank":       rank,
        "id":         row.get("id", ""),
        "direction":  row.get("direction", ""),
        "signals":    row.get("signals", ""),
        "nSignals":   int(_num(row, "n_signals")),
        "tp":         _num(row, "tp"),
        "sl":         _num(row, "sl"),
        "score":      _num(row, "test_score", _num(row, "score")),
        "train": {
            "returnPct":  _num(row, "train_return"),
            "sharpe":     _display_sharpe(row, "train"),
            "drawdown":   _num(row, "train_drawdown"),
            "trades":     int(_num(row, "train_trades")),
            "winRate":    _num(row, "train_winrate"),
        },
        "test": {
            "returnPct":  _num(row, "test_return"),
            "sharpe":     _display_sharpe(row, "test"),
            "drawdown":   _num(row, "test_drawdown"),
            "trades":     int(_num(row, "test_trades")),
            "winRate":    _num(row, "test_winrate"),
        },
        "regimes": regimes,
    }


def _report(symbol: str, timeframe: str, dataset: str, limit: int) -> dict:
    cache_key = f"report:{symbol}:{timeframe}:{dataset}:{limit}"
    cached = cache.get_json(cache_key)
    if cached:
        CACHE.labels("hit", "report").inc()
        return cached
    CACHE.labels("miss", "report").inc()
    report_path = _candidate_report(symbol, timeframe, dataset)
    if report_path is None:
        return {
            "available":  False,
            "symbol":     symbol,
            "timeframe":  timeframe,
            "dataset":    dataset,
            "runCommand": (
                f"python -X utf8 main.py --csv data\\{symbol}_{_file_timeframe(timeframe)}.csv "
                f"--n 10000 --top 500 --workers 8 "
                f"--out results\\strategy_results_{symbol}_{timeframe.lower()}_10k.csv"
            ),
        }

    with report_path.open("r", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    rows = _filter_result_rows(rows, symbol, timeframe, dataset)
    if not rows:
        return {
            "available": False,
            "symbol": symbol,
            "timeframe": timeframe,
            "dataset": dataset,
            "runCommand": (
                f"python -X utf8 main.py --csv data\\{symbol}_{_file_timeframe(timeframe)}.csv "
                f"--n 10000 --top 500 --workers 8 "
                f"--out results\\strategy_results_unified.csv"
            ),
        }

    score_key = "robust_score" if rows and "robust_score" in rows[0] else (
        "test_score" if dataset == "validated" and rows and "test_score" in rows[0] else "score"
    )
    rows.sort(key=lambda r: _num(r, score_key), reverse=True)
    top_rows = rows[:limit]

    result = {
        "available":        True,
        "symbol":           symbol,
        "timeframe":        timeframe,
        "dataset":          dataset,
        "sourceFile":       report_path.name,
        "totalStrategies":  len(rows),
        "top": [_strategy_payload(r, idx + 1) for idx, r in enumerate(top_rows)],
    }
    cache.set_json(cache_key, result, 60)
    return result


def _load_ranked_rows(symbol: str, timeframe: str, dataset: str) -> tuple[Path | None, list[dict]]:
    report_path = _candidate_report(symbol, timeframe, dataset)
    if report_path is None:
        return None, []
    with report_path.open("r", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    rows = _filter_result_rows(rows, symbol, timeframe, dataset)
    score_key = "robust_score" if rows and "robust_score" in rows[0] else (
        "test_score" if dataset == "validated" and rows and "test_score" in rows[0] else "score"
    )
    rows.sort(key=lambda r: _num(r, score_key), reverse=True)
    return report_path, rows


def _filter_result_rows(rows: list[dict], symbol: str, timeframe: str, dataset: str) -> list[dict]:
    if not rows or not any("result_type" in row for row in rows):
        return rows
    wanted_type = "validated" if dataset == "validated" else "train_top500"
    wanted_symbol = _normalize_symbol(symbol)
    wanted_timeframe = _display_timeframe(timeframe).lower()
    filtered = []
    for row in rows:
        if row.get("result_type") != wanted_type:
            continue
        row_symbol = _normalize_symbol(row.get("symbol", ""))
        row_timeframe = _display_timeframe(row.get("timeframe", "")).lower()
        if row_symbol == wanted_symbol and row_timeframe == wanted_timeframe:
            filtered.append(row)
    return filtered


def _strategy_pdf(symbol: str, timeframe: str, dataset: str, strategy_id: str) -> tuple[str, bytes]:
    cache_key = f"pdf:public:{symbol}:{timeframe}:{dataset}:{strategy_id}"
    cached = cache.get_bytes(cache_key)
    if cached:
        CACHE.labels("hit", "pdf").inc()
        return f"strategy_{symbol}_{timeframe}_{strategy_id}.pdf", cached
    CACHE.labels("miss", "pdf").inc()
    report_path, rows = _load_ranked_rows(symbol, timeframe, dataset)

    if report_path is None:
        from llm_generator import _fallback_text, build_pdf
        stub_payload = {
            "symbol":    symbol.upper(),
            "timeframe": timeframe.upper(),
            "dataset":   dataset,
            "sourceFile": "N/A",
            "strategy": {
                "id": strategy_id, "direction": "unknown",
                "signals": "", "nSignals": 0, "tp": 0, "sl": 0, "score": 0,
                "train": {"returnPct":0,"sharpe":0,"drawdown":0,"trades":0,"winRate":0},
                "test":  {"returnPct":0,"sharpe":0,"drawdown":0,"trades":0,"winRate":0},
                "regimes": {"train":{r:{} for r in REGIMES}, "test":{r:{} for r in REGIMES}},
                "signalDetails": [],
                "parameters": {
                    "direction":"unknown","numberOfSignals":0,"takeProfitPct":0,
                    "stopLossPct":0,"entryThreshold":"","buyTrade":"N/A",
                    "sellTrade":"N/A","exitRules":[]
                },
                "marketConditionSuggestion": {
                    "bestCondition":"unknown","weakestCondition":"unknown","suggestion":"No data."
                },
            },
            "stockUniverse": {"executionUniverse":symbol.upper(),"timeframe":timeframe.upper(),"dataSource":"N/A"},
            "backtestContext": {"initialCash":0,"commission":"N/A","trainTestSplit":"N/A","regimeModel":"N/A","tradeAttribution":"N/A"},
        }
        text = _fallback_text(stub_payload, "No results file found for this symbol/timeframe/dataset.")
        pdf  = build_pdf(stub_payload, text)
        return f"strategy_{strategy_id}_notfound.pdf", pdf

    selected = None
    selected_rank = 0
    for idx, row in enumerate(rows, start=1):
        if row.get("id") == strategy_id:
            selected      = row
            selected_rank = idx
            break

    if selected is None:
        from llm_generator import _fallback_text, build_pdf
        stub_payload = {
            "symbol":    symbol.upper(),
            "timeframe": timeframe.upper(),
            "dataset":   dataset,
            "sourceFile": report_path.name,
            "strategy": {
                "id": strategy_id, "direction": "unknown",
                "signals": "", "nSignals": 0, "tp": 0, "sl": 0, "score": 0,
                "train": {"returnPct":0,"sharpe":0,"drawdown":0,"trades":0,"winRate":0},
                "test":  {"returnPct":0,"sharpe":0,"drawdown":0,"trades":0,"winRate":0},
                "regimes": {"train":{r:{} for r in REGIMES}, "test":{r:{} for r in REGIMES}},
                "signalDetails": [],
                "parameters": {
                    "direction":"unknown","numberOfSignals":0,"takeProfitPct":0,
                    "stopLossPct":0,"entryThreshold":"","buyTrade":"N/A",
                    "sellTrade":"N/A","exitRules":[]
                },
                "marketConditionSuggestion": {
                    "bestCondition":"unknown","weakestCondition":"unknown","suggestion":"No data."
                },
            },
            "stockUniverse": {"executionUniverse":symbol.upper(),"timeframe":timeframe.upper(),"dataSource":"N/A"},
            "backtestContext": {"initialCash":0,"commission":"N/A","trainTestSplit":"N/A","regimeModel":"N/A","tradeAttribution":"N/A"},
        }
        text = _fallback_text(stub_payload, f"Strategy id {strategy_id} not found in {report_path.name}.")
        pdf  = build_pdf(stub_payload, text)
        return f"strategy_{strategy_id}_notfound.pdf", pdf

    strategy  = _strategy_payload(selected, selected_rank)
    payload   = build_document_payload(symbol, timeframe, dataset, report_path.name, strategy)
    pdf_bytes = generate_strategy_document(payload)
    filename  = f"strategy_{payload['symbol'].lower()}_{payload['timeframe'].lower()}_{strategy_id}.pdf"
    cache.set_bytes(cache_key, pdf_bytes, 86400)
    return filename, pdf_bytes


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="MultiStrategyGenerator Dashboard")

if prometheus_app is not None:
    app.mount("/metrics", prometheus_app)

@app.middleware("http")
async def prometheus_metrics(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    path = request.url.path if request.url.path != "/metrics" else "/metrics"
    REQUESTS.labels(request.method, path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, path).observe(time.perf_counter() - started)
    return response

@app.on_event("startup")
def _startup() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    DASHBOARD_DIR.mkdir(exist_ok=True)
    try:
        init_db()
        init_strategy_store()
        ensure_admin_user()
        print("PostgreSQL init OK")
    except Exception as exc:
        print(f"PostgreSQL init FAILED: {exc}")


def _request_base(request: Request) -> str:
    import os
    base = os.environ.get("APP_BASE_URL", "").strip()
    if base:
        return base.rstrip("/")
    return str(request.base_url).rstrip("/")


def _auth_cookie_response(user: dict, redirect_to: str = "/") -> RedirectResponse:
    token = create_access_token({"sub": str(user["id"])})
    response = RedirectResponse(redirect_to, status_code=303)
    response.set_cookie(
        "access_token",
        token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 2,
    )
    return response

# ── pages ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    return FileResponse(DASHBOARD_DIR / "index.html", media_type="text/html")


@app.get("/index.html", response_class=HTMLResponse)
def index_html(request: Request) -> Response:
    return FileResponse(DASHBOARD_DIR / "index.html", media_type="text/html")


@app.get("/app", response_class=HTMLResponse)
def app_page(user: dict = Depends(get_current_user)) -> Response:
    return FileResponse(DASHBOARD_DIR / "app.html", media_type="text/html")


@app.get("/strategies", response_class=HTMLResponse)
def strategy_builder_page(user: dict = Depends(get_current_user)) -> Response:
    return FileResponse(DASHBOARD_DIR / "strategy_builder.html", media_type="text/html")

@app.get("/document", response_class=HTMLResponse)
def document_page(user: dict = Depends(get_current_user)) -> Response:
    return FileResponse(DASHBOARD_DIR / "document.html", media_type="text/html")

@app.get("/services", response_class=HTMLResponse)
def services_page(user: dict = Depends(get_current_user)) -> Response:
    return FileResponse(DASHBOARD_DIR / "services.html", media_type="text/html")

@app.get("/images/{path:path}")
def image_files(path: str) -> Response:
    file_path = (IMAGES_DIR / path).resolve()
    if IMAGES_DIR not in file_path.parents and file_path != IMAGES_DIR:
        return Response(status_code=404)
    if not file_path.exists() or not file_path.is_file():
        return Response(status_code=404)
    return FileResponse(file_path)


@app.get("/login", response_class=HTMLResponse)
def login_page() -> Response:
    return FileResponse(DASHBOARD_DIR / "login.html", media_type="text/html")


@app.get("/signup", response_class=HTMLResponse)
def signup_page() -> Response:
    return FileResponse(DASHBOARD_DIR / "signup.html", media_type="text/html")

@app.post("/auth/signup")
def auth_signup(
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
) -> Response:

    if password != confirm_password:
        return RedirectResponse(
            "/signup?error=password_mismatch",
            status_code=303
        )

    try:
        user = create_user(email, full_name, password)

        print(f"USER CREATED: {email}")

    except Exception as e:

        print(f"SIGNUP ERROR: {repr(e)}")

        return PlainTextResponse(
            f"Signup failed: {repr(e)}",
            status_code=500
        )

    token = create_access_token({"sub": str(user["id"])})

    response = RedirectResponse("/app", status_code=303)

    response.set_cookie(
        "access_token",
        token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 2,
    )

    log_event("signup", "User created", user_id=user["id"])

    return response


@app.post("/auth/login")
def auth_login(
    email: str = Form(...),
    password: str = Form(...),
) -> Response:

    try:
        user = authenticate_user(email, password)

        if not user:
            print(f"LOGIN FAILED: Invalid credentials for {email}")

            return PlainTextResponse(
                "Invalid credentials",
                status_code=401
            )

    except Exception as e:
        print(f"LOGIN ERROR: {repr(e)}")

        return PlainTextResponse(
            f"Login failed: {repr(e)}",
            status_code=500
        )

    token = create_access_token({"sub": str(user["id"])})

    response = RedirectResponse("/app", status_code=303)

    response.set_cookie(
        "access_token",
        token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 2,
    )

    log_event("login", "User logged in", user_id=user["id"])

    return response


@app.get("/logout")
def logout() -> Response:
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie("access_token")
    return response


@app.get("/api/me")
def api_me(request: Request) -> Response:
    user = optional_user(request)
    if not user:
        return _json({"authenticated": False})
    return _json({
        "authenticated": True,
        "email": user.get("email"),
        "full_name": user.get("full_name"),
        "is_admin": bool(user.get("is_admin")),
        "is_premium": user_has_premium(user),
        "credits": get_user_credits(user),
    })


@app.get("/auth/google")
def auth_google(request: Request) -> Response:
    try:
        state = new_oauth_state()
        url = build_google_auth_url(_request_base(request), state)
    except Exception as exc:
        return RedirectResponse(f"/login?error=google_config&msg={exc}", status_code=303)
    response = RedirectResponse(url, status_code=303)
    response.set_cookie("oauth_state", state, httponly=True, samesite="lax", max_age=600)
    return response


@app.get("/auth/google/callback")
def auth_google_callback(
    request: Request,
    code: str = "",
    state: str = "",
    error: str = "",
) -> Response:
    if error:
        return RedirectResponse("/login?error=google_denied", status_code=303)
    saved_state = request.cookies.get("oauth_state", "")
    if not code or not state or state != saved_state:
        return RedirectResponse("/login?error=google_state", status_code=303)
    try:
        profile = exchange_code_for_user(code, _request_base(request))
        user = create_or_get_google_user(
            email=profile.get("email", ""),
            full_name=profile.get("name", ""),
            google_id=profile.get("sub", ""),
        )
        log_event("login", "Google OAuth login", user_id=user["id"])
        response = _auth_cookie_response(user, "/app")
        response.delete_cookie("oauth_state")
        return response
    except Exception as exc:
        print(f"GOOGLE OAUTH ERROR: {exc}")
        return RedirectResponse("/login?error=google_failed", status_code=303)


@app.post("/api/payments/create")
def api_payment_create(user: dict = Depends(get_current_user)) -> Response:
    if user_has_premium(user):
        return _json({"already_premium": True, "redirect": "/app"})
    try:
        order = create_checkout_order(user["id"])
        approve = next(
            (l.get("href") for l in order.get("links", []) if l.get("rel") == "approve"),
            None,
        )
        if not approve:
            return _json({"error": "PayPal approval link missing"}, status=500)
        return _json({"order_id": order.get("id"), "approval_url": approve})
    except Exception as exc:
        return _json({"error": str(exc)}, status=500)


@app.get("/payment/success")
def payment_success(request: Request, token: str = "") -> Response:
    user = optional_user(request)
    if not user or not token:
        return RedirectResponse("/?payment=failed", status_code=303)
    try:
        result = capture_order(token)
        if result.get("status") == "COMPLETED":
            set_user_premium(user["id"], True)
            amount = 0.0
            currency = "USD"
            units = (result.get("purchase_units") or [{}])[0]
            captures = ((units.get("payments") or {}).get("captures") or [{}])
            if captures:
                amount = float(captures[0].get("amount", {}).get("value", 0) or 0)
                currency = captures[0].get("amount", {}).get("currency_code", "USD")
            record_subscription(user["id"], token, "completed", amount, currency)
            log_event("payment", "PayPal capture success", user_id=user["id"], metadata={"order_id": token})
            return RedirectResponse("/app?payment=success", status_code=303)
    except Exception as exc:
        print(f"PAYMENT CAPTURE ERROR: {exc}")
    return RedirectResponse("/?payment=failed", status_code=303)


@app.get("/payment/cancel")
def payment_cancel() -> Response:
    return RedirectResponse("/?payment=cancelled", status_code=303)


@app.get("/dashboard/{path:path}")
def dashboard_files(path: str) -> Response:
    file_path = (DASHBOARD_DIR / path).resolve()
    if DASHBOARD_DIR not in file_path.parents and file_path != DASHBOARD_DIR:
        return Response(status_code=404)
    if not file_path.exists() or not file_path.is_file():
        return Response(status_code=404)
    return FileResponse(file_path)


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/api/options")
def api_options(user: dict = Depends(get_current_user)) -> Response:
    return _json(_scan_options())


def _saved_strategy_view(row: dict, user_id: int) -> dict:
    strategy = dict(row["strategy"])
    strategy["signals"] = list(strategy.get("signals", []))
    return {
        "id": str(row["id"]), "name": row["name"], "strategy": strategy,
        "isPublic": bool(row["is_public"]), "canEdit": row["owner_id"] == user_id and not row["is_public"],
        "createdAt": row["created_at"].isoformat(), "updatedAt": row["updated_at"].isoformat(),
    }


def _validate_saved_strategy(raw: dict) -> dict:
    signals = raw.get("signals")
    if signals is None:
        signals = list(SIGNALS)  # all signal definitions are selected by default
    if not isinstance(signals, list) or not signals or any(s not in SIGNALS for s in signals):
        raise ValueError("signals must be a non-empty list of signal IDs from /api/signals")
    direction = str(raw.get("direction", "bull")).lower()
    if direction not in {"bull", "bear"}:
        raise ValueError("direction must be bull or bear")
    tp, sl = float(raw.get("tp", 0.02)), float(raw.get("sl", 0.01))
    if not (0 < tp < 1 and 0 < sl < 1):
        raise ValueError("tp and sl must be decimal percentages between 0 and 1")
    return {"id": str(raw.get("id", "saved")), "direction": direction, "signals": signals,
            "n_signals": len(signals), "tp": tp, "sl": sl}


def _document_strategy(strategy: dict) -> dict:
    blank = {"returnPct": 0, "sharpe": 0, "drawdown": 0, "trades": 0, "winRate": 0}
    return {"id": strategy.get("id", "saved"), "direction": strategy["direction"],
            "signals": "|".join(strategy["signals"]), "nSignals": len(strategy["signals"]),
            "tp": strategy["tp"], "sl": strategy["sl"], "score": 0, "train": blank, "test": blank,
            "regimes": {"train": {r: {} for r in REGIMES}, "test": {r: {} for r in REGIMES}}}


def _optimization_view(row: dict) -> dict:
    return {"id": str(row["id"]), "strategyId": str(row["strategy_id"]), "status": row["status"],
            "result": row.get("result"), "error": row.get("error"),
            "createdAt": row["created_at"].isoformat(),
            "completedAt": row["completed_at"].isoformat() if row.get("completed_at") else None}


def _generation_view(row: dict) -> dict:
    return {"id": str(row["id"]), "status": row["status"], "requestedCount": row["requested_count"],
            "selectedSignals": row["selected_signals"], "result": row.get("result"), "error": row.get("error"),
            "createdAt": row["created_at"].isoformat(),
            "completedAt": row["completed_at"].isoformat() if row.get("completed_at") else None}


def _generation_worker(job_id: str, user_id: int, count: int, signals: list[str], symbol: str, timeframe: str) -> None:
    try:
        from main import run
        csv_path = DATA_DIR / f"{symbol.lower()}_{_file_timeframe(timeframe)}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path.name}")
        generated = run(str(csv_path), n_strategies=count, top_n=count, n_workers=1,
                        output_csv=str(RESULTS_DIR / "strategy_results_unified.csv"),
                        append_results=True, allowed_signals=signals)
        saved = []
        for index, item in enumerate(generated or [], start=1):
            strategy = item["strategy"]
            row = create_strategy(user_id, f"Generated {symbol.upper()} {timeframe} #{index}", strategy)
            saved.append(str(row["id"]))
        finish_generation_job(job_id, {"generatedCandidates": count, "savedStrategies": saved, "savedCount": len(saved)})
        STRATEGIES_GENERATED.inc(len(saved))
    except Exception as exc:
        finish_generation_job(job_id, error=str(exc))
    finally:
        _generation_lock.release()


@app.get("/api/signals")
def api_signals(user: dict = Depends(get_current_user)) -> Response:
    return _json({"defaultSelected": list(SIGNALS), "signals": [
        {"id": key, "description": meta.get("desc", key), "group": meta.get("group", ""), "direction": meta.get("dir", "")}
        for key, meta in SIGNALS.items()
    ]})


@app.post("/api/strategy-generation")
async def api_strategy_generation(request: Request, user: dict = Depends(get_current_user)) -> Response:
    body = await request.json()
    acquired = False
    try:
        count = int(body.get("count", 50))
        signals = body.get("signals", list(SIGNALS))
        if count not in {50, 100, 500}:
            raise ValueError("count must be one of 50, 100, or 500")
        if not isinstance(signals, list) or not signals or any(signal not in SIGNALS for signal in signals):
            raise ValueError("Select at least one valid signal")
        acquired = _generation_lock.acquire(blocking=False)
        if not acquired:
            return _json({"error": "A strategy generation job is already running"}, 409)
        job_id = create_generation_job(user["id"], count, signals)
        worker = threading.Thread(target=_generation_worker, args=(job_id, user["id"], count, signals, body.get("symbol", "ethusd"), body.get("timeframe", "15m")), daemon=True)
        worker.start()
        return _json({"id": job_id, "status": "running", "requestedCount": count}, 202)
    except (ValueError, TypeError) as exc:
        if acquired:
            _generation_lock.release()
        return _json({"error": str(exc)}, 400)
    except Exception as exc:
        if acquired:
            _generation_lock.release()
        return _json({"error": f"Could not start generation: {exc}"}, 500)


@app.get("/api/strategy-generation/{job_id}")
def api_strategy_generation_status(job_id: str, user: dict = Depends(get_current_user)) -> Response:
    row = get_generation_job(job_id, user["id"])
    return _json(_generation_view(row)) if row else _json({"error": "Generation job not found"}, 404)


@app.get("/api/strategies")
def api_strategies(user: dict = Depends(get_current_user)) -> Response:
    return _json({"strategies": [_saved_strategy_view(row, user["id"]) for row in list_strategies(user["id"])]})


@app.post("/api/strategies")
async def api_strategy_create(request: Request, user: dict = Depends(get_current_user)) -> Response:
    try:
        body = await request.json()
        strategy = _validate_saved_strategy(body.get("strategy", body))
        row = create_strategy(user["id"], str(body.get("name", "My strategy")).strip() or "My strategy", strategy)
        STRATEGIES_GENERATED.inc()
        return _json(_saved_strategy_view(row, user["id"]), 201)
    except (ValueError, TypeError) as exc:
        return _json({"error": str(exc)}, 400)


@app.put("/api/strategies/{strategy_id}")
async def api_strategy_update(strategy_id: str, request: Request, user: dict = Depends(get_current_user)) -> Response:
    try:
        body = await request.json()
        strategy = _validate_saved_strategy(body.get("strategy", body))
        row = update_strategy(strategy_id, user["id"], str(body.get("name", "My strategy")).strip() or "My strategy", strategy)
        if not row:
            return _json({"error": "Strategy not found or not editable"}, 404)
        return _json(_saved_strategy_view(row, user["id"]))
    except (ValueError, TypeError) as exc:
        return _json({"error": str(exc)}, 400)


@app.delete("/api/strategies/{strategy_id}")
def api_strategy_delete(strategy_id: str, user: dict = Depends(get_current_user)) -> Response:
    return _json({"deleted": True}) if delete_strategy(strategy_id, user["id"]) else _json({"error": "Strategy not found or not deletable"}, 404)


@app.get("/api/strategies/{strategy_id}/download")
def api_saved_strategy_download(strategy_id: str, format: str = "code", user: dict = Depends(get_current_user)) -> Response:
    row = get_strategy(strategy_id, user["id"])
    if not row:
        return _json({"error": "Strategy not found"}, 404)
    strategy = row["strategy"]
    payload = build_document_payload("saved", "custom", "saved", "saved_strategies", _document_strategy(strategy))
    if format == "code":
        source = generate_strategy_main_py_source(payload, use_llm=False)
        return Response(source.encode(), media_type="text/x-python; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{row["name"]}.py"'})
    cache_key = f"pdf:saved:{strategy_id}:{row['updated_at'].isoformat()}"
    pdf = cache.get_bytes(cache_key)
    if pdf:
        CACHE.labels("hit", "pdf").inc()
    else:
        CACHE.labels("miss", "pdf").inc()
        pdf = generate_strategy_document(payload)
        cache.set_bytes(cache_key, pdf, 86400)
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{row["name"]}.pdf"'})


def _optimization_worker(optimization_id: str, strategy: dict, symbol: str, timeframe: str) -> None:
    try:
        # result = optimize_walk_forward(strategy, symbol, timeframe, ROOT)
        # finish_optimization(optimization_id, result=result)
        # OPTIMIZATIONS.labels("completed").inc()
        result = optimize_walk_forward(strategy, symbol, timeframe, ROOT)
        result = _json_safe(result)
        finish_optimization(optimization_id, result=result)
        OPTIMIZATIONS.labels("completed").inc()
    except Exception as exc:
        finish_optimization(optimization_id, error=str(exc))
        OPTIMIZATIONS.labels("failed").inc()
    finally:
        release()


@app.post("/api/strategies/{strategy_id}/optimize")
async def api_strategy_optimize(strategy_id: str, request: Request, user: dict = Depends(get_current_user)) -> Response:
    row = get_strategy(strategy_id, user["id"])
    if not row:
        return _json({"error": "Strategy not found"}, 404)
    if row["owner_id"] != user["id"] or row["is_public"]:
        return _json({"error": "Public strategies are read-only; save your own copy before optimizing"}, 403)
    if not try_acquire():
        return _json({"error": "An optimization job is already running"}, 409)
    try:
        body = await request.json()
        optimization_id = create_optimization(strategy_id, user["id"])
        thread = threading.Thread(target=_optimization_worker, args=(optimization_id, row["strategy"], body.get("symbol", "ethusd"), body.get("timeframe", "15m")), daemon=True)
        thread.start()
        return _json({"id": optimization_id, "status": "running", "method": "sambo"}, 202)
    except Exception:
        release()
        raise


@app.get("/api/optimizations/{optimization_id}")
def api_optimization_status(optimization_id: str, user: dict = Depends(get_current_user)) -> Response:
    row = get_optimization(optimization_id, user["id"])
    return _json(_optimization_view(row)) if row else _json({"error": "Optimization not found"}, 404)


@app.get("/api/optimizations/{optimization_id}/document")
def api_optimization_document(optimization_id: str, user: dict = Depends(get_current_user)) -> Response:
    optimization = get_optimization(optimization_id, user["id"])
    if not optimization:
        return _json({"error": "Optimization not found"}, 404)
    if optimization["status"] != "completed":
        return _json({"error": "Optimization report is not ready"}, 409)
    saved = get_strategy(str(optimization["strategy_id"]), user["id"])
    if not saved:
        return _json({"error": "Strategy not found"}, 404)
    result = optimization["result"] or {}
    document = _document_strategy(saved["strategy"])
    document["train"].update({"returnPct": result.get("train", {}).get("return_pct", 0), "sharpe": result.get("train", {}).get("sharpe", 0), "trades": result.get("train", {}).get("trades", 0)})
    document["test"].update({"returnPct": result.get("test", {}).get("return_pct", 0), "sharpe": result.get("test", {}).get("sharpe", 0), "trades": result.get("test", {}).get("trades", 0)})
    payload = build_document_payload("saved", "walk-forward", "optimization", "sambo", document)
    cache_key = f"pdf:optimization:{optimization_id}"
    pdf = cache.get_bytes(cache_key)
    if pdf:
        CACHE.labels("hit", "pdf").inc()
    else:
        CACHE.labels("miss", "pdf").inc()
        pdf = generate_strategy_document(payload)
        cache.set_bytes(cache_key, pdf, 86400)
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="optimization_{optimization_id}.pdf"'})


@app.get("/api/report")
def api_report(
    symbol: str = "ethusd",
    timeframe: str = "15m",
    dataset: str = "validated",
    limit: int = 10,
    user: dict = Depends(get_current_user),
) -> Response:
    return _json(_report(symbol.lower(), timeframe, dataset.lower(), int(limit)))


@app.get("/api/prediction")
def api_prediction(
    symbol: str = "ethusdt",
    timeframe: str = "15m",
    user: dict = Depends(get_current_user),
) -> Response:
    try:
        payload = predict_probabilities(symbol.lower(), timeframe.lower())
        save_prediction(
            symbol,
            timeframe,
            payload["buy_probability"],
            payload["sell_probability"],
            payload["model_version"],
            payload,
            user_id=user["id"],
        )
        return _json(payload)
    except Exception as exc:
        return _json({"error": f"Prediction failed: {exc}"}, status=500)


@app.post("/api/chatbot")
async def api_chatbot(request: Request, user: dict = Depends(get_current_user)) -> Response:
    body = await request.json()
    message = str(body.get("message", "")).strip()
    lower = message.lower()
    symbol = body.get("symbol", "ethusdt")
    timeframe = body.get("timeframe", "15m")
    try:
        prediction = predict_probabilities(symbol.lower(), timeframe.lower())
    except Exception as exc:
        return _json({"answer": f"I could not load the current probability signal yet: {exc}"}, status=200)
    buy = prediction["buy_probability"]
    sell = prediction["sell_probability"]
    bias = "BUY" if buy >= sell else "SELL"
    report = _report(symbol.lower(), timeframe, "validated", 1)
    risk_note = ""
    if report.get("available") and report.get("top"):
        best = report["top"][0]
        risk_note = (
            f" Best validated strategy has train Sharpe {best['train']['sharpe']:.2f}, "
            f"test Sharpe {best['test']['sharpe']:.2f}, test return {best['test']['returnzPct']:.2f}%, "
            f"and {best['test']['trades']} test trades."
        )
    answer = (
        f"{symbol.upper()} {timeframe.upper()} currently leans {bias}: "
        f"BUY {buy:.2f}% vs SELL {sell:.2f}%. "
        f"The probability source is {prediction.get('source', 'local model/cache')}. "
        f"{risk_note} "
        "Treat this as a probability signal, not a trade instruction. "
        "Robustness comes from checking out-of-sample, cross-symbol, Monte Carlo, and parameter-sensitivity results before sizing risk."
    )
    log_event("chatbot", message[:200], user_id=user["id"], metadata={"symbol": symbol, "timeframe": timeframe})
    return _json({"answer": answer, "prediction": prediction})


@app.get("/api/validation/summary")
def api_validation_summary(user: dict = Depends(get_current_user)) -> Response:
    return _json(latest_validation_summary())


def _strategy_document_payload(
    symbol: str, timeframe: str, dataset: str, strategy_id: str
) -> tuple[dict | None, str]:
    report_path, rows = _load_ranked_rows(symbol.lower(), timeframe, dataset.lower())
    if report_path is None or not rows:
        return None, "No report for this symbol/timeframe/dataset."
    selected = None
    selected_rank = 0
    for idx, row in enumerate(rows, start=1):
        if row.get("id") == strategy_id:
            selected = row
            selected_rank = idx
            break
    if selected is None:
        return None, f"Strategy id {strategy_id} not found in results."
    strategy = _strategy_payload(selected, selected_rank)
    payload = build_document_payload(
        symbol.lower(), timeframe, dataset.lower(), report_path.name, strategy
    )
    return payload, ""


@app.get("/api/strategy-code")
def api_strategy_code(
    symbol: str = "ethusd",
    timeframe: str = "15m",
    dataset: str = "validated",
    strategyId: str = "",
    user: dict = Depends(get_current_user),
) -> Response:
    if not strategyId:
        return _json({"error": "strategyId is required"}, status=400)

    # Credit gate: 4 credits per code download.
    if not user_has_credits(user, CREDIT_COST_CODE):
        return _json(
            {
                "error": "Insufficient credits",
                "message": (
                    f"You need {CREDIT_COST_CODE} credits to download strategy code, "
                    f"but you only have {get_user_credits(user)}."
                ),
                "creditsRequired": CREDIT_COST_CODE,
                "creditsAvailable": get_user_credits(user),
            },
            status=402,
        )

    payload, err = _strategy_document_payload(symbol, timeframe, dataset, strategyId)
    if payload is None:
        return _json({"error": err}, status=404)
    try:
        source = generate_strategy_main_py_source(payload, use_llm=False)
    except Exception as exc:
        return _json({"error": f"Strategy code generation failed: {exc}"}, status=500)

    try:
        new_balance = deduct_credits(user["id"], CREDIT_COST_CODE, reason="strategy_code_download")
    except ValueError:
        # Race condition: balance changed between the check above and now.
        return _json(
            {
                "error": "Insufficient credits",
                "message": f"You need {CREDIT_COST_CODE} credits to download strategy code.",
                "creditsRequired": CREDIT_COST_CODE,
                "creditsAvailable": get_user_credits(user),
            },
            status=402,
        )

    filename = f"main_{payload['symbol'].lower()}_{payload['timeframe'].lower()}_{strategyId}.py"
    return Response(
        content=source.encode("utf-8"),
        media_type="text/x-python; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Allow-Origin": "*",
            "X-Credits-Remaining": str(new_balance),
        },
    )


@app.get("/api/document")
def api_document(
    symbol: str = "ethusd",
    timeframe: str = "15m",
    dataset: str = "validated",
    strategyId: str = "",
    user: dict = Depends(get_current_user),
) -> Response:
    if not strategyId:
        return _json({"error": "strategyId is required"}, status=400)

    # Credit gate: 2 credits per PDF document.
    if not user_has_credits(user, CREDIT_COST_DOCUMENT):
        return _json(
            {
                "error": "Insufficient credits",
                "message": (
                    f"You need {CREDIT_COST_DOCUMENT} credits to generate this document, "
                    f"but you only have {get_user_credits(user)}."
                ),
                "creditsRequired": CREDIT_COST_DOCUMENT,
                "creditsAvailable": get_user_credits(user),
            },
            status=402,
        )

    try:
        filename, pdf_bytes = _strategy_pdf(symbol.lower(), timeframe, dataset.lower(), strategyId)
    except Exception as exc:
        return _json({"error": f"PDF generation failed: {exc}"}, status=500)

    try:
        new_balance = deduct_credits(user["id"], CREDIT_COST_DOCUMENT, reason="document_generation")
    except ValueError:
        return _json(
            {
                "error": "Insufficient credits",
                "message": f"You need {CREDIT_COST_DOCUMENT} credits to generate this document.",
                "creditsRequired": CREDIT_COST_DOCUMENT,
                "creditsAvailable": get_user_credits(user),
            },
            status=402,
        )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Allow-Origin": "*",
            "X-Credits-Remaining": str(new_balance),
        },
    )

@app.get("/admin/initdb")
def admin_initdb() -> Response:
    try:
        init_db()
        return _json({"status": "ok", "message": "Database initialized successfully"})
    except Exception as exc:
        return _json({"status": "error", "message": str(exc)}, status=500)

# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    import os
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", HOST),
        port=int(os.environ.get("PORT", PORT)),
        log_level="info",
    )


if __name__ == "__main__":
    main()
