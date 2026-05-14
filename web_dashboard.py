

"""
web_dashboard.py
────────────────
Local strategy report website.

Run:
    python web_dashboard.py

Then open:
    http://127.0.0.1:8767

Changes from v1:
  • /api/document now returns a PDF (application/pdf) instead of Markdown.
  • Content-Disposition set to attachment so the browser downloads the file.
  • Strategy document generation now waits for the LLM (sleep is inside
    llm_generator.generate_strategy_document).
  • /api/strategy-code returns a downloadable Python bot (main_*.py) built via
    tools_llm_strategy + Gemini; falls back to a template if the model output is invalid.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
import sqlite3
import time
from datetime import datetime, timezone

from llm_generator import build_document_payload, generate_strategy_document
from tools_llm_strategy import generate_strategy_main_py_source

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.middleware.sessions import SessionMiddleware
from starlette.status import HTTP_303_SEE_OTHER
from passlib.context import CryptContext
from starlette.middleware.base import BaseHTTPMiddleware


ROOT         = Path(__file__).resolve().parent
DATA_DIR     = ROOT / "data"
RESULTS_DIR  = ROOT / "results"
DASHBOARD_DIR = ROOT / "dashboard"
PORT         = 10000
HOST         = "0.0.0.0"
AUTH_DB_PATH = ROOT / "users.db"

DATA_RE = re.compile(r"(?P<symbol>[a-z]+usd)_(?P<timeframe>[^.]+)\.csv$", re.IGNORECASE)
REGIMES = ("chop", "trendy", "volatile")
PWD_CTX = CryptContext(schemes=["bcrypt"], deprecated="auto")


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


# ── HTTP response helpers ─────────────────────────────────────────────────────

def _json(payload: dict, status: int = 200) -> JSONResponse:
    return JSONResponse(content=payload, status_code=status)


# ── scan / file helpers ───────────────────────────────────────────────────────

def _scan_options() -> dict:
    symbols: set[str] = set()
    timeframes_by_symbol: dict[str, set[str]] = {}
    reports: dict[str, dict[str, dict[str, str]]] = {}

    for path in DATA_DIR.glob("*.csv"):
        match = DATA_RE.match(path.name)
        if not match:
            continue
        symbol    = _normalize_symbol(match.group("symbol"))
        timeframe = _display_timeframe(match.group("timeframe"))
        symbols.add(symbol)
        timeframes_by_symbol.setdefault(symbol, set()).add(timeframe)

    for path in RESULTS_DIR.glob("strategy_results_*.csv"):
        parsed = _parse_result_name(path.name)
        if parsed is None:
            continue
        symbol, timeframe, kind = parsed
        symbols.add(symbol)
        timeframes_by_symbol.setdefault(symbol, set()).add(timeframe)
        reports.setdefault(symbol, {}).setdefault(timeframe, {})[kind] = path.name

    return {
        "symbols": sorted(symbols),
        "timeframesBySymbol": {
            sym: sorted(vals, key=_timeframe_sort_key)
            for sym, vals in timeframes_by_symbol.items()
        },
        "reports": reports,
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
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    return _normalize_symbol(parts[2]), _display_timeframe(parts[3]), kind


def _timeframe_sort_key(value: str) -> tuple[int, int | str]:
    suffix = value[-1].lower()
    number = value[:-1]
    if value.upper() == "1D":    return (4, 1)
    if suffix == "m" and number.isdigit(): return (1, int(number))
    if suffix == "h" and number.isdigit(): return (2, int(number))
    return (9, value)


def _candidate_report(symbol: str, timeframe: str, dataset: str) -> Path | None:
    symbol   = _normalize_symbol(symbol)
    file_tf  = _file_timeframe(timeframe)
    kind     = "validated" if dataset == "validated" else "train_top500"
    sym_cands = [symbol]
    if symbol.endswith("usd"):
        sym_cands.append(symbol[:-3])

    candidates = []
    for sym_name in sym_cands:
        pattern = f"strategy_results_{sym_name}_{file_tf}*_{kind}.csv"
        candidates.extend(RESULTS_DIR.glob(pattern))

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _num(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        return float(value) if value not in ("", None) else default
    except (TypeError, ValueError):
        return default


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
            "sharpe":     _num(row, "train_sharpe"),
            "drawdown":   _num(row, "train_drawdown"),
            "trades":     int(_num(row, "train_trades")),
            "winRate":    _num(row, "train_winrate"),
        },
        "test": {
            "returnPct":  _num(row, "test_return"),
            "sharpe":     _num(row, "test_sharpe"),
            "drawdown":   _num(row, "test_drawdown"),
            "trades":     int(_num(row, "test_trades")),
            "winRate":    _num(row, "test_winrate"),
        },
        "regimes": regimes,
    }


def _report(symbol: str, timeframe: str, dataset: str, limit: int) -> dict:
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

    score_key = "test_score" if dataset == "validated" and rows and "test_score" in rows[0] else "score"
    rows.sort(key=lambda r: _num(r, score_key), reverse=True)
    top_rows = rows[:limit]

    return {
        "available":        True,
        "symbol":           symbol,
        "timeframe":        timeframe,
        "dataset":          dataset,
        "sourceFile":       report_path.name,
        "totalStrategies":  len(rows),
        "top": [_strategy_payload(r, idx + 1) for idx, r in enumerate(top_rows)],
    }


def _load_ranked_rows(symbol: str, timeframe: str, dataset: str) -> tuple[Path | None, list[dict]]:
    report_path = _candidate_report(symbol, timeframe, dataset)
    if report_path is None:
        return None, []
    with report_path.open("r", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    score_key = "test_score" if dataset == "validated" and rows and "test_score" in rows[0] else "score"
    rows.sort(key=lambda r: _num(r, score_key), reverse=True)
    return report_path, rows


def _strategy_pdf(symbol: str, timeframe: str, dataset: str, strategy_id: str) -> tuple[str, bytes]:
    """
    Generate a PDF report for one strategy.

    Returns (filename, pdf_bytes).
    Falls back gracefully if no report exists.
    """
    report_path, rows = _load_ranked_rows(symbol, timeframe, dataset)

    if report_path is None:
        # Return a minimal error PDF
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

    # Find the strategy row
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

    # generate_strategy_document calls Gemini (with retries + sleep) then builds PDF
    pdf_bytes = generate_strategy_document(payload)
    filename  = f"strategy_{payload['symbol'].lower()}_{payload['timeframe'].lower()}_{strategy_id}.pdf"
    return filename, pdf_bytes


# ── auth db helpers ───────────────────────────────────────────────────────────

def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_auth_db() -> None:
    AUTH_DB_PATH.parent.mkdir(exist_ok=True)
    with _db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              full_name TEXT NOT NULL,
              email TEXT NOT NULL UNIQUE,
              password_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _create_user(full_name: str, email: str, password: str) -> None:
    email_norm = email.strip().lower()
    now = datetime.now(timezone.utc).isoformat()
    password_hash = PWD_CTX.hash(password)
    with _db_connect() as conn:
        conn.execute(
            "INSERT INTO users (full_name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (full_name.strip(), email_norm, password_hash, now),
        )
        conn.commit()


def _get_user_by_email(email: str) -> sqlite3.Row | None:
    email_norm = email.strip().lower()
    with _db_connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email_norm,)).fetchone()
    return row


def _verify_login(email: str, password: str) -> sqlite3.Row | None:
    user = _get_user_by_email(email)
    if user is None:
        return None
    if not PWD_CTX.verify(password, user["password_hash"]):
        return None
    return user


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="MultiStrategyGenerator Dashboard")


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url=url, status_code=HTTP_303_SEE_OTHER)


class AuthRedirectMiddleware(BaseHTTPMiddleware):
    """
    Enforce: user must be logged-in to view the dashboard and call APIs.
    Public routes stay accessible to allow login/signup.
    """

    async def dispatch(self, request: Request, call_next):
        public_prefixes = ("/auth/", "/dashboard/")
        public_exact = {"/login.html", "/signup.html", "/favicon.ico"}
        path = request.url.path or "/"

        if path in public_exact or any(path.startswith(p) for p in public_prefixes):
            return await call_next(request)

        # SessionMiddleware must be installed for request.session to exist.
        if request.session.get("user") is None:
            return _redirect("/login.html")

        return await call_next(request)


# Important: add auth first, session last (session becomes outermost wrapper)
app.add_middleware(AuthRedirectMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key="change-me-in-production",
    same_site="lax",
    https_only=False,
)


@app.on_event("startup")
def _startup() -> None:
    # Ensure required directories exist
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    DASHBOARD_DIR.mkdir(exist_ok=True)
    _init_auth_db()


# ── public pages ──────────────────────────────────────────────────────────────

@app.get("/login.html", response_class=HTMLResponse)
def login_page() -> Response:
    return FileResponse(DASHBOARD_DIR / "login.html", media_type="text/html")


@app.get("/signup.html", response_class=HTMLResponse)
def signup_page() -> Response:
    return FileResponse(DASHBOARD_DIR / "signup.html", media_type="text/html")


@app.get("/logout")
def logout(request: Request) -> Response:
    request.session.clear()
    return _redirect("/login.html")


# ── auth actions ──────────────────────────────────────────────────────────────

@app.post("/auth/signup")
def signup(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
) -> Response:
    if password != confirm_password:
        return _redirect("/signup.html?error=password_mismatch")
    try:
        _create_user(full_name=full_name, email=email, password=password)
    except sqlite3.IntegrityError:
        return _redirect("/signup.html?error=email_exists")
    user = _get_user_by_email(email)
    request.session["user"] = {"id": int(user["id"]), "email": user["email"], "full_name": user["full_name"]}
    return _redirect("/")


@app.post("/auth/login")
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
) -> Response:
    user = _verify_login(email=email, password=password)
    if user is None:
        return _redirect("/login.html?error=invalid_credentials")
    request.session["user"] = {"id": int(user["id"]), "email": user["email"], "full_name": user["full_name"]}
    return _redirect("/")


# ── protected pages ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index() -> Response:
    return FileResponse(DASHBOARD_DIR / "index.html", media_type="text/html")


@app.get("/index.html", response_class=HTMLResponse)
def index_html() -> Response:
    return FileResponse(DASHBOARD_DIR / "index.html", media_type="text/html")


@app.get("/dashboard/{path:path}")
def dashboard_files(path: str) -> Response:
    # optional: allow serving assets under /dashboard/... if you add JS/CSS later
    file_path = (DASHBOARD_DIR / path).resolve()
    if DASHBOARD_DIR not in file_path.parents and file_path != DASHBOARD_DIR:
        return Response(status_code=404)
    if not file_path.exists() or not file_path.is_file():
        return Response(status_code=404)
    return FileResponse(file_path)


# ── API (protected) ───────────────────────────────────────────────────────────

@app.get("/api/options")
def api_options() -> Response:
    return _json(_scan_options())


@app.get("/api/report")
def api_report(
    symbol: str = "ethusd",
    timeframe: str = "15m",
    dataset: str = "validated",
    limit: int = 10,
) -> Response:
    return _json(_report(symbol.lower(), timeframe, dataset.lower(), int(limit)))


def _strategy_document_payload(
    symbol: str, timeframe: str, dataset: str, strategy_id: str
) -> tuple[dict | None, str]:
    """
    Load CSV row and build the same structured payload used for PDF generation.
    Returns (payload, error_message). payload is None on failure.
    """
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
) -> Response:
    """
    Generate a downloadable Python bot script (main.py style) via LLM, using
    BinanceBroker only through tools_llm_strategy.SimpleStrategy.
    """
    if not strategyId:
        return _json({"error": "strategyId is required"}, status=400)
    payload, err = _strategy_document_payload(symbol, timeframe, dataset, strategyId)
    if payload is None:
        return _json({"error": err}, status=404)
    try:
        source = generate_strategy_main_py_source(payload, use_llm=True)
    except Exception as exc:
        return _json({"error": f"Strategy code generation failed: {exc}"}, status=500)
    filename = f"main_{payload['symbol'].lower()}_{payload['timeframe'].lower()}_{strategyId}.py"
    return Response(
        content=source.encode("utf-8"),
        media_type="text/x-python; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/document")
def api_document(
    symbol: str = "ethusd",
    timeframe: str = "15m",
    dataset: str = "validated",
    strategyId: str = "",
) -> Response:
    if not strategyId:
        return _json({"error": "strategyId is required"}, status=400)
    try:
        filename, pdf_bytes = _strategy_pdf(symbol.lower(), timeframe, dataset.lower(), strategyId)
    except Exception as exc:
        return _json({"error": f"PDF generation failed: {exc}"}, status=500)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Allow-Origin": "*",
        },
    )


# def main() -> None:
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


# if __name__ == "__main__":
#     main()

def main() -> None:
    import os
    import uvicorn

    uvicorn.run(
        app,
        host = os.environ.get("HOST",HOST),
        port=int(os.environ.get("PORT", PORT)),
        log_level="info",
    )


if __name__ == "__main__":
    main()