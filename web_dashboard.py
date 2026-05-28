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

import csv
import json
import re
from pathlib import Path
import time
from datetime import datetime, timezone

from llm_generator import build_document_payload, generate_strategy_document
from tools_llm_strategy import generate_strategy_main_py_source

from auth_db import (
    authenticate_user,
    create_access_token,
    create_user,
    get_current_user,
    init_db,
    log_event,
    save_prediction,
)
from ml_pipeline import predict_probabilities
from risk_validation import latest_validation_summary

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response


ROOT          = Path(__file__).resolve().parent
DATA_DIR      = ROOT / "data"
RESULTS_DIR   = ROOT / "results"
DASHBOARD_DIR = ROOT / "dashboard"
PORT          = 10000
# HOST          = "0.0.0.0"
HOST          = "127.0.0.1"

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
        if timeframe.lower() in HIDDEN_TIMEFRAMES:
            continue
        symbols.add(symbol)
        timeframes_by_symbol.setdefault(symbol, set()).add(timeframe)

    for path in RESULTS_DIR.glob("strategy_results_*.csv"):
        parsed = _parse_result_name(path.name)
        if parsed is None:
            continue
        symbol, timeframe, kind = parsed
        if timeframe.lower() in HIDDEN_TIMEFRAMES:
            continue
        symbols.add(symbol)
        timeframes_by_symbol.setdefault(symbol, set()).add(timeframe)
        reports.setdefault(symbol, {}).setdefault(timeframe, {})[kind] = path.name

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
                    kind = "validated" if row.get("result_type") == "validated" else "train"
                    symbols.add(symbol)
                    timeframes_by_symbol.setdefault(symbol, set()).add(timeframe)
                    reports.setdefault(symbol, {}).setdefault(timeframe, {})[kind] = unified.name
        except Exception:
            pass

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
    symbol   = _normalize_symbol(symbol)
    file_tf  = _file_timeframe(timeframe)
    kind     = "validated" if dataset == "validated" else "train_top500"
    sym_cands = [symbol]
    if symbol.endswith("usd"):
        sym_cands.append(symbol[:-3])

    candidates = []
    for sym_name in sym_cands:
        candidates.extend(RESULTS_DIR.glob(f"strategy_results_{sym_name}_{file_tf}*_{kind}.csv"))
        candidates.extend(RESULTS_DIR.glob(f"strategy_results_{file_tf}_{sym_name}*_{kind}.csv"))

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
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
    return filename, pdf_bytes


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="MultiStrategyGenerator Dashboard")


@app.on_event("startup")
def _startup() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    DASHBOARD_DIR.mkdir(exist_ok=True)
    try:
        init_db()
    except Exception as exc:
        print(f"PostgreSQL init skipped: {exc}")


# ── pages ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    try:
        get_current_user(request)
    except HTTPException:
        return RedirectResponse("/login", status_code=303)
    return FileResponse(DASHBOARD_DIR / "index.html", media_type="text/html")


@app.get("/index.html", response_class=HTMLResponse)
def index_html(request: Request) -> Response:
    try:
        get_current_user(request)
    except HTTPException:
        return RedirectResponse("/login", status_code=303)
    return FileResponse(DASHBOARD_DIR / "index.html", media_type="text/html")


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
        return RedirectResponse("/signup?error=password_mismatch", status_code=303)
    try:
        user = create_user(email, full_name, password)
    except Exception:
        return RedirectResponse("/signup?error=email_exists", status_code=303)
    token = create_access_token({"sub": str(user["id"])})
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("access_token", token, httponly=True, samesite="lax", max_age=60 * 60 * 2)
    log_event("signup", "User created", user_id=user["id"])
    return response


@app.post("/auth/login")
def auth_login(email: str = Form(...), password: str = Form(...)) -> Response:
    user = authenticate_user(email, password)
    if not user:
        return RedirectResponse("/login?error=invalid_credentials", status_code=303)
    token = create_access_token({"sub": str(user["id"])})
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("access_token", token, httponly=True, samesite="lax", max_age=60 * 60 * 2)
    log_event("login", "User logged in", user_id=user["id"])
    return response


@app.get("/logout")
def logout() -> Response:
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("access_token")
    return response


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
    allowed = any(word in lower for word in ("probability", "probabilities", "signal", "risk", "buy", "sell", "validation", "sharpe", "drawdown"))
    if not allowed:
        return _json({"answer": "Ask me about probabilities, signals, validation, or risk for the selected market."})
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
            f"test Sharpe {best['test']['sharpe']:.2f}, test return {best['test']['returnPct']:.2f}%, "
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
    user: dict = Depends(get_current_user),
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
