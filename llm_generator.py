"""
llm_generator.py
────────────────
LLM-backed strategy document generation → PDF output.

Uses the official `google-genai` SDK (not urllib, not the deprecated
google.generativeai package).

Key behaviours
  • Calls Gemini 2.5-flash via google-genai SDK
  • Retries up to MAX_RETRIES times with RETRY_DELAY_S gap
  • Waits PRE_SEND_SLEEP seconds AFTER a successful LLM response so the
    caller receives the document only after a realistic generation delay
  • Converts the LLM narrative + structured metrics into a styled PDF
    (reportlab) — no .md file is ever produced
  • Falls back to a deterministic text document if the LLM fails
"""

from __future__ import annotations

import time
import traceback
from io import BytesIO

# ── Gemini SDK ────────────────────────────────────────────────────────────────
from google import genai as google_genai
from google.genai import types as genai_types

# ── ReportLab ─────────────────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Project ───────────────────────────────────────────────────────────────────
from signals import SIGNALS

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import GEMINI_API_KEY
except Exception:
    GEMINI_API_KEY = "AIzaSyBSO6iLR30iegiBQdkfynxNnXK3W3WfHaY"

GEMINI_MODEL   = "gemini-2.5-flash"   # model that actually works with your key
MAX_RETRIES    = 3                     # retries on transient failures
RETRY_DELAY_S  = 5                     # seconds between retries
PRE_SEND_SLEEP = 5                     # seconds to hold after generation before serving
REGIMES        = ("chop", "trendy", "volatile")


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL FUNCTIONS  (called by build_document_payload)
# ══════════════════════════════════════════════════════════════════════════════

def get_signal_details(signal_text: str) -> list[dict]:
    """Map compact signal codes to descriptions and metadata."""
    details = []
    for key in [p.strip() for p in signal_text.split("|") if p.strip()]:
        meta = SIGNALS.get(key, {})
        fn   = meta.get("fn")
        # Pull first line of the function docstring as description
        desc = (fn.__doc__ or "No description").strip().split("\n")[0] if fn else "Unknown signal"
        details.append({
            "code":        key,
            "description": desc,
            "group":       meta.get("group", "unknown"),
            "direction":   meta.get("dir",   "unknown"),
        })
    return details


def get_strategy_parameters(strategy: dict) -> dict:
    """Extract clean execution parameters from a strategy dict."""
    direction = strategy.get("direction", "unknown")
    is_bull   = direction == "bull"
    return {
        "direction":       direction,
        "numberOfSignals": strategy.get("nSignals", 0),
        "takeProfitPct":   strategy.get("tp", 0) * 100,
        "stopLossPct":     strategy.get("sl", 0) * 100,
        "entryThreshold":  "All selected signals must be active simultaneously (AND logic).",
        "buyTrade": (
            "Enabled — enter long when the bullish signal stack confirms."
            if is_bull else
            "Not applicable — this is a bearish/short strategy."
        ),
        "sellTrade": (
            "Not applicable — this is a bullish/long strategy."
            if is_bull else
            "Enabled — enter short when the bearish signal stack confirms."
        ),
        "exitRules": [
            "Take profit exits at the configured TP percentage from entry price.",
            "Stop loss exits at the configured SL percentage from entry price.",
            "Exclusive orders — only one position is held at a time.",
        ],
    }


def get_market_condition_suggestion(strategy: dict) -> dict:
    """Rank market conditions by test regime performance."""
    regimes = strategy.get("regimes", {}).get("test", {})
    ranked  = sorted(
        REGIMES,
        key=lambda r: (
            regimes.get(r, {}).get("returnPct", 0),
            regimes.get(r, {}).get("trades",    0),
            regimes.get(r, {}).get("winRate",   0),
        ),
        reverse=True,
    )
    best    = ranked[0] if ranked else "unknown"
    weakest = ranked[-1] if ranked else "unknown"
    bd      = regimes.get(best,    {})
    wd      = regimes.get(weakest, {})
    return {
        "bestCondition":    best,
        "weakestCondition": weakest,
        "suggestion": (
            f"Best suited for {best} markets "
            f"(test return {bd.get('returnPct', 0):.2f}%, "
            f"{bd.get('trades', 0)} trades, "
            f"{bd.get('winRate', 0):.1f}% win rate). "
            f"Weakest in {weakest} markets "
            f"(test return {wd.get('returnPct', 0):.2f}%)."
        ),
    }


def build_document_payload(
    symbol:      str,
    timeframe:   str,
    dataset:     str,
    source_file: str,
    strategy:    dict,
) -> dict:
    """Build the complete structured payload for the LLM and PDF renderer."""
    enriched = dict(strategy)
    enriched["signalDetails"]             = get_signal_details(strategy.get("signals", ""))
    enriched["parameters"]                = get_strategy_parameters(strategy)
    enriched["marketConditionSuggestion"] = get_market_condition_suggestion(strategy)

    return {
        "symbol":     symbol.upper(),
        "timeframe":  timeframe.upper(),
        "dataset":    dataset,
        "sourceFile": source_file,
        "strategy":   enriched,
        "stockUniverse": {
            "executionUniverse": symbol.upper(),
            "timeframe":         timeframe.upper(),
            "dataSource":        "Generated OHLCV CSV and strategy result CSV from MultiStrategyGenerator.",
        },
        "backtestContext": {
            "initialCash":       1_000_000,
            "commission":        "0.2%",
            "trainTestSplit":    "70% train / 30% test",
            "regimeModel":       "GaussianNB on OHLCV volatility, range, trend, efficiency, and volume features",
            "tradeAttribution":  "Regime metrics attributed by the regime active on each trade entry bar.",
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI CALL  (google-genai SDK)
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(payload: dict) -> str:
    import json
    return (
        "You are writing a quantitative trading strategy documentation note.\n"
        "Use ONLY the facts in the JSON payload below. Do not invent any numbers.\n"
        "Return plain text with section headers using ## for sections and ### for subsections.\n"
        "Do NOT use markdown tables, bullet points with *, asterisks for bold, or code blocks.\n"
        "Do NOT include client names, broker integrations, or business proposal content.\n\n"
        "Required sections:\n"
        "1. Project Overview\n"
        "2. Strategy Description (list each signal code with its description)\n"
        "3. Stock Universe and Timeframe\n"
        "4. Entry and Exit Conditions\n"
        "   - Buy Trade\n"
        "   - Sell Trade\n"
        "   - Exit Conditions\n"
        "5. Strategy Parameters\n"
        "6. Performance Metrics\n"
        "   - Train Metrics\n"
        "   - Test Metrics\n"
        "7. Market Condition Analysis\n"
        "   - Best Condition\n"
        "   - Weakest Condition\n"
        "8. Final Assessment\n\n"
        "If strategy direction is bull, clearly state sell/short is not applicable.\n"
        "If strategy direction is bear, clearly state buy/long is not applicable.\n\n"
        f"Payload:\n{json.dumps(payload, indent=2)}"
    )


def _call_gemini(prompt: str, max_output_tokens: int = 2400) -> str:
    """
    Call Gemini 2.5-flash using the google-genai SDK.
    Retries up to MAX_RETRIES times on failure.
    Raises RuntimeError if all attempts fail.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to config.py:\n"
            "  GEMINI_API_KEY = 'your-key-here'"
        )

    client = google_genai.Client(api_key=GEMINI_API_KEY)

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[LLM] Calling Gemini {GEMINI_MODEL} (attempt {attempt}/{MAX_RETRIES}) …")

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.8,
                    max_output_tokens=max_output_tokens,
                ),
            )

            text = response.text.strip()
            if not text:
                raise RuntimeError("Gemini returned an empty response.")

            print(f"[LLM] Response received ({len(text)} chars).")
            return text

        except Exception as exc:
            last_exc = exc
            print(f"[LLM] Attempt {attempt} failed: {exc}")
            if attempt < MAX_RETRIES:
                print(f"[LLM] Retrying in {RETRY_DELAY_S}s …")
                time.sleep(RETRY_DELAY_S)

    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} attempts. Last error: {last_exc}")


# ══════════════════════════════════════════════════════════════════════════════
#  FALLBACK TEXT  (deterministic — used when LLM is unavailable)
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_text(payload: dict, note: str = "") -> str:
    strategy  = payload["strategy"]
    params    = strategy.get("parameters", {})
    condition = strategy.get("marketConditionSuggestion", {})

    signals_block = "\n".join(
        f"  {item['code']}: {item['description']}  [{item['group']}]"
        for item in strategy.get("signalDetails", [])
    )

    def _regime_block(split: str) -> str:
        lines = []
        for regime in REGIMES:
            item = strategy.get("regimes", {}).get(split, {}).get(regime, {})
            lines.append(
                f"  {regime.capitalize()}: {item.get('trades', 0)} trades, "
                f"return {item.get('returnPct', 0):.2f}%, "
                f"win rate {item.get('winRate', 0):.2f}%, "
                f"avg trade return {item.get('avgTradeReturnPct', 0):.3f}%"
            )
        return "\n".join(lines)

    note_line = f"\nNote: {note}\n" if note else ""
    train = strategy.get("train", {})
    test  = strategy.get("test",  {})

    return f"""## Project Overview
{note_line}
Strategy {strategy.get('id','')} was generated by the MultiStrategyGenerator system for {payload['symbol']} on the {payload['timeframe']} timeframe. It is a {strategy.get('direction','')} strategy using {strategy.get('nSignals', 0)} entry signals with fixed take-profit and stop-loss parameters.

## Strategy Description

The strategy uses the following signal conditions combined with AND logic:

{signals_block}

All signals must be active simultaneously for an entry to be triggered.

## Stock Universe and Timeframe

Execution universe: {payload['stockUniverse']['executionUniverse']}
Timeframe: {payload['stockUniverse']['timeframe']}
Dataset: {payload['dataset']}
Source file: {payload['sourceFile']}

## Entry and Exit Conditions

### Buy Trade

{params.get('buyTrade', 'N/A')}

### Sell Trade

{params.get('sellTrade', 'N/A')}

### Exit Conditions

Take Profit: {params.get('takeProfitPct', 0):.2f}%
Stop Loss: {params.get('stopLossPct', 0):.2f}%
Entry logic: {params.get('entryThreshold', '')}

## Strategy Parameters

Direction: {params.get('direction', '')}
Number of signals: {params.get('numberOfSignals', 0)}
Take profit: {params.get('takeProfitPct', 0):.2f}%
Stop loss: {params.get('stopLossPct', 0):.2f}%
Initial backtest cash: {payload['backtestContext']['initialCash']:,}
Commission: {payload['backtestContext']['commission']}
Train/Test split: {payload['backtestContext']['trainTestSplit']}

## Performance Metrics

### Train Metrics

Return: {train.get('returnPct', 0):.2f}%
Sharpe Ratio: {train.get('sharpe', 0):.3f}
Max Drawdown: {train.get('drawdown', 0):.2f}%
Number of Trades: {train.get('trades', 0)}
Win Rate: {train.get('winRate', 0):.2f}%

### Test Metrics

Return: {test.get('returnPct', 0):.2f}%
Sharpe Ratio: {test.get('sharpe', 0):.3f}
Max Drawdown: {test.get('drawdown', 0):.2f}%
Number of Trades: {test.get('trades', 0)}
Win Rate: {test.get('winRate', 0):.2f}%
Composite Score: {strategy.get('score', 0):.5f}

## Market Condition Analysis

{condition.get('suggestion', '')}

### Train Regime Performance

{_regime_block('train')}

### Test Regime Performance

{_regime_block('test')}

## Final Assessment

This report is based solely on historical backtest results and machine-generated regime labels. It should be used as a starting point for research only. Forward-test and validate before any live deployment.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  PDF BUILDER  (reportlab — Platypus)
# ══════════════════════════════════════════════════════════════════════════════

# ── colour palette ─────────────────────────────────────────────────────────────
_DARK     = colors.HexColor("#1a1a2e")
_ACCENT   = colors.HexColor("#0f3460")
_TEAL     = colors.HexColor("#16213e")
_GREEN    = colors.HexColor("#0d7377")
_RED      = colors.HexColor("#c0392b")
_LIGHT_BG = colors.HexColor("#f4f6f8")
_BORDER   = colors.HexColor("#dee2e6")
_MID_GREY = colors.HexColor("#6c757d")
_WHITE    = colors.white


def _styles() -> dict:
    return {
        "title": ParagraphStyle(
            "DocTitle", fontName="Helvetica-Bold", fontSize=20,
            textColor=_WHITE, spaceAfter=4, leading=24,
        ),
        "subtitle": ParagraphStyle(
            "DocSubtitle", fontName="Helvetica", fontSize=10,
            textColor=colors.HexColor("#adb5bd"), spaceAfter=0, leading=13,
        ),
        "h2": ParagraphStyle(
            "H2", fontName="Helvetica-Bold", fontSize=13,
            textColor=_ACCENT, spaceBefore=14, spaceAfter=4, leading=16,
        ),
        "h3": ParagraphStyle(
            "H3", fontName="Helvetica-Bold", fontSize=10,
            textColor=_DARK, spaceBefore=8, spaceAfter=3, leading=13,
        ),
        "body": ParagraphStyle(
            "Body", fontName="Helvetica", fontSize=9,
            textColor=_DARK, leading=13, spaceAfter=4,
        ),
        "label": ParagraphStyle(
            "Label", fontName="Helvetica-Bold", fontSize=8,
            textColor=_MID_GREY, leading=11,
        ),
        "metric": ParagraphStyle(
            "Metric", fontName="Helvetica-Bold", fontSize=11,
            textColor=_DARK, leading=14,
        ),
    }


def _banner(story: list, payload: dict, st: dict) -> None:
    strategy  = payload["strategy"]
    sid       = strategy.get("id", "")
    symbol    = payload["symbol"]
    tf        = payload["timeframe"]
    direction = strategy.get("direction", "").upper()
    dir_col   = "#0d7377" if direction == "BULL" else "#c0392b"
    score     = strategy.get("score", 0)

    banner = [[
        Paragraph("Strategy Report", st["title"]),
        Paragraph(
            f'<font color="{dir_col}"><b>{symbol} {tf}</b></font>  '
            f'<font color="#adb5bd">{direction}</font>',
            st["subtitle"],
        ),
        Paragraph(
            f'<font color="#adb5bd">ID: </font><b>{sid}</b><br/>'
            f'<font color="#adb5bd">Score: </font>{score:.5f}',
            st["subtitle"],
        ),
    ]]
    t = Table(banner, colWidths=[7*cm, 5.5*cm, 5.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _DARK),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 16),
        ("BOTTOMPADDING", (0,0), (-1,-1), 16),
        ("LEFTPADDING",   (0,0), (0, 0),  16),
        ("RIGHTPADDING",  (-1,0),(-1, 0), 16),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))


def _params_table(story: list, payload: dict, st: dict) -> None:
    strategy = payload["strategy"]
    params   = strategy.get("parameters", {})
    story.append(Paragraph("Strategy Parameters", st["h2"]))

    data = [
        ["Direction", "Signals", "Take Profit", "Stop Loss", "Commission", "Cash"],
        [
            params.get("direction", "").upper(),
            str(params.get("numberOfSignals", 0)),
            f"{params.get('takeProfitPct', 0):.2f}%",
            f"{params.get('stopLossPct', 0):.2f}%",
            payload["backtestContext"]["commission"],
            f"${payload['backtestContext']['initialCash']:,}",
        ],
    ]
    cw = [3*cm, 2.5*cm, 3*cm, 3*cm, 3*cm, 3.5*cm]
    t  = Table(data, colWidths=cw, rowHeights=18)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), _TEAL),
        ("TEXTCOLOR",     (0,0), (-1,0), _WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("BOX",           (0,0), (-1,-1), 0.5, _BORDER),
        ("INNERGRID",     (0,0), (-1,-1), 0.25, _BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))


def _metrics_tables(story: list, strategy: dict, st: dict) -> None:
    story.append(Paragraph("Performance Metrics", st["h2"]))

    for split, label in [("train", "Train"), ("test", "Test")]:
        m   = strategy.get(split, {})
        ret = m.get("returnPct", 0)
        sh  = m.get("sharpe",    0)
        dd  = m.get("drawdown",  0)
        tr  = m.get("trades",    0)
        wr  = m.get("winRate",   0)

        story.append(Paragraph(f"{label} Performance", st["h3"]))
        data = [
            ["Return", "Sharpe", "Max Drawdown", "Trades", "Win Rate"],
            [
                Paragraph(
                    f'<font color="{"#0d7377" if ret >= 0 else "#c0392b"}">{ret:.2f}%</font>',
                    st["metric"]),
                Paragraph(f"{sh:.3f}", st["metric"]),
                Paragraph(
                    f'<font color="{"#0d7377" if dd >= -10 else "#c0392b"}">{dd:.2f}%</font>',
                    st["metric"]),
                Paragraph(str(tr), st["metric"]),
                Paragraph(f"{wr:.1f}%", st["metric"]),
            ],
        ]
        cw = [3.6*cm] * 5
        t  = Table(data, colWidths=cw, rowHeights=[18, 28])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), _LIGHT_BG),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,0), 8),
            ("TEXTCOLOR",     (0,0), (-1,0), _MID_GREY),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("BOX",           (0,0), (-1,-1), 0.5, _BORDER),
            ("INNERGRID",     (0,0), (-1,-1), 0.25, _BORDER),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 6))


def _regime_tables(story: list, strategy: dict, st: dict) -> None:
    story.append(Paragraph("Regime Analysis", st["h2"]))

    for split, label in [("train", "Train"), ("test", "Test")]:
        story.append(Paragraph(f"Regime Performance — {label}", st["h3"]))
        rd   = strategy.get("regimes", {}).get(split, {})
        rows = [["Regime", "Trades", "Return %", "Win Rate %", "Avg Trade %"]]
        for regime in REGIMES:
            item = rd.get(regime, {})
            rows.append([
                regime.capitalize(),
                str(item.get("trades", 0)),
                f"{item.get('returnPct', 0):.2f}%",
                f"{item.get('winRate', 0):.2f}%",
                f"{item.get('avgTradeReturnPct', 0):.3f}%",
            ])
        cw = [3*cm, 2.5*cm, 3*cm, 3*cm, 3*cm]
        t  = Table(rows, colWidths=cw, rowHeights=18)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), _ACCENT),
            ("TEXTCOLOR",     (0,0), (-1,0), _WHITE),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8),
            ("ALIGN",         (1,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [_WHITE, _LIGHT_BG]),
            ("BOX",           (0,0), (-1,-1), 0.5, _BORDER),
            ("INNERGRID",     (0,0), (-1,-1), 0.25, _BORDER),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 6))


def _signals_section(story: list, strategy: dict, st: dict) -> None:
    story.append(Paragraph("Entry Signals", st["h2"]))
    for sig in strategy.get("signalDetails", []):
        code  = sig.get("code",        "")
        desc  = sig.get("description", "")
        group = sig.get("group",       "")
        dir_  = sig.get("direction",   "")
        col   = "#0d7377" if dir_ == "bull" else "#c0392b" if dir_ == "bear" else "#6c757d"
        story.append(Paragraph(
            f'<font color="{col}"><b>{code}</b></font>'
            f'  <font color="#6c757d">[{group}]</font>  {desc}',
            st["body"],
        ))
    story.append(Spacer(1, 6))


def _narrative_section(story: list, text: str, st: dict) -> None:
    """Parse the flat LLM text into reportlab flowables."""
    story.append(HRFlowable(width="100%", thickness=1, color=_ACCENT, spaceBefore=8, spaceAfter=6))
    story.append(Paragraph("Strategy Analysis", st["h2"]))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 4))
        elif line.startswith("## "):
            story.append(HRFlowable(width="100%", thickness=0.4, color=_BORDER, spaceAfter=3))
            story.append(Paragraph(line[3:], st["h2"]))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], st["h3"]))
        elif line.startswith("# "):
            story.append(Paragraph(line[2:], st["h2"]))
        else:
            # Strip stray markdown symbols
            clean = line.lstrip("*-# ").replace("**", "").replace("__", "")
            if clean:
                story.append(Paragraph(clean, st["body"]))


def build_pdf(payload: dict, text: str) -> bytes:
    """
    Render the complete strategy PDF from payload + LLM/fallback narrative.

    Returns raw PDF bytes.
    """
    buf      = BytesIO()
    strategy = payload["strategy"]
    st       = _styles()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.5*cm,  bottomMargin=1.8*cm,
        title=f"Strategy {strategy.get('id','')} — {payload['symbol']} {payload['timeframe']}",
        author="MultiStrategyGenerator",
    )

    story: list = []
    _banner(story, payload, st)
    _signals_section(story, strategy, st)
    _params_table(story, payload, st)
    _metrics_tables(story, strategy, st)
    _regime_tables(story, strategy, st)
    _narrative_section(story, text, st)

    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API  — called by web_dashboard.py
# ══════════════════════════════════════════════════════════════════════════════

def generate_strategy_document(payload: dict) -> bytes:
    """
    Generate a PDF strategy document.

    Workflow:
      1. Call Gemini 2.5-flash via google-genai SDK (with retries).
      2. Sleep PRE_SEND_SLEEP seconds so the document is not served
         instantaneously (gives realistic generation feel + ensures
         the response is fully ready).
      3. Build and return PDF bytes via reportlab.

    Falls back to a deterministic narrative if Gemini is unavailable.
    """
    llm_note = ""
    try:
        narrative = _call_gemini(_build_prompt(payload))
    except RuntimeError as exc:
        llm_note  = str(exc)
        narrative = _fallback_text(payload, llm_note)
        print(f"[LLM] Falling back to deterministic document. Reason: {llm_note}")
    except Exception as exc:
        llm_note  = f"Unexpected error: {traceback.format_exc()}"
        narrative = _fallback_text(payload, llm_note)
        print(f"[LLM] Unexpected error, using fallback:\n{llm_note}")

    # Hold before serving — gives time for the LLM to be perceived as "working"
    print(f"[LLM] Waiting {PRE_SEND_SLEEP}s before serving document …")
    time.sleep(PRE_SEND_SLEEP)

    print("[LLM] Building PDF …")
    pdf_bytes = build_pdf(payload, narrative)
    print(f"[LLM] PDF ready ({len(pdf_bytes):,} bytes).")
    return pdf_bytes