"""
generator.py

Generate random but valid trading strategies.

Validity rules
  1. Use 2, 3, or 4 entry conditions.
  2. Entry signals must be compatible with the strategy side.
     Bull strategies can include bullish continuation or oversold
     mean-reversion signals; bear strategies can include bearish continuation
     or overbought mean-reversion signals.
  3. Neutral signals may be added to either direction.
  4. No two signals from the same contradiction group in one strategy.
  5. TP/SL ranges are kept intraday-friendly so trades can close and re-enter.
"""

import random
import uuid

from signals import SIGNALS, BULL_SIGNALS, BEAR_SIGNALS, NEUTRAL_SIGNALS


VALID_SIGNALS = set(SIGNALS.keys())

SIGNAL_COUNT_WEIGHTS = ((2, 0.10), (3, 0.60), (4, 0.25), (5,0.05))
NEUTRAL_SLOT_PROBABILITY = 0.04
QUALITY_POOL_PROBABILITY = 0.80

TP_RANGE = (0.008, 0.080)
SL_RANGE = (0.004, 0.040)
MIN_REWARD_RISK = 1.25

QUALITY_BULL = [s for s in BULL_SIGNALS if s in VALID_SIGNALS]
QUALITY_BEAR = [s for s in BEAR_SIGNALS if s in VALID_SIGNALS]
QUALITY_NEUTRAL = [s for s in NEUTRAL_SIGNALS if s in VALID_SIGNALS]


def _pick_signal_count() -> int:
    counts, weights = zip(*SIGNAL_COUNT_WEIGHTS, strict=True)
    return random.choices(counts, weights=weights)[0]


def _pick_pool(direction: str, quality_only: bool) -> list[str]:
    if quality_only:
        return QUALITY_BULL if direction == "bull" else QUALITY_BEAR
    return BULL_SIGNALS if direction == "bull" else BEAR_SIGNALS


def _valid_combo(selected: list[str]) -> bool:
    """Return True if no two selected signals share a contradiction group."""
    groups = [SIGNALS[s]["group"] for s in selected]
    return len(groups) == len(set(groups))


def _neutral_candidates(selected: list[str], quality_only: bool) -> list[str]:
    used_groups = {SIGNALS[s]["group"] for s in selected}
    pool = QUALITY_NEUTRAL if quality_only else NEUTRAL_SIGNALS
    return [s for s in pool if SIGNALS[s]["group"] not in used_groups]


def _pick_risk() -> tuple[float, float]:
    tp = round(random.uniform(*TP_RANGE), 10)
    sl = round(random.uniform(*SL_RANGE), 10)
    return round(tp,2), round(sl,2)


def generate_one(direction: str | None = None, max_attempts: int = 200) -> dict | None:
    """Generate one valid strategy dict, or None if no valid combo is found."""
    if direction is None:
        direction = random.choice(["bull", "bear"])

    n_signals = _pick_signal_count()

    for _ in range(max_attempts):
        use_quality = random.random() < QUALITY_POOL_PROBABILITY
        pool = _pick_pool(direction, use_quality)

        wants_neutral = n_signals > 2 and random.random() < NEUTRAL_SLOT_PROBABILITY
        n_directional = n_signals - 1 if wants_neutral else n_signals

        if len(pool) < n_directional:
            continue

        selected = random.sample(pool, n_directional)
        if not all(s in VALID_SIGNALS for s in selected):
            continue

        if wants_neutral:
            neutral_pool = _neutral_candidates(selected, use_quality)
            if not neutral_pool:
                continue
            selected.append(random.choice(neutral_pool))

        if not _valid_combo(selected):
            continue

        tp, sl = _pick_risk()
        if tp / sl < MIN_REWARD_RISK:
            continue

        return {
            "id": uuid.uuid4().hex[:8],
            "direction": direction,
            "signals": selected,
            "n_signals": len(selected),
            "tp": tp,
            "sl": sl,
        }

    return None


def generate_strategies(
    n: int = 750,
    bull_ratio: float = 0.5,
    seed: int | None = 42,
) -> list[dict]:
    """Generate n unique valid strategies."""
    if seed is not None:
        random.seed(seed)

    strategies: list[dict] = []
    seen: set[frozenset] = set()

    n_bull = int(n * bull_ratio)
    n_bear = n - n_bull

    for direction, count in [("bull", n_bull), ("bear", n_bear)]:
        attempts = 0
        generated = 0
        max_total_attempts = count * 12

        while generated < count and attempts < max_total_attempts:
            attempts += 1
            strat = generate_one(direction)
            if strat is None:
                continue

            key = frozenset(strat["signals"]) | {strat["direction"]}
            if key in seen:
                continue

            seen.add(key)
            strategies.append(strat)
            generated += 1

    random.shuffle(strategies)
    return strategies
