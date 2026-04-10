"""
generator.py
────────────
Generates random but valid trading strategies as dicts.

Validity rules
  1. 2-3 entry conditions combined with AND
  2. All entry signals must be the same direction (bull or bear)
     neutral signals (vol spike, squeeze, ATR) may be added as a
     3rd condition to either direction.
  3. No two signals from the same contradiction group in one strategy.
  4. TP: 1%-2%,  SL: 0.5%-1%
  5. Duplicate detection: same frozenset of signals → skip.
"""

import random
import uuid
from signals import SIGNALS, BULL_SIGNALS, BEAR_SIGNALS, NEUTRAL_SIGNALS


# ── High-signal-quality pools ──────────────────────────────────────────────────
# Manually curated subsets that are most alpha-relevant for crypto 15m data.
# Generator draws preferentially from these (80% of picks) before falling
# back to the full pool.

# QUALITY_BULL = [
#     "s1","s3","s5","s7","s9","s13","s15","s17","s19",
#     "s21","s23","s25","s27","s29","s31","s33","s35",
#     "s41","s43","s45","s49","s51","s55","s62","s66",
#     "s73","s75","s78","s82","s84","s86","s89","s93",
#     "s95","s98","s100",
# ]
# QUALITY_BEAR = [
#     "s2","s4","s6","s8","s10","s14","s16","s18","s20",
#     "s22","s24","s26","s28","s30","s32","s34","s36",
#     "s42","s44","s46","s50","s52","s56","s63","s67",
#     "s74","s76","s79","s83","s85","s87","s90","s94",
#     "s99",
# ]
# QUALITY_NEUTRAL = ["s71","s72","s77","s88"]

VALID_SIGNALS = set(SIGNALS.keys())

QUALITY_BULL = [s for s in BULL_SIGNALS if s in VALID_SIGNALS]
QUALITY_BEAR = [s for s in BEAR_SIGNALS if s in VALID_SIGNALS]
QUALITY_NEUTRAL = [s for s in NEUTRAL_SIGNALS if s in VALID_SIGNALS]


def _pick_pool(direction: str, quality_only: bool) -> list:
    if quality_only:
        return QUALITY_BULL if direction == "bull" else QUALITY_BEAR
    return BULL_SIGNALS if direction == "bull" else BEAR_SIGNALS

def _valid_combo(selected: list[str]) -> bool:
    """Return True if no two selected signals share a contradiction group."""
    # groups = [SIGNALS[s]["group"] for s in selected]
    groups = [SIGNALS[s]["group"] for s in selected]
    return len(groups) == len(set(groups))


def generate_one(direction: str | None = None, max_attempts: int = 200) -> dict | None:
    """
    Generate one valid strategy dict.

    Returns None if a valid combo cannot be found within max_attempts.

    Strategy dict schema
    ────────────────────
    {
        "id"        : str  (uuid4 short)
        "direction" : "bull" | "bear"
        "signals"   : [str, ...]   e.g. ["s1", "s21", "s77"]
        "n_signals" : int  (2 or 3)
        "tp"        : float  (e.g. 0.015)
        "sl"        : float  (e.g. 0.007)
    }
    """
    if direction is None:
        direction = random.choice(["bull", "bear"])

    n_signals = random.choices([3, 4, 5, 6, 7], weights=[0.20, 0.30, 0.25, 0.15, 0.10])[0] # initial params are 2,3 and 0.45,0.55

    for _ in range(max_attempts):
        # 80% quality pool, 20% full pool
        use_quality = random.random() < 0.80
        pool = _pick_pool(direction, use_quality)

        # Sample the directional signals
        n_dir = n_signals if random.random() < 0.7 else n_signals - 1
        n_dir = max(2, n_dir)                       # always at least 2 directional

        if len(pool) < n_dir:
            continue
        # selected = random.sample(pool, n_dir)
        valid_keys = set(SIGNALS.keys())

        selected = random.sample(pool, n_dir)

        # ✅ VALIDATION HERE
        if not all(s in valid_keys for s in selected):
            continue

        # Optionally add a neutral signal as the 3rd condition
        if n_signals == 3 and n_dir == 2:
            neutral_pool = QUALITY_NEUTRAL if use_quality else NEUTRAL_SIGNALS
            neutral_pool = [s for s in neutral_pool
                            if SIGNALS[s]["group"] not in
                            [SIGNALS[x]["group"] for x in selected]]
            if neutral_pool:
                # selected.append(random.choice(neutral_pool))
                selected.append(random.choice(neutral_pool))

                # ✅ VALIDATION AGAIN (important)
                if not all(s in valid_keys for s in selected):
                    continue
            else:
                continue                            # can't find a valid neutral — retry

        if not _valid_combo(selected):
            continue

        tp = round(random.uniform(0.018, 0.080), 10)   # 0.5% – 4.0% # changed and initial params were 0.010-0.020 , 4
        sl = round(random.uniform(0.008, 0.020), 10)   # 0.2% – 2.0% # changed and initial params were 0.005-0.010 , 4

        if tp/sl < 1.5:
            continue

        return {
            "id":        uuid.uuid4().hex[:8],
            "direction": direction,
            "signals":   selected,
            "n_signals": len(selected),
            "tp":        tp,
            "sl":        sl,
        }

    return None   # failed to generate a valid combo


def generate_strategies(
    n: int = 750,
    bull_ratio: float = 0.5,
    seed: int | None = 42,
) -> list[dict]:
    """
    Generate `n` unique valid strategies.

    Parameters
    ----------
    n          : total strategies to generate
    bull_ratio : fraction that are long strategies
    seed       : random seed for reproducibility (None = random)

    Returns
    -------
    List of strategy dicts (unique by signal frozenset + direction).
    """
    if seed is not None:
        random.seed(seed)

    strategies: list[dict] = []
    seen: set[frozenset] = set()            # deduplication key

    n_bull = int(n * bull_ratio)
    n_bear = n - n_bull

    for direction, count in [("bull", n_bull), ("bear", n_bear)]:
        attempts = 0
        generated = 0
        max_total_attempts = count * 30     # safety ceiling

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

    # Shuffle so bull/bear aren't batched
    random.shuffle(strategies)
    return strategies