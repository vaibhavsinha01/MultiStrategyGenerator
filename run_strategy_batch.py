"""
Run the strategy pipeline across multiple crypto markets and append results
into unified CSV datasets.

Example:
  python run_strategy_batch.py --n 1000000 --top 1000 --workers 13
"""
import argparse
import multiprocessing as mp
from pathlib import Path

from main import run


DEFAULT_SYMBOLS = ("btcusdt","ethusdt","xrpusdt","solusdt")
# DEFAULT_SYMBOLS = ("solusdt","_")
DEFAULT_TIMEFRAMES = ("15m", "30m", "1h", "4h", "1d")
# DEFAULT_TIMEFRAMES = ("1h","4h","1d")

def parse_csv_list(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append strategy results for multiple symbols/timeframes into unified CSVs."
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default=r"results\strategy_results_unified.csv")
    parser.add_argument("--n", type=int, default=100000, help="Strategies to generate per run")
    parser.add_argument("--top", type=int, default=1000, help="Top N to validate per run")
    parser.add_argument("--workers", type=int, default=13)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--overwrite-first",
        action="store_true",
        help="Overwrite unified CSVs for the first completed run, then append the rest.",
    )
    args = parser.parse_args()

    symbols = parse_csv_list(args.symbols)
    timeframes = parse_csv_list(args.timeframes)
    data_dir = Path(args.data_dir)
    output_csv = str(Path(args.out))

    completed = 0
    total = len(symbols) * len(timeframes)
    for symbol in symbols:
        for timeframe in timeframes:
            csv_path = data_dir / f"{symbol}_{timeframe}.csv"
            if not csv_path.exists():
                print(f"Skipping missing data file: {csv_path}")
                continue

            completed += 1
            pair_seed = args.seed + completed - 1
            append_results = not (args.overwrite_first and completed == 1)
            print(
                f"\n[{completed}/{total}] Running {symbol.upper()} {timeframe} "
                f"(seed={pair_seed}, append={append_results})"
            )
            run(
                csv_path=str(csv_path),
                n_strategies=args.n,
                top_n=args.top,
                n_workers=args.workers,
                output_csv=output_csv,
                seed=pair_seed,
                append_results=append_results,
            )

    print(f"\nBatch complete. Unified outputs use base path: {output_csv}")

if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
