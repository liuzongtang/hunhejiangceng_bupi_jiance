"""
Re-render the reward training figure from an existing Ultralytics run directory.

Usage:
    python scripts/plot_reward_training.py --dir runs/rtdetr-reward/reward_on
    python scripts/plot_reward_training.py --dir runs/rtdetr-reward/reward_on --window 100

Reads ``results.csv`` (supervised losses + val mAP) and ``reward_history.csv``
(per-step reward stats) and writes ``reward_training.png`` into the run dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reward_viz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Ultralytics run directory")
    parser.add_argument(
        "--window", type=int, default=50, help="Rolling-mean window for reward curves"
    )
    args = parser.parse_args()

    out = reward_viz.plot_reward_training(args.dir, window=args.window)
    if out is None:
        print(
            f"[plot] nothing to plot in {args.dir} (missing results.csv / reward_history.csv)"
        )
        return 1
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
