"""Command-line entry point: `g73-uba` (or `python -m g73uba`)."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from . import __version__


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="g73-uba",
        description="G73 prototype: UBA + ML detection of a compromised-credential "
                    "ransomware intrusion (Change Healthcare 2024 replay, synthetic data).")
    ap.add_argument("--seed", type=int, default=7, help="random seed for the simulation (default 7)")
    ap.add_argument("--fpr", type=float, default=0.002,
                    help="false-positive budget used to set the alert threshold (default 0.002 = 0.2%%)")
    ap.add_argument("--out", default="output", help="output folder (default ./output)")
    ap.add_argument("--open", action="store_true", help="open the HTML report when done")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    from .pipeline import run
    from .report import write_report

    result = run(seed=args.seed, fpr_budget=args.fpr, out_dir=args.out)
    report = write_report(result, Path(args.out) / "report.html")
    s = result["summary"]
    e = s["metrics"]["ensemble"]
    print()
    print(f"Ensemble recall {e['recall']:.1%}  FPR {e['fpr']:.4%}  "
          f"first true alert {s['lead_time_hours']} h before encryption")
    print(f"Report: {report.resolve()}")
    if args.open:
        webbrowser.open(report.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
