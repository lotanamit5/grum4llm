#!/usr/bin/env python3
"""Quick social-choice synthetic experiment runner.

This script is intentionally lightweight and can be refactored later into
an experiment CLI framework.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grums.experiments.benchmark import compare_criteria_social_choice, run_asymptotic_social_choice
from grums.inference import MCEMConfig


def _parse_agent_counts(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    counts = [int(p) for p in parts]
    if not counts:
        raise ValueError("agent-counts must contain at least one integer")
    if any(c <= 0 for c in counts):
        raise ValueError("all agent-counts values must be positive")
    return counts


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[{_utc_now_iso()}] {message}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic social-choice GRUM experiments.")
    parser.add_argument("--mode", choices=["asymptotic", "criteria", "both"], default="both")
    parser.add_argument("--agent-counts", default="10,20,30")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--gibbs-samples", type=int, default=25)
    parser.add_argument("--gibbs-burnin", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--prior-precision", type=float, default=1e-2)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--random-seed", type=int, default=0)

    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logs.")
    args = parser.parse_args()

    log_enabled = not args.quiet
    run_start = perf_counter()
    _log(log_enabled, "Experiment run started")

    cfg = MCEMConfig(
        n_iterations=args.iterations,
        n_gibbs_samples=args.gibbs_samples,
        n_gibbs_burnin=args.gibbs_burnin,
        sigma=args.sigma,
        prior_precision=args.prior_precision,
        tolerance=args.tolerance,
        random_seed=args.random_seed,
    )

    payload: dict[str, object] = {
        "config": asdict(cfg),
        "seed": args.seed,
        "repeats": args.repeats,
        "started_at_utc": _utc_now_iso(),
    }

    timing: dict[str, float] = {}

    counts: list[int] = []
    if args.mode in ("asymptotic", "both"):
        counts = _parse_agent_counts(args.agent_counts)
        _log(
            log_enabled,
            (
                "Asymptotic phase started "
                f"(agent_counts={counts}, repeats={args.repeats}, iterations={args.iterations}, "
                f"gibbs_samples={args.gibbs_samples}, gibbs_burnin={args.gibbs_burnin})"
            ),
        )
        t0 = perf_counter()
        points = run_asymptotic_social_choice(
            agent_counts=counts,
            repeats=args.repeats,
            seed=args.seed,
            mcem_config=cfg,
        )
        asym_seconds = perf_counter() - t0
        timing["asymptotic_seconds"] = asym_seconds
        payload["asymptotic"] = [asdict(p) for p in points]

        asym_work_units = max(1, len(counts) * args.repeats)
        _log(
            log_enabled,
            (
                f"Asymptotic phase finished in {asym_seconds:.2f}s "
                f"({asym_seconds / asym_work_units:.2f}s per count-repeat unit)"
            ),
        )

    if args.mode in ("criteria", "both"):
        _log(
            log_enabled,
            (
                "Criteria phase started "
                f"(rounds={args.rounds}, repeats={args.repeats}, criteria=4, iterations={args.iterations})"
            ),
        )
        t0 = perf_counter()
        scores = compare_criteria_social_choice(
            n_rounds=args.rounds,
            repeats=args.repeats,
            seed=args.seed,
            mcem_config=cfg,
        )
        criteria_seconds = perf_counter() - t0
        timing["criteria_seconds"] = criteria_seconds
        payload["criteria"] = scores

        criteria_work_units = max(1, args.rounds * args.repeats * 4)
        _log(
            log_enabled,
            (
                f"Criteria phase finished in {criteria_seconds:.2f}s "
                f"({criteria_seconds / criteria_work_units:.4f}s per criterion-round unit)"
            ),
        )

    total_seconds = perf_counter() - run_start
    timing["total_seconds"] = total_seconds
    payload["timing"] = timing
    payload["finished_at_utc"] = _utc_now_iso()

    _log(log_enabled, f"Experiment run finished in {total_seconds:.2f}s")
    if args.mode in ("asymptotic", "both") and counts:
        _log(
            log_enabled,
            (
                "Runtime estimate hint: if you double both repeats and agent-count points, "
                "asymptotic runtime is expected to roughly double."
            ),
        )
    if args.mode in ("criteria", "both"):
        _log(
            log_enabled,
            (
                "Runtime estimate hint: criteria runtime scales approximately linearly with "
                "rounds x repeats x number_of_criteria."
            ),
        )

    text = json.dumps(payload, indent=2)
    print(text)

    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
        _log(log_enabled, f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
