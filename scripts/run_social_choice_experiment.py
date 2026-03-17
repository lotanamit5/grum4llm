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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grums.experiments.benchmark import compare_criteria_social_choice, run_asymptotic_social_choice
from grums.inference import MCEMConfig
import yaml

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback if tqdm is unavailable
    tqdm = None


def _parse_agent_counts(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    counts = [int(p) for p in parts]
    if not counts:
        raise ValueError("agent-counts must contain at least one integer")
    if any(c <= 0 for c in counts):
        raise ValueError("all agent-counts values must be positive")
    return counts


DEFAULTS: dict[str, Any] = {
    "mode": "both",
    "agent_counts": "10,20,30",
    "repeats": 3,
    "rounds": 20,
    "seed": 0,
    "iterations": 6,
    "gibbs_samples": 25,
    "gibbs_burnin": 12,
    "sigma": 1.0,
    "prior_precision": 1e-2,
    "tolerance": 1e-5,
    "random_seed": 0,
    "output_json": "",
    "quiet": False,
    "no_progress": False,
}


def _normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = set(DEFAULTS.keys())
    unknown = set(raw.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    normalized = dict(raw)
    if "agent_counts" in normalized:
        counts = normalized["agent_counts"]
        if isinstance(counts, list):
            normalized["agent_counts"] = ",".join(str(v) for v in counts)
        elif not isinstance(counts, str):
            raise ValueError("agent_counts must be a comma-separated string or list of integers")

    if "mode" in normalized and normalized["mode"] not in {"asymptotic", "criteria", "both"}:
        raise ValueError("mode must be one of: asymptotic, criteria, both")

    for key in ["repeats", "rounds", "seed", "iterations", "gibbs_samples", "gibbs_burnin", "random_seed"]:
        if key in normalized and not isinstance(normalized[key], int):
            raise ValueError(f"{key} must be an integer")

    for key in ["sigma", "prior_precision", "tolerance"]:
        if key in normalized and not isinstance(normalized[key], (float, int)):
            raise ValueError(f"{key} must be numeric")

    for key in ["quiet", "no_progress"]:
        if key in normalized and not isinstance(normalized[key], bool):
            raise ValueError(f"{key} must be boolean")

    return normalized


def _load_config_file(path: str) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a YAML mapping")
    return _normalize_config(loaded)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[{_utc_now_iso()}] {message}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="", help="Path to YAML run config")
    pre_args, _ = pre.parse_known_args(argv)

    defaults = dict(DEFAULTS)
    if pre_args.config:
        defaults.update(_load_config_file(pre_args.config))

    parser = argparse.ArgumentParser(description="Run synthetic social-choice GRUM experiments.")
    parser.add_argument("--config", type=str, default=pre_args.config, help="Path to YAML run config")
    parser.add_argument("--mode", choices=["asymptotic", "criteria", "both"], default=defaults["mode"])
    parser.add_argument("--agent-counts", default=defaults["agent_counts"])
    parser.add_argument("--repeats", type=int, default=defaults["repeats"])
    parser.add_argument("--rounds", type=int, default=defaults["rounds"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])

    parser.add_argument("--iterations", type=int, default=defaults["iterations"])
    parser.add_argument("--gibbs-samples", type=int, default=defaults["gibbs_samples"])
    parser.add_argument("--gibbs-burnin", type=int, default=defaults["gibbs_burnin"])
    parser.add_argument("--sigma", type=float, default=defaults["sigma"])
    parser.add_argument("--prior-precision", type=float, default=defaults["prior_precision"])
    parser.add_argument("--tolerance", type=float, default=defaults["tolerance"])
    parser.add_argument("--random-seed", type=int, default=defaults["random_seed"])

    parser.add_argument("--output-json", type=str, default=defaults["output_json"])
    parser.add_argument("--quiet", action="store_true", default=bool(defaults["quiet"]), help="Disable progress logs.")
    parser.add_argument("--no-progress", action="store_true", default=bool(defaults["no_progress"]), help="Disable tqdm progress bars.")
    args = parser.parse_args(argv)

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
        "config_file": args.config if args.config else None,
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
        asym_total_units = len(counts) * args.repeats
        asym_bar = None
        if not args.no_progress and tqdm is not None:
            asym_bar = tqdm(total=asym_total_units, desc="Asymptotic", unit="unit")

        def _asym_update(delta: int) -> None:
            if asym_bar is not None:
                asym_bar.update(delta)

        points = run_asymptotic_social_choice(
            agent_counts=counts,
            repeats=args.repeats,
            seed=args.seed,
            mcem_config=cfg,
            progress_update=_asym_update,
        )
        if asym_bar is not None:
            asym_bar.close()
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
        criteria_total_units = args.rounds * args.repeats * 4
        criteria_bar = None
        if not args.no_progress and tqdm is not None:
            criteria_bar = tqdm(total=criteria_total_units, desc="Criteria", unit="round")

        def _criteria_update(delta: int) -> None:
            if criteria_bar is not None:
                criteria_bar.update(delta)

        scores = compare_criteria_social_choice(
            n_rounds=args.rounds,
            repeats=args.repeats,
            seed=args.seed,
            mcem_config=cfg,
            progress_update=_criteria_update,
        )
        if criteria_bar is not None:
            criteria_bar.close()
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
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        _log(log_enabled, f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
