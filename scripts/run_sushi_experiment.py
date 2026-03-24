#!/usr/bin/env python3
"""Sushi experiment runner."""

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

from grums.experiments.sushi import compare_criteria_sushi_choice
from grums.inference import MCEMConfig
import yaml

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

DEFAULTS = {
    "criterion": "social",
    "dataset_path": ".data",
    "repeats": 3,
    "rounds": 20,
    "seed": 0,
    "n_jobs": 1,
    "iterations": 6,
    "gibbs_samples": 25,
    "gibbs_burnin": 12,
    "sigma": 1.0,
    "prior_precision": 1e-2,
    "tolerance": 1e-5,
    "random_seed": 0,
    "metric": "social",
    "output_json": "",
    "quiet": False,
    "no_progress": False,
}

def _normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = set(DEFAULTS.keys())
    unknown = set(raw.keys()) - allowed
    if unknown: raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    normalized = dict(raw)
    return normalized

def _load_config_file(path: str) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not loaded: return {}
    return _normalize_config(loaded)

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _log(enabled: bool, message: str) -> None:
    if enabled: print(f"[{_utc_now_iso()}] {message}", file=sys.stderr, flush=True)

def main(argv: list[str] | None = None) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args(argv)

    defaults = dict(DEFAULTS)
    if pre_args.config:
        defaults.update(_load_config_file(pre_args.config))

    parser = argparse.ArgumentParser(description="Sushi runner")
    parser.add_argument("--config", type=str, default=pre_args.config)
    parser.add_argument("--criterion", choices=["random", "d_opt", "e_opt", "social", "personalized"], default=defaults["criterion"])
    parser.add_argument("--dataset-path", type=str, default=defaults["dataset_path"])
    parser.add_argument("--repeats", type=int, default=defaults["repeats"])
    parser.add_argument("--rounds", type=int, default=defaults["rounds"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--n-jobs", type=int, default=defaults["n_jobs"])

    parser.add_argument("--iterations", type=int, default=defaults["iterations"])
    parser.add_argument("--gibbs-samples", type=int, default=defaults["gibbs_samples"])
    parser.add_argument("--gibbs-burnin", type=int, default=defaults["gibbs_burnin"])
    parser.add_argument("--sigma", type=float, default=defaults["sigma"])
    parser.add_argument("--prior-precision", type=float, default=defaults["prior_precision"])
    parser.add_argument("--tolerance", type=float, default=defaults["tolerance"])
    parser.add_argument("--random-seed", type=int, default=defaults["random_seed"])
    parser.add_argument("--metric", choices=["social", "mean_person", "raw_person"], default=defaults["metric"])

    parser.add_argument("--output-json", type=str, default=defaults["output_json"])
    parser.add_argument("--quiet", action="store_true", default=bool(defaults["quiet"]))
    parser.add_argument("--no-progress", action="store_true", default=bool(defaults["no_progress"]))
    args = parser.parse_args(argv)

    log_enabled = not args.quiet
    run_start = perf_counter()
    _log(log_enabled, "Sushi experiment run started")

    cfg = MCEMConfig(
        n_iterations=args.iterations,
        n_gibbs_samples=args.gibbs_samples,
        n_gibbs_burnin=args.gibbs_burnin,
        sigma=args.sigma,
        prior_precision=args.prior_precision,
        tolerance=args.tolerance,
        random_seed=args.random_seed,
    )

    payload = {
        "config": asdict(cfg),
        "config_file": args.config if args.config else None,
        "dataset_path": args.dataset_path,
        "seed": args.seed,
        "rounds": args.rounds,
        "repeats": args.repeats,
        "n_jobs": args.n_jobs,
        "started_at_utc": _utc_now_iso(),
    }

    timing = {}

    _log(log_enabled, f"Criteria phase started (criterion={args.criterion}, rounds={args.rounds}, repeats={args.repeats})")
    t0 = perf_counter()

    pbar_total = args.repeats
    pbar = None
    if not args.no_progress and tqdm is not None:
        pbar = tqdm(total=pbar_total, desc="Criteria", unit="rep")

    def _update(delta: int) -> None:
        if pbar is not None:
            pbar.update(delta)

    try:
        score = compare_criteria_sushi_choice(
            dataset_path=args.dataset_path,
            n_rounds=args.rounds,
            repeats=args.repeats,
            criterion_name=args.criterion,
            seed=args.seed,
            mcem_config=cfg,
            n_jobs=args.n_jobs,
            progress_update=_update,
        )
    finally:
        if pbar is not None:
            pbar.close()

    criteria_seconds = perf_counter() - t0
    timing["criteria_seconds"] = criteria_seconds
    payload["criteria"] = {args.criterion: score}

    _log(log_enabled, f"Criteria phase finished in {criteria_seconds:.2f}s (social_tau={score['social']:.4f})")

    total_seconds = perf_counter() - run_start
    timing["total_seconds"] = total_seconds
    payload["timing"] = timing
    payload["finished_at_utc"] = _utc_now_iso()

    text = json.dumps(payload, indent=2)
    print(text)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        _log(log_enabled, f"Results written to {args.output_json}")

if __name__ == "__main__":
    main()
