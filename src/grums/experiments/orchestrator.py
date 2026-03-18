"""Experiment orchestration utilities for multi-run reproducible sweeps."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

import yaml


@dataclass(frozen=True)
class SubrunSpec:
    run_index: int
    seed: int
    config_path: Path
    output_path: Path
    log_path: Path


@dataclass(frozen=True)
class SubrunResult:
    run_index: int
    seed: int
    config_path: Path
    output_path: Path
    log_path: Path
    command: tuple[str, ...]
    status: str
    exit_code: int
    duration_seconds: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in prefix.strip())
    return f"{sanitized}-{stamp}"


def _parse_seed_values(seed_cfg: Any) -> list[int]:
    if isinstance(seed_cfg, list):
        out = [int(v) for v in seed_cfg]
    elif isinstance(seed_cfg, str):
        out = [int(v.strip()) for v in seed_cfg.split(",") if v.strip()]
    elif isinstance(seed_cfg, dict):
        start = int(seed_cfg.get("start", 0))
        stop = int(seed_cfg.get("stop", 0))
        step = int(seed_cfg.get("step", 1))
        if step <= 0:
            raise ValueError("sweep.seeds.step must be positive")
        if stop < start:
            raise ValueError("sweep.seeds.stop must be >= start")
        out = list(range(start, stop + 1, step))
    else:
        raise ValueError("sweep.seeds must be list, csv string, or mapping with start/stop/step")

    if not out:
        raise ValueError("sweep.seeds must contain at least one value")
    return out


def load_orchestration_config(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Orchestration config must be a YAML mapping")

    required = {"run_prefix", "output_root", "base_run", "sweep"}
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {sorted(missing)}")

    base_run = raw.get("base_run")
    if not isinstance(base_run, dict):
        raise ValueError("base_run must be a mapping")

    sweep = raw.get("sweep")
    if not isinstance(sweep, dict):
        raise ValueError("sweep must be a mapping")

    seeds = _parse_seed_values(sweep.get("seeds", []))

    aggregations = raw.get("aggregations", ["asymptotic", "criteria", "timing"])
    if not isinstance(aggregations, list) or not all(isinstance(v, str) for v in aggregations):
        raise ValueError("aggregations must be a list of strings")

    max_parallel = int(raw.get("max_parallel_subprocesses", 1))
    if max_parallel <= 0:
        raise ValueError("max_parallel_subprocesses must be positive")

    worker_script = str(raw.get("worker_script", "scripts/run_social_choice_experiment.py"))

    return {
        "run_prefix": str(raw["run_prefix"]),
        "output_root": str(raw["output_root"]),
        "worker_script": worker_script,
        "base_run": dict(base_run),
        "seed_values": seeds,
        "max_parallel_subprocesses": max_parallel,
        "aggregations": aggregations,
    }


def create_run_folder(output_root: Path, run_prefix: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = _safe_run_id(run_prefix)
    run_dir = output_root / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{run_id}-{suffix}"
        suffix += 1

    for child in ["subconfigs", "outputs", "aggregates", "logs"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def build_subrun_specs(run_dir: Path, base_run: dict[str, Any], seeds: list[int]) -> list[SubrunSpec]:
    specs: list[SubrunSpec] = []

    for idx, seed in enumerate(seeds):
        out_path = run_dir / "outputs" / f"run_{idx:03d}_seed_{seed}.json"
        cfg_path = run_dir / "subconfigs" / f"run_{idx:03d}_seed_{seed}.yml"
        log_path = run_dir / "logs" / f"run_{idx:03d}_seed_{seed}.log"

        sub_cfg = dict(base_run)
        sub_cfg["seed"] = seed
        sub_cfg["output_json"] = str(out_path)

        cfg_path.write_text(yaml.safe_dump(sub_cfg, sort_keys=False), encoding="utf-8")
        specs.append(
            SubrunSpec(
                run_index=idx,
                seed=seed,
                config_path=cfg_path,
                output_path=out_path,
                log_path=log_path,
            )
        )

    return specs


def _run_subprocess_spec(spec: SubrunSpec, worker_script: Path, workspace_root: Path) -> SubrunResult:
    started = datetime.now(timezone.utc)
    cmd = (sys.executable, str(worker_script), "--config", str(spec.config_path))
    proc = subprocess.run(
        cmd,
        cwd=workspace_root,
        capture_output=True,
        text=True,
    )
    finished = datetime.now(timezone.utc)
    duration_seconds = (finished - started).total_seconds()

    log_text = "\n".join(
        [
            f"started_at_utc: {started.isoformat(timespec='seconds')}",
            f"finished_at_utc: {finished.isoformat(timespec='seconds')}",
            f"duration_seconds: {duration_seconds:.6f}",
            f"exit_code: {proc.returncode}",
            f"command: {' '.join(cmd)}",
            "",
            "[stdout]",
            proc.stdout,
            "",
            "[stderr]",
            proc.stderr,
        ]
    )
    spec.log_path.write_text(log_text, encoding="utf-8")

    status = "success" if proc.returncode == 0 else "failed"
    return SubrunResult(
        run_index=spec.run_index,
        seed=spec.seed,
        config_path=spec.config_path,
        output_path=spec.output_path,
        log_path=spec.log_path,
        command=cmd,
        status=status,
        exit_code=proc.returncode,
        duration_seconds=duration_seconds,
    )


def _json_payloads_from_paths(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            payloads.append((path, data))
    return payloads


def aggregate_asymptotic(payloads: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    grouped: dict[int, list[float]] = {}

    for file_path, payload in payloads:
        seed = payload.get("seed")
        for pt in payload.get("asymptotic", []):
            n_agents = int(pt["n_agents"])
            value = float(pt["mean_tau"])
            rows.append(
                {
                    "file": file_path.name,
                    "seed": seed,
                    "n_agents": n_agents,
                    "mean_tau": value,
                }
            )
            grouped.setdefault(n_agents, []).append(value)

    summary: list[dict[str, Any]] = []
    for n_agents in sorted(grouped.keys()):
        values = grouped[n_agents]
        summary.append(
            {
                "n_agents": n_agents,
                "count": len(values),
                "mean": float(mean(values)),
                "std": float(pstdev(values)) if len(values) > 1 else 0.0,
                "min": float(min(values)),
                "max": float(max(values)),
            }
        )

    return {"rows": rows, "summary": summary}


def aggregate_criteria(payloads: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = {}

    for file_path, payload in payloads:
        seed = payload.get("seed")
        criteria = payload.get("criteria", {})
        if not isinstance(criteria, dict):
            continue
        for name, score in criteria.items():
            value = float(score)
            rows.append(
                {
                    "file": file_path.name,
                    "seed": seed,
                    "criterion": str(name),
                    "score": value,
                }
            )
            grouped.setdefault(str(name), []).append(value)

    summary: list[dict[str, Any]] = []
    for criterion in sorted(grouped.keys()):
        values = grouped[criterion]
        summary.append(
            {
                "criterion": criterion,
                "count": len(values),
                "mean": float(mean(values)),
                "std": float(pstdev(values)) if len(values) > 1 else 0.0,
                "min": float(min(values)),
                "max": float(max(values)),
            }
        )

    return {"rows": rows, "summary": summary}


def aggregate_timing(payloads: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    totals: list[float] = []

    for file_path, payload in payloads:
        timing = payload.get("timing", {})
        if not isinstance(timing, dict):
            continue
        row = {
            "file": file_path.name,
            "seed": payload.get("seed"),
            "asymptotic_seconds": float(timing.get("asymptotic_seconds", 0.0)),
            "criteria_seconds": float(timing.get("criteria_seconds", 0.0)),
            "total_seconds": float(timing.get("total_seconds", 0.0)),
        }
        rows.append(row)
        totals.append(row["total_seconds"])

    summary = {
        "count": len(totals),
        "total_mean_seconds": float(mean(totals)) if totals else 0.0,
        "total_std_seconds": float(pstdev(totals)) if len(totals) > 1 else 0.0,
        "total_min_seconds": float(min(totals)) if totals else 0.0,
        "total_max_seconds": float(max(totals)) if totals else 0.0,
    }

    return {"rows": rows, "summary": summary}


AGGREGATION_FUNCTIONS: dict[str, Callable[[list[tuple[Path, dict[str, Any]]]], dict[str, Any]]] = {
    "asymptotic": aggregate_asymptotic,
    "criteria": aggregate_criteria,
    "timing": aggregate_timing,
}


def run_aggregations_for_json_paths(
    json_paths: list[Path],
    output_dir: Path,
    aggregation_names: list[str],
) -> dict[str, Path]:
    payloads = _json_payloads_from_paths(json_paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for name in aggregation_names:
        fn = AGGREGATION_FUNCTIONS.get(name)
        if fn is None:
            raise ValueError(f"Unknown aggregation function: {name}")
        out_data = fn(payloads)
        out_path = output_dir / f"{name}.json"
        out_path.write_text(json.dumps(out_data, indent=2) + "\n", encoding="utf-8")
        written[name] = out_path

    return written


def run_aggregations_for_run_folder(run_dir: Path, aggregation_names: list[str]) -> dict[str, Path]:
    outputs_dir = run_dir / "outputs"
    json_paths = sorted(outputs_dir.glob("*.json"))
    return run_aggregations_for_json_paths(
        json_paths=json_paths,
        output_dir=run_dir / "aggregates",
        aggregation_names=aggregation_names,
    )


def run_orchestration(orchestration_config_path: Path, workspace_root: Path) -> Path:
    loaded = load_orchestration_config(orchestration_config_path)

    run_dir = create_run_folder(
        output_root=(workspace_root / loaded["output_root"]).resolve(),
        run_prefix=loaded["run_prefix"],
    )

    worker_script = (workspace_root / loaded["worker_script"]).resolve()
    if not worker_script.exists():
        raise FileNotFoundError(f"Worker script not found: {worker_script}")

    specs = build_subrun_specs(
        run_dir=run_dir,
        base_run=loaded["base_run"],
        seeds=loaded["seed_values"],
    )

    metadata: dict[str, Any] = {
        "orchestration_config": str(orchestration_config_path),
        "run_dir": str(run_dir),
        "run_prefix": loaded["run_prefix"],
        "started_at_utc": _utc_now_iso(),
        "worker_script": str(worker_script),
        "max_parallel_subprocesses": loaded["max_parallel_subprocesses"],
        "aggregations": loaded["aggregations"],
        "seed_values": loaded["seed_values"],
        "subruns": [],
    }

    results: list[SubrunResult] = []
    with ThreadPoolExecutor(max_workers=loaded["max_parallel_subprocesses"]) as ex:
        futures = [ex.submit(_run_subprocess_spec, spec, worker_script, workspace_root) for spec in specs]
        for fut in as_completed(futures):
            results.append(fut.result())

    results = sorted(results, key=lambda r: r.run_index)

    for res in results:
        metadata["subruns"].append(
            {
                "run_index": res.run_index,
                "seed": res.seed,
                "status": res.status,
                "exit_code": res.exit_code,
                "duration_seconds": res.duration_seconds,
                "config_path": str(res.config_path),
                "output_path": str(res.output_path),
                "log_path": str(res.log_path),
                "command": list(res.command),
            }
        )

    metadata["finished_at_utc"] = _utc_now_iso()
    metadata["n_total"] = len(results)
    metadata["n_success"] = sum(1 for r in results if r.status == "success")
    metadata["n_failed"] = sum(1 for r in results if r.status == "failed")

    run_aggregations_for_run_folder(
        run_dir=run_dir,
        aggregation_names=loaded["aggregations"],
    )

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return run_dir
