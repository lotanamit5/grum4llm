from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from grums.experiments.orchestrator import (
    build_subrun_specs,
    load_orchestration_config,
    run_aggregations_for_json_paths,
)


def test_load_orchestration_config_parses_seed_range(tmp_path: Path) -> None:
    cfg = tmp_path / "orch.yml"
    cfg.write_text(
        "\n".join(
            [
                "run_prefix: fig2-repro",
                "output_root: results/repro",
                "base_run:",
                "  mode: asymptotic",
                "  dataset: dataset2",
                "  agent_counts: [5]",
                "  repeats: 1",
                "sweep:",
                "  seeds:",
                "    start: 1",
                "    stop: 3",
                "    step: 1",
            ]
        ),
        encoding="utf-8",
    )

    parsed = load_orchestration_config(cfg)
    assert parsed["seed_values"] == [1, 2, 3]
    assert parsed["run_prefix"] == "fig2-repro"


def test_load_orchestration_config_parses_sweep_parameters(tmp_path: Path) -> None:
    cfg = tmp_path / "orch_params.yml"
    cfg.write_text(
        "\n".join(
            [
                "run_prefix: crit-repro",
                "output_root: results/repro",
                "base_run:",
                "  mode: criteria",
                "  dataset: dataset2",
                "  repeats: 1",
                "sweep:",
                "  seeds: [0, 1]",
                "  parameters:",
                "    rounds: [5, 10]",
                "    dataset: [dataset1, dataset2]",
            ]
        ),
        encoding="utf-8",
    )

    parsed = load_orchestration_config(cfg)
    assert parsed["seed_values"] == [0, 1]
    assert parsed["sweep_parameters"] == {
        "rounds": [5, 10],
        "dataset": ["dataset1", "dataset2"],
    }


def test_build_subrun_specs_cartesian_product_overrides(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for child in ["subconfigs", "outputs", "logs"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)

    specs = build_subrun_specs(
        run_dir=run_dir,
        base_run={"mode": "criteria", "repeats": 1},
        seeds=[0, 1],
        sweep_parameters={"rounds": [5, 10]},
    )

    assert len(specs) == 4
    assert [s.run_index for s in specs] == [0, 1, 2, 3]
    assert {s.seed for s in specs} == {0, 1}
    assert {s.sweep_overrides.get("rounds") for s in specs} == {5, 10}

    for spec in specs:
        assert spec.config_path.exists()
        text = spec.config_path.read_text(encoding="utf-8")
        assert f"seed: {spec.seed}" in text
        assert f"rounds: {spec.sweep_overrides['rounds']}" in text
        assert str(spec.output_path) in text


def test_orchestration_smoke_creates_mapped_subconfigs_outputs_and_aggregates(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_experiment_orchestration.py"
    cfg = tmp_path / "orch_smoke.yml"
    output_root = tmp_path / "orchestration_outputs"

    cfg.write_text(
        "\n".join(
            [
                "run_prefix: smoke-orch",
                f"output_root: {output_root}",
                "worker_script: scripts/run_social_choice_experiment.py",
                "max_parallel_subprocesses: 2",
                "aggregations: [asymptotic, criteria, timing]",
                "base_run:",
                "  mode: asymptotic",
                "  dataset: dataset2",
                "  agent_counts: [5, 10]",
                "  repeats: 1",
                "  n_jobs: 1",
                "  iterations: 0",
                "  gibbs_samples: 6",
                "  gibbs_burnin: 3",
                "  sigma: 1.0",
                "  prior_precision: 0.01",
                "  tolerance: 1.0e-5",
                "  random_seed: 0",
                "  quiet: true",
                "  no_progress: true",
                "sweep:",
                "  seeds: [0, 1]",
            ]
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg)],
        capture_output=True,
        text=True,
        check=True,
    )

    run_dir = Path(proc.stdout.strip().splitlines()[-1])
    assert run_dir.exists()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["n_total"] == 2
    assert metadata["n_success"] == 2
    assert metadata["n_failed"] == 0
    assert len(metadata["subruns"]) == 2

    for sub in metadata["subruns"]:
        cfg_path = Path(sub["config_path"])
        out_path = Path(sub["output_path"])
        log_path = Path(sub["log_path"])
        assert cfg_path.exists()
        assert out_path.exists()
        assert log_path.exists()
        assert sub["status"] == "success"

    assert (run_dir / "aggregates" / "asymptotic.json").exists()
    assert (run_dir / "aggregates" / "criteria.json").exists()
    assert (run_dir / "aggregates" / "timing.json").exists()


def test_can_reaggregate_old_jsons_with_function_call(tmp_path: Path) -> None:
    outputs = tmp_path / "jsons"
    agg = tmp_path / "agg"
    outputs.mkdir(parents=True, exist_ok=True)

    payload_a = {
        "seed": 0,
        "asymptotic": [{"n_agents": 50, "mean_tau": 0.6}],
        "criteria": {"random": 0.4, "social": 0.7},
        "timing": {"asymptotic_seconds": 1.0, "criteria_seconds": 2.0, "total_seconds": 3.0},
    }
    payload_b = {
        "seed": 1,
        "asymptotic": [{"n_agents": 50, "mean_tau": 0.8}],
        "criteria": {"random": 0.5, "social": 0.9},
        "timing": {"asymptotic_seconds": 1.5, "criteria_seconds": 2.5, "total_seconds": 4.0},
    }

    a = outputs / "a.json"
    b = outputs / "b.json"
    a.write_text(json.dumps(payload_a), encoding="utf-8")
    b.write_text(json.dumps(payload_b), encoding="utf-8")

    written = run_aggregations_for_json_paths(
        json_paths=[a, b],
        output_dir=agg,
        aggregation_names=["asymptotic", "criteria", "timing"],
    )

    assert set(written.keys()) == {"asymptotic", "criteria", "timing"}

    asym = json.loads(written["asymptotic"].read_text(encoding="utf-8"))
    assert len(asym["rows"]) == 2
    assert asym["summary"][0]["n_agents"] == 50
    assert asym["summary"][0]["count"] == 2


def test_aggregate_criteria_curve_groups_by_criterion_and_n(tmp_path: Path) -> None:
    outputs = tmp_path / "curve_jsons"
    agg = tmp_path / "curve_agg"
    outputs.mkdir(parents=True, exist_ok=True)

    payload_a = {
        "seed": 0,
        "criterion": "social",
        "criteria_curve": [
            {"n_observations": 1, "kendall_tau": 0.1},
            {"n_observations": 2, "kendall_tau": 0.2},
        ],
        "timing": {"total_seconds": 1.0},
    }
    payload_b = {
        "seed": 1,
        "criterion": "social",
        "criteria_curve": [
            {"n_observations": 1, "kendall_tau": 0.3},
            {"n_observations": 2, "kendall_tau": 0.4},
        ],
        "timing": {"total_seconds": 2.0},
    }
    (outputs / "a.json").write_text(json.dumps(payload_a), encoding="utf-8")
    (outputs / "b.json").write_text(json.dumps(payload_b), encoding="utf-8")

    written = run_aggregations_for_json_paths(
        json_paths=sorted(outputs.glob("*.json")),
        output_dir=agg,
        aggregation_names=["criteria_curve"],
    )
    data = json.loads(written["criteria_curve"].read_text(encoding="utf-8"))
    assert len(data["rows"]) == 4
    by_n = {row["n_observations"]: row["mean"] for row in data["summary"] if row["criterion"] == "social"}
    assert by_n[1] == pytest.approx(0.2)
    assert by_n[2] == pytest.approx(0.3)
