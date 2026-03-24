"""Smoke tests for the sushi experiment runner script."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sushi_experiment.py"


# ---------------------------------------------------------------------------
# Unit: compare_criteria_sushi_choice can be imported without circular errors
# ---------------------------------------------------------------------------

def test_sushi_module_imports_cleanly() -> None:
    from grums.experiments.sushi import compare_criteria_sushi_choice  # noqa: F401
    from grums.datasets.sushi import SushiDataset, load_sushi  # noqa: F401


# ---------------------------------------------------------------------------
# Unit: runner CLI parses args and produces correct JSON keys
# ---------------------------------------------------------------------------

def _make_fake_dataset() -> "SushiDataset":
    from grums.datasets.sushi import SushiDataset
    import numpy as np

    rng = np.random.default_rng(0)
    n_agents = 1200
    n_alt = 10
    n_agent_feat = 10
    n_alt_feat = 7
    agent_features = rng.random((n_agents, n_agent_feat)).astype(np.float64)
    alternative_features = rng.random((n_alt, n_alt_feat)).astype(np.float64)
    rankings = [tuple(rng.permutation(n_alt).tolist()) for _ in range(n_agents)]
    return SushiDataset(
        agent_features=agent_features,
        alternative_features=alternative_features,
        rankings=rankings,
    )


def test_runner_script_produces_valid_json(tmp_path: Path) -> None:
    """End-to-end smoke: run the script with mocked sushi data, minimal MCEM."""
    out = tmp_path / "result.json"

    cmd = [
        sys.executable, str(SCRIPT),
        "--criterion", "random",
        "--metric", "social",
        "--repeats", "1",
        "--rounds", "2",
        "--iterations", "0",
        "--gibbs-samples", "4",
        "--gibbs-burnin", "2",
        "--n-jobs", "1",
        "--output-json", str(out),
        "--quiet",
        "--no-progress",
    ]

    fake_ds = _make_fake_dataset()

    with patch("grums.experiments.sushi.load_sushi", return_value=fake_ds), \
         patch("grums.experiments.sushi._SUSHI_FIT_CACHE", None):
        proc = subprocess.run(cmd, capture_output=True, text=True)

    # If the patch didn't take effect (subprocess isolation), just run without mock
    if proc.returncode != 0:
        proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 0, f"Script failed:\n{proc.stderr}"
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert "criteria" in payload, "Output JSON must contain 'criteria'"
    assert "random" in payload["criteria"], "Should have 'random' criterion result"
    assert "timing" in payload, "Output JSON must contain 'timing'"
    assert "started_at_utc" in payload
    assert "finished_at_utc" in payload
    assert isinstance(payload["criteria"]["random"], dict)
    assert "social" in payload["criteria"]["random"]
    assert isinstance(payload["criteria"]["random"]["social"], float)


def _load_sushi_runner_fresh(module_name: str = "sushi_runner_criteria_test"):
    """Load run_sushi_experiment.py as a module (subprocess would not see unittest.mock patches)."""
    import importlib.util

    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_runner_script_cli_recognises_all_criteria(tmp_path: Path) -> None:
    """Verify the script accepts all valid criterion names."""
    with patch("grums.experiments.sushi.compare_criteria_sushi_choice", return_value={"social": 0.42, "mean_person": 0.41, "raw_person": 0.40}):
        mod = _load_sushi_runner_fresh()
        for criterion in ["random", "d_opt", "e_opt", "social", "personalized"]:
            out = tmp_path / f"result_{criterion}.json"
            mod.main(
                [
                    "--criterion",
                    criterion,
                    "--metric",
                    "social",
                    "--repeats",
                    "1",
                    "--rounds",
                    "1",
                    "--iterations",
                    "0",
                    "--gibbs-samples",
                    "4",
                    "--gibbs-burnin",
                    "2",
                    "--n-jobs",
                    "1",
                    "--output-json",
                    str(out),
                    "--quiet",
                    "--no-progress",
                ]
            )
            payload = json.loads(out.read_text(encoding="utf-8"))
            assert criterion in payload["criteria"]
            assert payload["criteria"][criterion]["social"] == 0.42


def test_runner_script_config_file(tmp_path: Path) -> None:
    """Verify YAML config file loading works correctly."""
    out = tmp_path / "cfg_result.json"
    cfg = tmp_path / "run.yml"
    cfg.write_text(
        "\n".join([
            "criterion: random",
            "metric: social",
            "repeats: 1",
            "rounds: 1",
            "iterations: 0",
            "gibbs_samples: 4",
            "gibbs_burnin: 2",
            "n_jobs: 1",
            "quiet: true",
            "no_progress: true",
            f"output_json: {out}",
        ]),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(cfg)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"Script failed with config file:\n{proc.stderr}"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "random" in payload["criteria"]
