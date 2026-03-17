from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_social_choice_experiment_script_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_social_choice_experiment.py"
    out = tmp_path / "social_choice_results.json"

    cmd = [
        sys.executable,
        str(script),
        "--mode",
        "both",
        "--agent-counts",
        "5,10",
        "--rounds",
        "4",
        "--repeats",
        "1",
        "--seed",
        "2",
        "--output-json",
        str(out),
    ]
    subprocess.run(cmd, check=True)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "asymptotic" in payload
    assert "criteria" in payload
    assert "timing" in payload
    assert len(payload["asymptotic"]) == 2
    assert set(payload["criteria"].keys()) == {"random", "d_opt", "e_opt", "social"}
    assert payload["timing"]["total_seconds"] >= 0.0


def test_run_with_config_file(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_social_choice_experiment.py"
    out = tmp_path / "social_choice_results_cfg.json"
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        "\n".join(
            [
                "mode: asymptotic",
                "agent_counts: [5, 10]",
                "repeats: 1",
                "seed: 3",
                "iterations: 0",
                "no_progress: true",
                f"output_json: {out}",
            ]
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(cfg),
    ]
    subprocess.run(cmd, check=True)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["config_file"] == str(cfg)
    assert "asymptotic" in payload
    assert len(payload["asymptotic"]) == 2
    assert "criteria" not in payload


def test_cli_overrides_config_value(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_social_choice_experiment.py"
    out = tmp_path / "social_choice_results_override.json"
    cfg = tmp_path / "cfg_override.yml"
    cfg.write_text(
        "\n".join(
            [
                "mode: criteria",
                "rounds: 2",
                "repeats: 1",
                "seed: 1",
                "iterations: 0",
                "no_progress: true",
                f"output_json: {out}",
            ]
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(cfg),
        "--rounds",
        "4",
    ]
    subprocess.run(cmd, check=True)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["config_file"] == str(cfg)
    assert "criteria" in payload
    assert payload["repeats"] == 1


def test_output_path_creates_parent_directories(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_social_choice_experiment.py"
    out = tmp_path / "nested" / "deeper" / "results.json"

    cmd = [
        sys.executable,
        str(script),
        "--mode",
        "asymptotic",
        "--agent-counts",
        "5",
        "--repeats",
        "1",
        "--iterations",
        "0",
        "--no-progress",
        "--output-json",
        str(out),
    ]
    subprocess.run(cmd, check=True)

    assert out.exists()
