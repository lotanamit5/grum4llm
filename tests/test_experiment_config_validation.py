from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_unknown_config_key_fails(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_social_choice_experiment.py"
    cfg = tmp_path / "bad.yml"
    cfg.write_text("unknown_key: 1\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg)],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "Unknown config keys" in (proc.stderr + proc.stdout)


def test_invalid_mode_in_config_fails(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_social_choice_experiment.py"
    cfg = tmp_path / "bad_mode.yml"
    cfg.write_text("mode: nope\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg)],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "mode must be one of" in (proc.stderr + proc.stdout)


def test_invalid_dataset_in_config_fails(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_social_choice_experiment.py"
    cfg = tmp_path / "bad_dataset.yml"
    cfg.write_text("dataset: nope\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg)],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "dataset must be one of" in (proc.stderr + proc.stdout)


def test_invalid_n_jobs_in_config_fails(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_social_choice_experiment.py"
    cfg = tmp_path / "bad_n_jobs.yml"
    cfg.write_text("n_jobs: 0\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg)],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "n_jobs must be positive" in (proc.stderr + proc.stdout)
