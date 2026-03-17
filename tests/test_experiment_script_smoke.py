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
