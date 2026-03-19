#!/usr/bin/env python3
"""Run a multi-subprocess experiment orchestration from a single YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grums.experiments.orchestrator import run_orchestration


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run orchestration from a single YAML config")
    parser.add_argument("--config", required=True, help="Path to orchestration YAML config")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable the tqdm progress bar for sub-runs.",
    )
    args = parser.parse_args(argv)

    run_dir = run_orchestration(
        orchestration_config_path=(ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config),
        workspace_root=ROOT,
        show_progress=not args.no_progress,
    )
    print(str(run_dir))


if __name__ == "__main__":
    main()
