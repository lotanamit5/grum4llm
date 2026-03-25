#!/usr/bin/env python3
"""Run experiment trials locally in a sequential for-loop."""

import argparse
import yaml
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import sys

# Add project root to sys.path to import experiments.utils
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.utils import expand_sweep

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

def main():
    parser = argparse.ArgumentParser(description="Local GRUM Runner")
    parser.add_argument("--config", type=Path, required=True, help="Main experiment YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Parse Config Sections
    experiment_cfg = cfg.get("experiment", {})
    base_cfg = cfg.get("base", {})
    sweep_cfg = cfg.get("sweep", {})

    run_prefix = experiment_cfg.get("run_prefix", "local-exp")
    output_root = Path(experiment_cfg.get("output_root", "results/local"))
    
    # 2. Expand Sweep overrides
    overrides_list = expand_sweep(sweep_cfg)
    
    # 3. Create unique experiment directory
    timestamp = _utc_now_iso()
    experiment_id = f"{run_prefix}-{timestamp}"
    experiment_dir = (output_root / experiment_id).resolve()
    
    (experiment_dir / "subconfigs").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)

    print(f"Starting local experiment: {experiment_id}")
    print(f"Total trials to run: {len(overrides_list)}")
    print(f"Results will be saved to: {experiment_dir}\n")

    # 4. Run Trials in a loop
    for i, overrides in enumerate(overrides_list):
        suffix = "_".join(f"{k}_{v}" for k, v in overrides.items())
        trial_id = f"run_{i:03d}_{suffix}" if suffix else f"run_{i:03d}"
        
        # Merge base and sweep overrides
        trial_cfg = dict(base_cfg)
        trial_cfg.update(overrides)
        
        # Embed trial metadata
        trial_cfg["trial_id"] = trial_id
        trial_cfg["exp_dir"] = str(experiment_dir)
        
        config_path = experiment_dir / "subconfigs" / f"{trial_id}.yml"
        with open(config_path, "w") as sc:
            yaml.safe_dump(trial_cfg, sc, sort_keys=False)
            
        print(f"[{i+1}/{len(overrides_list)}] Running {trial_id}...")
        
        # Call fit_grum.py
        # We assume the user has the environment activated
        cmd = [sys.executable, str(ROOT / "experiments/fit_grum.py"), "--config", str(config_path)]
        
        log_path = experiment_dir / "logs" / f"{trial_id}.log"
        with open(log_path, "w") as log_f:
            subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, check=True)

    print(f"\nExperiment complete. All results are in {experiment_dir}")

if __name__ == "__main__":
    main()
