#!/usr/bin/env python3
"""Run experiment trials locally in a sequential for-loop."""

import argparse
import yaml
import subprocess
from pathlib import Path
import sys

# 1. Add project root to sys.path to import experiments.utils
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.utils import expand_sweep, get_utc_timestamp, get_trial_id, create_trial_config
from experiments.paths import get_experiment_dir, ExperimentPaths

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
    
    # 2. Setup Experiment Paths
    timestamp = get_utc_timestamp()
    exp_dir = get_experiment_dir(output_root, run_prefix, timestamp)
    paths = ExperimentPaths.create(exp_dir)

    # 3. Expand Sweep overrides
    overrides_list = expand_sweep(sweep_cfg)
    
    print(f"Starting local experiment: {exp_dir.name}")
    print(f"Total trials to run: {len(overrides_list)}")
    print(f"Results will be saved to: {exp_dir}\n")

    # 4. Run Trials in a loop
    for i, overrides in enumerate(overrides_list):
        trial_id = get_trial_id(i, overrides)
        config_path = paths.subconfigs / f"{trial_id}.yml"
        
        # Create trial config
        create_trial_config(base_cfg, overrides, trial_id, paths.root, config_path)
            
        print(f"[{i+1}/{len(overrides_list)}] Running {trial_id}...")
        
        # Call fit_grum.py
        cmd = [sys.executable, str(ROOT / "experiments/fit_grum.py"), "--config", str(config_path)]
        
        log_path = paths.logs / f"{trial_id}.log"
        with open(log_path, "w") as log_f:
            subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, check=True)

    print(f"\nExperiment complete. All results are in {exp_dir}")

if __name__ == "__main__":
    main()
