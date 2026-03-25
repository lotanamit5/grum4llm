#!/usr/bin/env python3
"""Run a multi-subprocess experiment orchestration from a single YAML config."""

import argparse
import yaml
from pathlib import Path
import random
import sys

# 1. Add project root to sys.path to import experiments.utils
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.utils import expand_sweep, get_utc_timestamp, get_trial_id, create_trial_config
from experiments.paths import get_experiment_dir, ExperimentPaths

def main():
    parser = argparse.ArgumentParser(description="Standalone GRUM Orchestrator")
    parser.add_argument("--config", type=Path, required=True, help="Main experiment YAML")
    parser.add_argument("--nodes", type=str, default="", help="Comma-separated list of nodes to run on")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Parse Config Sections
    experiment_cfg = cfg.get("experiment", {})
    base_cfg = cfg.get("base", {})
    sweep_cfg = cfg.get("sweep", {})

    run_prefix = experiment_cfg.get("run_prefix", "exp")
    output_root = Path(experiment_cfg.get("output_root", "results"))
    
    # 2. Setup Experiment Paths
    timestamp = get_utc_timestamp(compact=True)
    exp_dir = get_experiment_dir(output_root, run_prefix, timestamp)
    paths = ExperimentPaths.create(exp_dir)

    # 3. Expand Sweep overrides (Product of all sweep params)
    overrides_list = expand_sweep(sweep_cfg)
    
    # 4. Generate Subconfigs & Runner
    nodes_list = [n.strip() for n in args.nodes.split(",") if n.strip()] if args.nodes else []
    runner_path = paths.root / "slurm_runner.sh"
    
    # Read worker_script from config or default to scripts/worker_slurm.sh
    worker_script_rel = experiment_cfg.get("worker_script", "scripts/worker_slurm.sh")
    worker_script = (ROOT / worker_script_rel).resolve()
    
    with open(runner_path, "w") as f:
        f.write("#!/usr/bin/env bash\n\n")
        f.write(f"# Experiment: {exp_dir.name}\n")
        f.write(f"# Total runs: {len(overrides_list)}\n\n")
        
        for i, overrides in enumerate(overrides_list):
            trial_id = get_trial_id(i, overrides)
            config_path = paths.subconfigs / f"{trial_id}.yml"
            
            # Create trial config
            create_trial_config(base_cfg, overrides, trial_id, paths.root, config_path)
            
            # Use %j for numeric job id, sorted by job id first
            log_out = paths.logs / f"%j_{trial_id}.out"
            log_err = paths.logs / f"%j_{trial_id}.err"
            
            node_arg = f"-w {random.choice(nodes_list)} " if nodes_list else ""
            
            # Pass only the config_path to the worker
            cmd = (
                f"sbatch -A bml -p bml {node_arg}"
                f"-o {log_out} -e {log_err} "
                f"{worker_script} {config_path}"
            )
            f.write(f"{cmd}\n")

    runner_path.chmod(0o755)
    print(f"Created experiment directory: {exp_dir}")
    print(f"Generated {len(overrides_list)} subconfigs.")
    print(f"Slurm runner script: {runner_path}")

    # Run the slurm_runner.sh script
    import subprocess
    print(f"\nLaunching jobs with: bash {runner_path}")
    subprocess.run(["bash", str(runner_path)], check=True)

if __name__ == "__main__":
    main()
