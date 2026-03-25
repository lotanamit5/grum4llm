#!/usr/bin/env python3
"""Run a multi-subprocess experiment orchestration from a single YAML config."""

import argparse
import yaml
from pathlib import Path
from datetime import datetime, timezone
import random
import sys

# 1. Add project root to sys.path to import experiments.utils
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.utils import expand_sweep

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

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
    
    # 2. Expand Sweep overrides (Product of all sweep params)
    overrides_list = expand_sweep(sweep_cfg)
    
    # 3. Create unique experiment directory
    timestamp = _utc_now_iso()
    experiment_id = f"{run_prefix}-{timestamp}"
    experiment_dir = output_root / experiment_id
    
    (experiment_dir / "subconfigs").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)

    # 4. Generate Subconfigs
    subrun_specs = []
    for i, overrides in enumerate(overrides_list):
        # Merge base and sweep overrides
        trial_cfg = dict(base_cfg)
        trial_cfg.update(overrides)
        
        suffix = "_".join(f"{k}_{v}" for k, v in overrides.items())
        config_path = experiment_dir / "subconfigs" / f"run_{i:03d}_{suffix}.yml"
        output_path = experiment_dir / "outputs" / f"run_{i:03d}_{suffix}.json"
        log_path = experiment_dir / "logs" / f"run_{i:03d}_{suffix}.log"
        
        with open(config_path, "w") as f:
            yaml.safe_dump(trial_cfg, f, sort_keys=False)
            
        subrun_specs.append({"config": config_path, "output": output_path, "log": log_path})

    # 5. Generate slurm_runner.sh
    nodes_list = [n.strip() for n in args.nodes.split(",") if n.strip()] if args.nodes else []
    runner_path = experiment_dir / "slurm_runner.sh"
    worker_script = Path("scripts/worker_slurm.sh").resolve()
    
    with open(runner_path, "w") as f:
        f.write("#!/usr/bin/env bash\n\n")
        f.write(f"# Experiment: {experiment_id}\n")
        f.write(f"# Total runs: {len(subrun_specs)}\n\n")
        
        for spec in subrun_specs:
            node_arg = f"-w {random.choice(nodes_list)} " if nodes_list else ""
            cmd = (
                f"sbatch -A bml -p bml {node_arg}"
                f"-o {spec['log'].resolve()} -e {spec['log'].resolve()} "
                f"{worker_script} {spec['config'].resolve()} {spec['output'].resolve()}"
            )
            f.write(f"{cmd}\n")

    runner_path.chmod(0o755)
    print(f"Created experiment directory: {experiment_dir}")
    print(f"Generated {len(subrun_specs)} subconfigs.")
    print(f"Slurm runner script: {runner_path}")

if __name__ == "__main__":
    main()
