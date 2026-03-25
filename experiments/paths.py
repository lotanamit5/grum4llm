from pathlib import Path
from typing import NamedTuple

class ExperimentPaths(NamedTuple):
    root: Path
    subconfigs: Path
    outputs: Path
    logs: Path

    @classmethod
    def create(cls, base_dir: Path) -> "ExperimentPaths":
        """Creates the directory structure for an experiment and returns the paths."""
        base_dir.mkdir(parents=True, exist_ok=True)
        
        paths = cls(
            root=base_dir,
            subconfigs=base_dir / "subconfigs",
            outputs=base_dir / "outputs",
            logs=base_dir / "logs"
        )
        
        paths.subconfigs.mkdir(exist_ok=True)
        paths.outputs.mkdir(exist_ok=True)
        paths.logs.mkdir(exist_ok=True)
        
        return paths

def get_experiment_dir(output_root: Path, run_prefix: str, timestamp: str) -> Path:
    """Generates a standard experiment directory path."""
    return (output_root / f"{run_prefix}-{timestamp}").resolve()
