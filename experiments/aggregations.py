import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

def aggregate_criteria_curve(payloads: list[dict]):
    rows = []
    grouped = {}
    for p in payloads:
        seed = p.get("seed")
        criterion = p.get("criterion")
        curve = p.get("criteria_curve", [])
        for pt in curve:
            n_obs = int(pt["n_observations"])
            for m_key in ["social_tau", "mean_person_tau", "raw_person_tau"]:
                if m_key in pt and pt[m_key] is not None:
                    value = float(pt[m_key])
                    rows.append({"seed": seed, "criterion": str(criterion), "n_observations": n_obs, "metric": m_key, "score": value})
                    grouped.setdefault((str(criterion), n_obs, m_key), []).append(value)
    
    summary = []
    for (crit, n_obs, m_key), values in sorted(grouped.items()):
        summary.append({
            "criterion": crit, "n_observations": n_obs, "metric": m_key,
            "count": len(values), "mean": mean(values), "std": pstdev(values) if len(values) > 1 else 0.0
        })
    return {"rows": rows, "summary": summary}

def aggregate_timing(payloads: list[dict]):
    totals = [p.get("timing", {}).get("total_seconds", 0.0) for p in payloads]
    return {
        "count": len(totals),
        "mean_seconds": mean(totals) if totals else 0.0,
        "std_seconds": pstdev(totals) if len(totals) > 1 else 0.0
    }

def run_aggregation(run_dir: Path):
    outputs_dir = run_dir / "outputs"
    agg_dir = run_dir / "aggregates"
    agg_dir.mkdir(exist_ok=True)
    
    payloads = []
    for f in outputs_dir.glob("*.json"):
        with open(f, "r") as r:
            payloads.append(json.load(r))
    
    if not payloads:
        print(f"No results found in {outputs_dir}")
        return

    curve_data = aggregate_criteria_curve(payloads)
    time_data = aggregate_timing(payloads)
    
    with open(agg_dir / "criteria_curve.json", "w") as f:
        json.dump(curve_data, f, indent=2)
    with open(agg_dir / "timing.json", "w") as f:
        json.dump(time_data, f, indent=2)
        
    print(f"Aggregated results to {agg_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()
    run_aggregation(args.run_dir)
