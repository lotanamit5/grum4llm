"""Sushi dataset loader."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grums.contracts import AgentRecord, AlternativeRecord

SUSHI_URL = "https://www.kamishima.net/asset/sushi3-2016.zip"


@dataclass(frozen=True)
class SushiDataset:
    agent_features: np.ndarray
    alternative_features: np.ndarray
    rankings: list[tuple[int, ...]]

    @property
    def n_agents(self) -> int:
        return self.agent_features.shape[0]

    @property
    def n_alternatives(self) -> int:
        return self.alternative_features.shape[0]


def load_sushi(data_dir: str | Path = ".data") -> SushiDataset:
    """Download and parse Kamishima sushi3a dataset.
    
    Downloads from SUSHI_URL if not found locally in data_dir.
    Extracts sushi3.udata (Agents), sushi3.idata (Alternatives), and sushi3a.5000.10.order (Rankings).
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "sushi3-2016.zip"

    if not zip_path.exists():
        print(f"Downloading Sushi dataset from {SUSHI_URL}...")
        req = urllib.request.Request(SUSHI_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            zip_path.write_bytes(response.read())

    with zipfile.ZipFile(zip_path, "r") as z:
        udata_text = z.read("sushi3-2016/sushi3.udata").decode("utf-8")
        idata_text = z.read("sushi3-2016/sushi3.idata").decode("utf-8")
        order_text = z.read("sushi3-2016/sushi3a.5000.10.order").decode("utf-8")

    # Parse agents (udata)
    # columns: [user_id, gender, age, time_taken, pref_until15, region_until15, east_west15, current_pref, current_region, east_west, same_pref]
    # We drop the user_id (column 0)
    agent_rows = []
    for line in udata_text.strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 11:
            row = [float(x) for x in parts[1:]]  # Drop column 0 (user_id)
            agent_rows.append(row)
    agent_features = np.array(agent_rows, dtype=float)

    # Parse alternatives (idata)
    # columns: [item_id, name, style, major_group, minor_group, oiliness, eating_freq, price, sell_freq]
    # We drop the item_id (col 0) and name (col 1)
    alt_rows = []
    for line in idata_text.strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 9:
            row = [float(x) for x in parts[2:]]  # Drop col 0 (id) and col 1 (name)
            alt_rows.append(row)
    alternative_features = np.array(alt_rows, dtype=float)

    # Parse rankings (sushi3a.5000.10.order)
    # First line is "10 1"
    # Proceeding 5000 lines are "0 10 r0 r1 ... r9"
    lines = order_text.strip().splitlines()
    rankings = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 12:
            # First two elements are meta (usually "0", "10")
            item_order = tuple(int(x) for x in parts[2:12])
            rankings.append(item_order)

    return SushiDataset(
        agent_features=agent_features,
        alternative_features=alternative_features,
        rankings=rankings,
    )
