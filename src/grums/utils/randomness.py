"""Randomness helpers for deterministic experiments and tests."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set deterministic seed for Python and NumPy.

    This utility centralizes seed management so Monte Carlo tests remain reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
