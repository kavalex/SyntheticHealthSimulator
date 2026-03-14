"""
Pytest configuration and shared fixtures - adapted for new structure
"""
import sys
from pathlib import Path

new_path = Path(__file__).resolve().parent.parent
if str(new_path) not in sys.path:
    sys.path.insert(0, str(new_path))

import numpy as np
import pytest
from src.config import SimulationConfig, DEFAULT_CONFIG
from src.logging_config import setup_logging
import logging

setup_logging(level=logging.WARNING)


@pytest.fixture
def config():
    return DEFAULT_CONFIG


@pytest.fixture
def small_config():
    return SimulationConfig(years=5, target_n_people=100, seed=42)


@pytest.fixture
def rng():
    return np.random.default_rng(42)
