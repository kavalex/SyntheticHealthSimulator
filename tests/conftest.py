"""
Pytest configuration and shared fixtures
"""

import sys
from pathlib import Path

# CRITICAL: Add project root to sys.path BEFORE any other imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from config import SimulationConfig, DEFAULT_CONFIG
from logging_config import setup_logging
import logging

# Quiet logging during tests
setup_logging(level=logging.WARNING)


@pytest.fixture
def config():
    """Default simulation configuration."""
    return DEFAULT_CONFIG


@pytest.fixture
def small_config():
    """Small configuration for fast tests."""
    return SimulationConfig(
        years=5,
        target_n_people=100,
        seed=42
    )


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def generator(config):
    """Fresh DataGenerator instance."""
    # Импорт здесь, чтобы избежать циклических импортов
    from generator import DataGenerator
    return DataGenerator(seed=42, config=config)
