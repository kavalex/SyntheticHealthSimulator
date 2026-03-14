"""
Tests for configuration module
"""

import pytest

from config import SimulationConfig, DEFAULT_CONFIG


class TestSimulationConfig:
    """Test configuration validation and defaults."""

    def test_default_creation(self):
        """Config should create with valid defaults."""
        config = SimulationConfig()
        assert config.years == 20
        assert config.target_n_people == 5000
        assert config.seed == 42

    def test_invalid_years(self):
        """Should reject non-positive years."""
        with pytest.raises(ValueError, match="years must be positive"):
            SimulationConfig(years=0)

    def test_invalid_diabetes_threshold(self):
        """Should reject diabetes threshold outside [15, 25]."""
        with pytest.raises(ValueError, match="outside typical range"):
            SimulationConfig(diabetes_threshold=30)

    def test_immutability(self):
        """Frozen dataclass should prevent modification."""
        config = SimulationConfig()
        with pytest.raises(AttributeError):
            DEFAULT_CONFIG.years = 10


class TestDefaultConfig:
    """Test default configuration instance."""

    def test_default_available(self):
        """DEFAULT_CONFIG should be accessible."""
        assert DEFAULT_CONFIG.years == 20
        assert DEFAULT_CONFIG.target_n_people == 5000
