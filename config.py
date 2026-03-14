"""
Configuration module for SyntheticHealthSimulator.
All constants and parameters in one place for easy tuning and reproducibility.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class SimulationConfig:
    """
    Main configuration class for health data simulation.

    Frozen dataclass ensures immutability — configuration cannot be
    accidentally modified during simulation.

    Attributes:
        years: Simulation horizon in years (default: 20)
        target_n_people: Desired final sample size after filtering (default: 5000)
        seed: Random seed for reproducibility (default: 42)
        boundary_coefficient: Oversampling factor for boundary filtering (default: 1.35)
    """

    # Core simulation parameters
    years: int = 20
    target_n_people: int = 5000
    seed: int = 42
    boundary_coefficient: float = 1.35

    # Age range for cohort generation
    age_range: Tuple[int, int] = (20, 50)

    # OU Process parameters for lifestyle trajectories
    # theta: mean reversion speed, sigma: volatility, min/max: physiological bounds
    ou_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {

        "alcohol": {
            "theta": 0.8,
            "sigma": 20.0,
            "min": 0,
            "max": 500
        },

        "cardio": {
            "theta": 0.2,
            "sigma": 150.0,
            "min": 0,
            "max": 3000
        },

        "strength": {
            "theta": 0.25,
            "sigma": 120.0,
            "min": 0,
            "max": 1500
        },

        "smoking": {
            "theta": 0.3,
            "sigma": 0.3,
            "min": 0,
            "max": 40
        },

        "stress": {
            "theta": 0.2,
            "sigma": 1.0,
            "min": 0,
            "max": 10
        },

        "sleep": {
            "theta": 0.2,
            "sigma": 0.8,
            "min": 4,
            "max": 10
        },
    })

    # Calorie generation with physiological feedback
    # BMI compensation prevents weight drift through appetite regulation
    calorie_params: Dict[str, float] = field(default_factory=lambda: {

        # Average caloric surplus (kcal)
        "base_bias_mean": 30.0,

        # Individual variation in appetite
        "base_bias_std": 60.0,

        # kcal per BMI unit from 25
        "bmi_compensation": -40.0,

        # kcal per stress point
        "stress_effect": 15.0,

        # kcal when sleep < 6 hours
        "sleep_deficit_effect": 35.0,

        # Year-to-year appetite correlation
        "autocorrelation": 0.35,

        # Adaptive thermogenesis factor
        "metabolic_efficiency": 0.9,
    })

    # Base metabolic parameters
    base_calories: Dict[str, int] = field(default_factory=lambda: {

        # Male baseline
        "M": 2500,

        # Female baseline
        "F": 2000,
    })

    # Macronutrient norms (proportions)
    macro_norms: Dict[str, float] = field(default_factory=lambda: {
        "protein": 0.15,
        "fat": 0.30,
        "carb": 0.55,
        "fiber_g": 30.0,
        "sodium_mg": 2300.0,
    })

    # Muscle and metabolism parameters
    muscle_params: Dict[str, float] = field(default_factory=lambda: {

        # kcal per kg muscle difference
        "bmr_correction": 8.5,

        # Typical muscle mass to weight ratio
        "typical_ratio": 0.30,

        # MET-minutes per strength session
        "strength_to_training": 150.0,

        # g/kg for muscle maintenance
        "protein_requirement": 1.6,

        # g/kg threshold for muscle loss
        "protein_deficit": 0.8,

        # Threshold for metabolic impact
        "sleep_deficit_hours": 6.0,

        "max_factor": 20.0,

        "min_factor": 0.0,

        # Conversion factor to actual kg
        "mm_to_weight_kg": 0.5,
    })

    # Lipid profile effect strength
    lipid_effect: float = 5.0

    # Genetic risk parameters (standard deviations for log-normal)
    genetic_risk_std: Dict[str, float] = field(default_factory=lambda: {
        "cvd": 0.4,
        "diabetes": 0.3,
        "stroke": 0.3,
        "nafld": 0.25,
        "colorectal": 0.35,
        "cirrhosis": 0.4,
    })

    # Disease risk model intercepts
    risk_intercepts: Dict[str, float] = field(default_factory=lambda: {
        "cvd": -3.5,
        "stroke": -3.0,
        "nafld": -2.5,
        "cancer": -2.0,
        "cirrhosis": -2.2,
    })

    # Diabetes threshold (FINDRISC-like points scale)
    diabetes_threshold: float = 15

    # Measurement noise (standard deviations)
    measurement_noise: Dict[str, float] = field(default_factory=lambda: {
        "sbp": 5.0,  # mmHg
        "hdl": 5.0,  # mg/dL
        "tc": 10.0,  # mg/dL (total cholesterol)
        "hba1c": 0.2,  # percentage points
        "weight": 1.5,  # kg
    })

    # Extended ranges for generation (allows OU process fluctuations)
    biomarker_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "bmi": (14, 55),
        "hdl_mgdl": (20, 100),
        "total_cholesterol_mgdl": (120, 400),
        "sbp_mmhg": (70, 220),
        "hba1c_percent": (3.5, 12.0),
        "weight_kg": (35, 220),
        "non_hdl_mgdl": (48, 235),
        "non_hdl_mmol": (1.9, 6.2),
        "muscle_mass_factor": (0, 20),
    })

    # Strict ranges for final validation (physiologically plausible)
    final_biomarker_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "bmi": (16, 50),
        "hdl_mgdl": (20, 100),
        "total_cholesterol_mgdl": (150, 350),
        "sbp_mmhg": (80, 200),
        "hba1c_percent": (3.5, 12.0),
        "weight_kg": (40, 200),
        "non_hdl_mgdl": (50, 232),
        "non_hdl_mmol": (2.0, 6.0),
        "muscle_mass_factor": (0, 20),
    })

    # MAR missing value generation
    mar_missing_range: Tuple[float, float] = (0.02, 0.15)  # 2-15% missingness

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.years <= 0:
            raise ValueError(f"years must be positive, got {self.years}")
        if self.target_n_people <= 0:
            raise ValueError(f"target_n_people must be positive, got {self.target_n_people}")
        if not 15 <= self.diabetes_threshold <= 25:
            raise ValueError(
                f"diabetes_threshold={self.diabetes_threshold} outside typical range [15, 25]"
            )
        if self.boundary_coefficient < 1.0:
            raise ValueError(
                f"boundary_coefficient must be >= 1.0, got {self.boundary_coefficient}"
            )


# Default instance for convenience
DEFAULT_CONFIG = SimulationConfig()
