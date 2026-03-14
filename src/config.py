"""
Configuration module for SyntheticHealthSimulator.
All constants and parameters in one place for easy tuning and reproducibility.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any


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
        age_range: (min, max) age at baseline
        ou_params: Parameters for Ornstein-Uhlenbeck processes (theta, sigma, min, max)
        calorie_params: Parameters for calorie generation and appetite regulation
        base_calories: Baseline calorie intake by sex
        macro_norms: Target macronutrient proportions and norms
        muscle_params: Parameters related to muscle mass and metabolism
        lipid_effect: Strength of latent lipid factor
        genetic_risk_std: Standard deviations for log-normal genetic risks
        risk_intercepts: Intercepts for logistic disease risk models
        diabetes_threshold: Threshold for FINDRISC-like diabetes risk
        measurement_noise: Standard deviations for biomarker noise
        biomarker_ranges: Extended physiological ranges during generation
        final_biomarker_ranges: Strict ranges for final validation
        mar_missing_range: (min, max) missingness probability for MAR
    """

    # Core simulation parameters
    years: int = 20
    target_n_people: int = 5000
    seed: int = 42
    boundary_coefficient: float = 1.35
    age_range: Tuple[int, int] = (25, 65)

    # OU Process parameters for lifestyle trajectories
    # theta: mean reversion speed, sigma: volatility, min/max: bounds
    # Increased theta values for faster mean reversion (less inertia) to improve realism.
    ou_params: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "alcohol": {"theta": 1.0, "sigma": 22.0, "min": 0, "max": 500},
            "cardio": {"theta": 0.3, "sigma": 165.0, "min": 0, "max": 3000},
            "strength": {"theta": 0.4, "sigma": 132.0, "min": 0, "max": 1500},
            "smoking": {"theta": 0.7, "sigma": 0.4, "min": 0, "max": 40},
            "stress": {"theta": 0.3, "sigma": 1.1, "min": 0, "max": 10},
            "sleep": {"theta": 0.3, "sigma": 0.9, "min": 4, "max": 10},
        }
    )

    # Calorie generation with physiological feedback
    # BMI compensation prevents weight drift through appetite regulation
    calorie_params: Dict[str, float] = field(
        default_factory=lambda: {
            "base_bias_mean": 30.0,  # Average caloric surplus (kcal)
            "base_bias_std": 60.0,  # Individual variation in appetite
            "bmi_compensation": -40.0,  # kcal per BMI unit from 25
            "stress_effect": 15.0,  # kcal per stress point
            "sleep_deficit_effect": 35.0,  # kcal when sleep < 6 hours
            "autocorrelation": 0.35,  # Year-to-year appetite correlation
            "metabolic_efficiency": 0.9,  # Adaptive thermogenesis factor
        }
    )

    # Base metabolic parameters
    base_calories: Dict[str, int] = field(
        default_factory=lambda: {
            "M": 2500,  # Male baseline
            "F": 2000,  # Female baseline
        }
    )

    # Macronutrient norms (proportions)
    macro_norms: Dict[str, float] = field(
        default_factory=lambda: {
            "protein": 0.15,
            "fat": 0.30,
            "carb": 0.55,
            "fiber_g": 30.0,
            "sodium_mg": 2300.0,
        }
    )

    # Muscle and metabolism parameters
    muscle_params: Dict[str, float] = field(
        default_factory=lambda: {
            "bmr_correction": 8.5,  # kcal per kg muscle difference
            "typical_ratio": 0.30,  # Typical muscle mass to weight ratio
            "strength_to_training": 150.0,  # MET-minutes per strength session
            "protein_requirement": 1.6,  # g/kg for muscle maintenance
            "protein_deficit": 0.8,  # g/kg threshold for muscle loss
            "sleep_deficit_hours": 6.0,  # Threshold for metabolic impact
            "max_factor": 20.0,
            "min_factor": 0.0,
            "mm_to_weight_kg": 0.5,  # Conversion factor to actual kg
        }
    )

    # Lipid profile effect strength
    lipid_effect: float = 2.5

    # Genetic risk parameters (standard deviations for log-normal)
    genetic_risk_std: Dict[str, float] = field(
        default_factory=lambda: {
            "cvd": 0.4,
            "diabetes": 0.3,
            "stroke": 0.3,
            "nafld": 0.25,
            "colorectal": 0.35,
            "cirrhosis": 0.4,
        }
    )

    # Disease risk model intercepts
    risk_intercepts: Dict[str, float] = field(
        default_factory=lambda: {
            "cvd": -5.5,
            "stroke": -5.5,
            "nafld": -2.5,
            "cancer": -5.0,
            "cirrhosis": -4.0,
        }
    )

    # Diabetes threshold (FINDRISC-like points scale)
    diabetes_threshold: float = 17

    # Measurement noise (standard deviations)
    measurement_noise: Dict[str, float] = field(
        default_factory=lambda: {
            "sbp": 5.0,  # mmHg
            "hdl": 5.0,  # mg/dL
            "tc": 10.0,  # mg/dL (total cholesterol)
            "hba1c": 0.2,  # percentage points
            "weight": 1.5,  # kg
        }
    )

    # Extended ranges for generation (allows OU process fluctuations)
    # Lower bounds relaxed to avoid artificial accumulation at edges.
    biomarker_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "bmi": (14, 55),
            "hdl_mgdl": (15, 100),  # was 20
            "total_cholesterol_mgdl": (120, 280),  # was 150
            "sbp_mmhg": (70, 220),
            "hba1c_percent": (3.5, 12.0),
            "weight_kg": (35, 180),
            "non_hdl_mgdl": (40, 232),
            "non_hdl_mmol": (1.5, 6.0),
            "muscle_mass_factor": (0, 20),
        }
    )

    # Strict ranges for final validation (physiologically plausible)
    # Lower bounds relaxed to match extended ranges.
    final_biomarker_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "bmi": (16, 50),
            "hdl_mgdl": (15, 100),  # was 20
            "total_cholesterol_mgdl": (120, 280),  # was 150
            "sbp_mmhg": (80, 200),
            "hba1c_percent": (3.5, 12.0),
            "weight_kg": (40, 150),
            "non_hdl_mgdl": (40, 232),
            "non_hdl_mmol": (1.5, 6.0),
            "muscle_mass_factor": (0, 20),
        }
    )

    # MAR missing value generation
    mar_missing_range: Tuple[float, float] = (0.02, 0.15)  # 2-15% missingness

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.years <= 0:
            raise ValueError(f"years must be positive, got {self.years}")
        if self.target_n_people <= 0:
            raise ValueError(
                f"target_n_people must be positive, got {self.target_n_people}"
            )
        if self.age_range[0] >= self.age_range[1]:
            raise ValueError(
                f"age_range must be (min, max) with min < max, got {self.age_range}"
            )
        if not 10 <= self.diabetes_threshold <= 25:
            raise ValueError(
                f"diabetes_threshold={self.diabetes_threshold} outside typical range [10, 25]"
            )
        if self.boundary_coefficient < 1.0:
            raise ValueError(
                f"boundary_coefficient must be >= 1.0, got {self.boundary_coefficient}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary for logging or serialization.
        Useful for saving configuration alongside generated data.
        """
        result = {}
        for key, value in self.__dataclass_fields__.items():
            result[key] = getattr(self, key)
        return result


# Default instance for convenience
DEFAULT_CONFIG = SimulationConfig()
