"""
Cohort generation module.

Generates baseline cohort with demographics, genetic risks,
and initial biomarkers.
"""

from typing import Tuple, Optional

import numpy as np
import pandas as pd

from config import SimulationConfig
from utils.helpers import generate_truncnorm, generate_genetic_risk


class CohortGenerator:
    """
    Generates baseline cohort for health simulation.

    Responsibilities:
    - Demographics (age, sex, height)
    - Genetic risk factors
    - Initial biomarkers (HDL, TC, SBP, HbA1c)
    - Muscle mass distribution
    - BMR calculation
    """

    def __init__(
            self,
            config: SimulationConfig,
            rng: np.random.Generator,
    ):
        self.config = config
        self.rng = rng

        # Import genetic risk std params from config
        self.genetic_std = config.genetic_risk_std

        # Lipid effect for initial biomarkers
        self.lipid_effect = config.lipid_effect

        # Muscle params
        self.muscle_params = config.muscle_params
        self.bmr_correction = config.muscle_params["bmr_correction"]
        self.typical_ratio = config.muscle_params["typical_ratio"]
        self.mm_to_weight_kg = config.muscle_params["mm_to_weight_kg"]

    def generate(
            self,
            n_people: int,
            age_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Generate baseline cohort with demographics and initial biomarkers.

        Args:
            n_people: Number of people to generate
            age_range: Optional (min, max) age override

        Returns:
            DataFrame with cohort data
        """
        age_range = age_range or self.config.age_range

        # Age distribution
        age = generate_truncnorm(
            self.rng, 35, 8, age_range[0], age_range[1], n_people
        ).astype(int)

        # Sex distribution (50/50)
        sexes = self.rng.choice(["M", "F"], n_people)
        male_mask = sexes == "M"
        female_mask = ~male_mask

        # Height distribution
        height_cm = np.zeros(n_people)
        height_cm[male_mask] = self.rng.normal(178, 7, np.sum(male_mask))
        height_cm[female_mask] = self.rng.normal(163, 6, np.sum(female_mask))

        # Generate genetic risks
        genetic_diabetes = generate_genetic_risk(
            self.rng, n_people, "diabetes", self.genetic_std
        )
        genetic_cvd = generate_genetic_risk(
            self.rng, n_people, "cvd", self.genetic_std
        )
        genetic_cancer = generate_genetic_risk(
            self.rng, n_people, "colorectal", self.genetic_std
        )
        genetic_stroke = generate_genetic_risk(
            self.rng, n_people, "stroke", self.genetic_std
        )
        genetic_nafld = generate_genetic_risk(
            self.rng, n_people, "nafld", self.genetic_std
        )
        genetic_cirrhosis = generate_genetic_risk(
            self.rng, n_people, "cirrhosis", self.genetic_std
        )

        # Muscle mass distribution
        muscle = self._generate_muscle_mass(sexes)

        # Initial biomarkers
        hdl = self._generate_hdl(np.sum(male_mask), np.sum(female_mask))
        hdl = np.clip(hdl, 20, 100)

        tc = generate_truncnorm(
            self.rng, loc=200, scale=25, low=150, high=350, size=n_people
        )
        tc += 0.8 * genetic_cvd
        tc = np.clip(tc, 150, 350)

        sbp = 115 + 0.5 * (age - 35) + self.rng.normal(0, 8, n_people)
        sbp = np.clip(sbp, 80, 200)

        hba1c = generate_truncnorm(
            self.rng, loc=5.4, scale=0.3, low=4.0, high=6.5, size=n_people
        )
        hba1c = np.clip(hba1c, 3.5, 12.0)

        # Weight and BMI
        bmi_start = generate_truncnorm(
            self.rng, loc=24, scale=3.5, low=18, high=32, size=n_people
        )
        height_m = height_cm / 100
        weight_start = bmi_start * (height_m ** 2)

        # BMR calculation
        bmr = self._calculate_bmr_vectorized(
            weight_start, height_cm, age, sexes, muscle
        )

        # Generate lipid factors
        lipid_factor_hdl, lipid_factor_tc = self._generate_lipid_factors(
            n_people, correlation=0.3
        )

        cohort = pd.DataFrame({
            "person_id": range(1, n_people + 1),
            "age_start": age,
            "sex": sexes,
            "height_cm": height_cm.round(1),
            "weight_start_kg": weight_start.round(1),
            "bmi_start": bmi_start.round(2),
            "muscle_mass_start": muscle.round(2),
            "bmr_start_kcal": bmr.round(0),
            "base_calories": np.where(male_mask, 2500, 2000),
            "genetic_risk_diabetes": genetic_diabetes.round(3),
            "genetic_risk_cvd": genetic_cvd.round(3),
            "genetic_risk_cancer": genetic_cancer.round(3),
            "genetic_risk_stroke": genetic_stroke.round(3),
            "genetic_risk_nafld": genetic_nafld.round(3),
            "genetic_risk_cirrhosis": genetic_cirrhosis.round(3),
            "initial_hdl_mgdl": hdl.round(1),
            "initial_total_cholesterol_mgdl": tc.round(1),
            "initial_sbp_mmhg": sbp.round(1),
            "initial_hba1c_percent": hba1c.round(2),
            "lipid_factor_hdl": lipid_factor_hdl.round(3),
            "lipid_factor_tc": lipid_factor_tc.round(3),
        })

        return cohort

    def _generate_muscle_mass(self, sexes: np.ndarray) -> np.ndarray:
        """Generate baseline muscle mass factor using sex-specific truncated normals."""
        n_people = len(sexes)
        n_m = np.sum(sexes == "M")
        n_f = n_people - n_m

        muscle_m = generate_truncnorm(
            self.rng, loc=12, scale=3, low=0, high=20, size=n_m
        )
        muscle_f = generate_truncnorm(
            self.rng, loc=8, scale=2.5, low=0, high=20, size=n_f
        )

        return np.concatenate([muscle_m, muscle_f])

    def _generate_hdl(self, n_m: int, n_f: int) -> np.ndarray:
        """Generate HDL cholesterol values for males and females."""
        hdl_m = generate_truncnorm(
            self.rng, loc=48, scale=12, low=20, high=100, size=n_m
        )
        hdl_f = generate_truncnorm(
            self.rng, loc=58, scale=12, low=20, high=100, size=n_f
        )
        return np.concatenate([hdl_m, hdl_f])

    def _calculate_bmr_vectorized(
            self,
            weight_kg: np.ndarray,
            height_cm: np.ndarray,
            age: np.ndarray,
            sex: np.ndarray,
            muscle_mass: np.ndarray,
    ) -> np.ndarray:
        """Calculate BMR with muscle mass correction (vectorized)."""
        # Mifflin-St Jeor base
        is_male = sex == "M"
        bmr = np.where(
            is_male,
            10 * weight_kg + 6.25 * height_cm - 5 * age + 5,
            10 * weight_kg + 6.25 * height_cm - 5 * age - 161
        )

        # Muscle correction
        muscle_kg = muscle_mass * self.mm_to_weight_kg
        typical_muscle = self.typical_ratio * weight_kg
        correction = self.bmr_correction * (muscle_kg - typical_muscle)

        return bmr + correction

    def _generate_lipid_factors(
            self,
            n: int,
            correlation: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate correlated lipid factors for HDL and TC."""
        mean = [0, 0]
        cov = [[1, correlation], [correlation, 1]]
        factors = self.rng.multivariate_normal(mean, cov, n)
        return (
            np.clip(factors[:, 0], -2, 2),
            np.clip(factors[:, 1], -2, 2),
        )
