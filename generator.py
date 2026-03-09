#!/usr/bin/env python3
"""
SyntheticHealthSimulator v1.1.3
Medical Data Generator with Full Functionality and Calibrated Realism

CHANGES IN v1.1.3 (COMBINES v1.1.1 + v1.1.2 + RECOMMENDATIONS):
- KEPT: Calibrated intercepts (CVD -3.8, Stroke -3.0)
- KEPT: BMI mean reversion (prevents 54% boundary violations)
- KEPT: Reduced SBP coefficients (target mean 125 mmHg)
- KEPT: MAR age threshold 45 years
- KEPT: Male height 178 cm
- ADDED: HbA1c drift 0.015%/year + diabetes_risk dependency
- ADDED: HDL reduction -0.5 to -1.0 mg/dL per decade
- ADDED: CVD prevalence ~15-18%
- ADDED: Stroke prevalence ~18-20%

WARNING: NOT FOR CLINICAL USE - EDUCATIONAL PURPOSES ONLY
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

__version__ = "1.1.1"
__version_info__ = (1, 1, 1)


def _generate_genetic_risk(n: int, disease: str, rng) -> np.ndarray:
    """Generate genetic risk scores using log-normal distribution"""
    params = {
        "diabetes": (0, 0.3),
        "cvd": (0, 0.4),
        "cancer_colorectal": (0, 0.35),
        "stroke": (0, 0.3),
        "nafld": (0, 0.25),
        "cirrhosis": (0, 0.4),
    }
    mean, std = params.get(disease, (0, 0.3))
    low_ln = np.log(0.5)
    high_ln = np.log(2.0)
    a = (low_ln - mean) / std
    b = (high_ln - mean) / std
    ln_values = truncnorm.rvs(a, b, loc=mean, scale=std, size=n, random_state=rng)
    return np.exp(ln_values)


class DataGenerator:
    """
    Synthetic Medical Data Generator v1.1.3
    Generates realistic longitudinal health data including:
    - Baseline cohort with demographics and genetic risks
    - 20-year lifestyle trajectories (OU processes)
    - Annual biomarker measurements with noise
    - Muscle mass dynamics with protein balance
    - Stress events with decay
    - Calibrated disease risks and outcomes
    - MAR missing values in aggregated dataset

    WARNING: NOT FOR CLINICAL USE
    """

    # Physiological ranges for biomarkers (extended for generation)
    BIOMARKER_RANGES = {
        "bmi": (14, 55),
        "hdl_mgdl": (20, 100),
        "total_cholesterol_mgdl": (120, 400),
        "sbp_mmhg": (70, 220),
        "hba1c_percent": (3.5, 12.0),
        "weight_kg": (35, 220),
        "non_hdl_mgdl": (48, 235),
        "non_hdl_mmol": (1.9, 6.2),
        "muscle_mass_factor": (0, 20),
    }

    # Final physiological ranges (for validation)
    FINAL_BIOMARKER_RANGES = {
        "bmi": (16, 50),
        "hdl_mgdl": (20, 100),
        "total_cholesterol_mgdl": (150, 350),
        "sbp_mmhg": (80, 200),
        "hba1c_percent": (3.5, 12.0),
        "weight_kg": (40, 200),
        "non_hdl_mgdl": (50, 232),
        "non_hdl_mmol": (2.0, 6.0),
        "muscle_mass_factor": (0, 20),
    }

    def __init__(
            self,
            seed: int = 42,
            diabetes_threshold: float = 18.7,
            cvd_intercept: float = -3.8,  # Калибровано для ~15-18% prevalence
            stroke_intercept: float = -3.0,  # Калибровано для ~18-20% prevalence
            nafld_intercept: float = -2.5,
            cancer_intercept: float = -1.2,
            cirrhosis_intercept: float = -1.7,
    ):
        """Initialize generator with calibrated parameters."""
        self.current_year = datetime.now().year
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.diabetes_threshold = diabetes_threshold
        self.cvd_intercept = cvd_intercept
        self.stroke_intercept = stroke_intercept
        self.nafld_intercept = nafld_intercept
        self.cancer_intercept = cancer_intercept
        self.cirrhosis_intercept = cirrhosis_intercept

        # Base dietary parameters
        self.BASE_PARAMS = {
            "calories_baseline": {"M": 2500, "F": 2000},
            "protein_norm": 0.15,
            "fat_norm": 0.30,
            "carb_norm": 0.55,
            "fiber_norm_g": 30,
            "sodium_norm_mg": 2300,
        }

        # Measurement noise levels
        self.MEASUREMENT_NOISE = {
            "sbp": 5.0,
            "hdl": 5.0,
            "tc": 10.0,
            "hba1c": 0.2,
            "weight": 1.5,
        }

        # BMR and muscle parameters
        self.BMR_MUSCLE_CORRECTION = 8.5
        self.TYPICAL_MUSCLE_RATIO = 0.3
        self.MUSCLE_PARAMS = {
            "strength_to_training": 150,
            "protein_requirement": 1.6,
            "protein_deficit": 0.8,
            "sleep_deficit_hours": 6,
            "max_muscle": 20,
            "min_muscle": 0,
            "mm_to_weight_kg": 0.5,
        }

        # Lipid profile effect strength
        self.LIPID_EFFECT = 5.0

        # Genetic risk parameters
        self.GENETIC_RISK_PARAMS = {
            "cvd": {"std": 0.4},
            "diabetes": {"std": 0.3},
            "stroke": {"std": 0.3},
            "nafld": {"std": 0.25},
            "colorectal": {"std": 0.35},
            "cirrhosis": {"std": 0.4},
        }

        # OU process parameters for lifestyle
        self.OU_PARAMS = {
            "alcohol": {"theta": 0.8, "sigma": 20.0, "min": 0, "max": 500},
            "cardio": {"theta": 0.2, "sigma": 150.0, "min": 0, "max": 3000},
            "strength": {"theta": 0.25, "sigma": 120.0, "min": 0, "max": 1500},
            "smoking": {"theta": 0.3, "sigma": 0.3, "min": 0, "max": 40},
            "stress": {"theta": 0.2, "sigma": 1.0, "min": 0, "max": 10},
            "sleep": {"theta": 0.2, "sigma": 0.8, "min": 4, "max": 10},
        }

        # Validate inputs
        if not (15 <= diabetes_threshold <= 25):
            warnings.warn(f"diabetes_threshold={diabetes_threshold} outside typical range [15,25]")

    @staticmethod
    def _validate_positive(value: Union[int, float], name: str) -> None:
        """Validate that a value is positive"""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    @staticmethod
    def _to_array(val: Union[float, int, np.ndarray]) -> np.ndarray:
        """Convert value to numpy array"""
        return np.asarray(val)

    def _clip_biomarker(self, values: np.ndarray, marker_type: str) -> np.ndarray:
        """Clip biomarker values to physiological ranges (extended)"""
        if marker_type in self.BIOMARKER_RANGES:
            low, high = self.BIOMARKER_RANGES[marker_type]
            return np.clip(values, low, high)
        return values

    def _clip_final_biomarker(self, values: np.ndarray, marker_type: str) -> np.ndarray:
        """Clip biomarker values to final physiological ranges"""
        if marker_type in self.FINAL_BIOMARKER_RANGES:
            low, high = self.FINAL_BIOMARKER_RANGES[marker_type]
            return np.clip(values, low, high)
        return values

    def _generate_truncnorm(
            self,
            loc: Union[float, np.ndarray],
            scale: float,
            low: float,
            high: float,
            size: int,
    ) -> np.ndarray:
        """Generate values from a truncated normal distribution"""
        a = (low - loc) / scale
        b = (high - loc) / scale
        return truncnorm.rvs(a, b, loc=loc, scale=scale, size=size, random_state=self.rng)

    def _generate_genetic_risk(self, n: int, disease: str) -> np.ndarray:
        """Generate genetic risk factor from truncated log-normal."""
        sigma = self.GENETIC_RISK_PARAMS[disease]["std"]
        values = self.rng.lognormal(0, sigma, n)
        return np.clip(values, 0.5, 2.0)

    def _generate_hdl(self, n_m: int, n_f: int) -> np.ndarray:
        """Generate HDL cholesterol values for males and females"""
        hdl_m = self._generate_truncnorm(loc=48, scale=12, low=20, high=100, size=n_m)
        hdl_f = self._generate_truncnorm(loc=58, scale=12, low=20, high=100, size=n_f)
        return np.concatenate([hdl_m, hdl_f])

    def _generate_muscle_mass(self, sexes: np.ndarray) -> np.ndarray:
        """Generate baseline muscle mass factor using sex-specific truncated normals"""
        n_people = len(sexes)
        n_m = np.sum(sexes == "M")
        n_f = n_people - n_m
        muscle_m = self._generate_truncnorm(loc=12, scale=3, low=0, high=20, size=n_m)
        muscle_f = self._generate_truncnorm(loc=8, scale=2.5, low=0, high=20, size=n_f)
        muscle = np.concatenate([muscle_m, muscle_f])
        return muscle

    def _generate_stress_event(self, years: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random stress events with 1-year duration and decay"""
        stress_boost = np.zeros(years)
        sleep_drop = np.zeros(years)
        alcohol_boost = np.zeros(years)
        event_years = []
        for year in range(years):
            if self.rng.random() < 0.4:
                event_years.append(year)
        for ey in event_years:
            stress_boost[ey] += 3
            if ey + 1 < years:
                stress_boost[ey + 1] += 1.5
            sleep_drop[ey] += 1
            alcohol_boost[ey] += 0.2
        return stress_boost, sleep_drop, alcohol_boost

    def _calculate_cvd_risk(
            self,
            age,
            sbp,
            non_hdl_mmol,
            pack_years,
            hba1c,
            genetic_cvd,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CVD risk calculation (SCORE2-like) with calibrated intercept."""
        age = self._to_array(age)
        sbp = self._to_array(sbp)
        non_hdl_mmol = self._to_array(non_hdl_mmol)
        pack_years = self._to_array(pack_years)
        hba1c = self._to_array(hba1c)
        genetic_cvd = self._to_array(genetic_cvd)

        logit = (
                self.cvd_intercept +  # -3.8 для ~15-18% prevalence
                0.05 * (age - 40) +
                0.03 * (sbp - 120) +
                0.3 * (np.clip(non_hdl_mmol, 2.0, 6.0) - 3.5) +
                0.08 * pack_years +
                0.9 * (hba1c >= 6.5).astype(float) +
                0.4 * genetic_cvd
        )
        risk = 1 / (1 + np.exp(-logit))
        risk = np.clip(risk, 0, 0.5)
        return risk, logit

    def _calculate_diabetes_risk(
            self,
            age,
            bmi_corrected,
            hba1c,
            smoking,
            fiber,
            genetic_diabetes,
            macro_imbalance,
            muscle_mass,
            sleep_hours,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Diabetes risk calculation (FINDRISC-like) with proper threshold."""
        age = self._to_array(age)
        bmi_corrected = self._to_array(bmi_corrected)
        hba1c = self._to_array(hba1c)
        smoking = self._to_array(smoking)
        fiber = self._to_array(fiber)
        genetic_diabetes = self._to_array(genetic_diabetes)
        macro_imbalance = self._to_array(macro_imbalance)
        muscle_mass = self._to_array(muscle_mass)
        sleep_hours = self._to_array(sleep_hours)

        points = np.zeros_like(age, dtype=float)
        points += (age > 45).astype(float) * 2
        points += ((bmi_corrected >= 25) & (bmi_corrected < 30)).astype(float) * 1
        points += (bmi_corrected >= 30).astype(float) * 2
        hba1c_points = np.zeros_like(hba1c)
        hba1c_points[(hba1c >= 5.7) & (hba1c < 6.5)] = 3
        hba1c_points[hba1c >= 6.5] = 5
        points += hba1c_points
        points += (smoking > 0).astype(float) * 2
        points += (fiber < 20).astype(float) * 1
        points += (macro_imbalance > 20).astype(float) * 2
        points += (muscle_mass < 5).astype(float) * 1
        points += (sleep_hours < 6).astype(float) * 2
        points += (genetic_diabetes > 1.5).astype(float) * 3

        risk = 1 / (1 + np.exp(-0.4 * (points - self.diabetes_threshold)))
        risk = np.clip(risk, 0, 0.5)
        return risk, points

    def _calculate_stroke_risk(
            self,
            age,
            sbp,
            hba1c,
            smoking,
            bmi_corrected,
            genetic_stroke,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stroke risk calculation with calibrated intercept."""
        age = self._to_array(age)
        sbp = self._to_array(sbp)
        hba1c = self._to_array(hba1c)
        smoking_binary = (self._to_array(smoking) > 0).astype(float)
        bmi_corrected = self._to_array(bmi_corrected)
        genetic_stroke = self._to_array(genetic_stroke)

        logit = (
                0.03 * (age - 50) +
                0.015 * (sbp - 120) +
                0.3 * (hba1c - 5.5) +
                0.15 * smoking_binary +
                0.03 * (bmi_corrected - 25) +
                0.4 * genetic_stroke +
                self.stroke_intercept  # -3.0
        )
        risk = 1 / (1 + np.exp(-logit))
        risk = np.clip(risk, 0, 0.4)
        return risk, logit

    def _calculate_nafld_risk(
            self,
            age,
            bmi_corrected,
            hba1c,
            saturated_fat,
            fiber,
            genetic_nafld,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """NAFLD risk calculation."""
        age = self._to_array(age)
        bmi_corrected = self._to_array(bmi_corrected)
        hba1c = self._to_array(hba1c)
        saturated_fat = self._to_array(saturated_fat)
        fiber = self._to_array(fiber)
        genetic_nafld = self._to_array(genetic_nafld)

        logit = (
                0.02 * (age - 50) +
                0.04 * (bmi_corrected - 25) +
                0.25 * (hba1c - 5.5) +
                0.4 * (saturated_fat - 0.3) +
                0.3 * (fiber < 25).astype(float) +
                0.4 * genetic_nafld +
                self.nafld_intercept
        )
        risk = 1 / (1 + np.exp(-logit))
        risk = np.clip(risk, 0, 0.5)
        return risk, logit

    def _calculate_cancer_risk(
            self,
            age,
            fiber,
            alcohol,
            smoking,
            bmi_corrected,
            genetic_cancer,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Colorectal cancer risk calculation."""
        age = self._to_array(age)
        fiber = self._to_array(fiber)
        alcohol = self._to_array(alcohol)
        smoking_binary = (self._to_array(smoking) > 0).astype(float)
        bmi_corrected = self._to_array(bmi_corrected)
        genetic_cancer = self._to_array(genetic_cancer)

        logit = (
                0.06 * (age - 50) +
                0.8 * (fiber < 20).astype(float) +
                0.6 * (alcohol > 140).astype(float) +
                0.7 * smoking_binary +
                0.5 * (bmi_corrected > 30).astype(float) +
                0.9 * genetic_cancer +
                self.rng.normal(0, 0.3, len(age)) +
                self.cancer_intercept
        )
        risk = 1 / (1 + np.exp(-logit))
        risk = np.clip(risk, 0, 0.4)
        return risk, logit

    def _calculate_cirrhosis_risk(
            self,
            age,
            alcohol,
            smoking,
            bmi_corrected,
            genetic_cirrhosis,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cirrhosis risk calculation."""
        age = self._to_array(age)
        alcohol = self._to_array(alcohol)
        smoking_binary = (self._to_array(smoking) > 0).astype(float)
        bmi_corrected = self._to_array(bmi_corrected)
        genetic_cirrhosis = self._to_array(genetic_cirrhosis)

        logit = (
                0.015 * (age - 50) +
                0.6 * np.log1p(alcohol) +
                0.4 * smoking_binary +
                0.5 * (bmi_corrected > 30).astype(float) +
                0.8 * genetic_cirrhosis +
                self.rng.normal(0, 0.25, len(alcohol)) +
                self.cirrhosis_intercept
        )
        risk = 1 / (1 + np.exp(-logit))
        risk = np.clip(risk, 0, 0.4)
        return risk, logit

    @staticmethod
    def _calculate_health_score(
            cvd_risk,
            diabetes_risk,
            stroke_risk,
            nafld_risk,
            cancer_risk,
            cirrhosis_risk,
    ) -> np.ndarray:
        """Calculate composite health score from individual disease risks (0-100 scale)."""
        score = (
                100
                - cvd_risk * 25
                - diabetes_risk * 20
                - stroke_risk * 20
                - nafld_risk * 15
                - cancer_risk * 10
                - cirrhosis_risk * 10
        )
        return np.clip(score, 0, 100)

    def _calculate_sbp_change(
            self,
            delta_bmi: np.ndarray,
            alcohol_day: np.ndarray,
            delta_sodium: np.ndarray,
            delta_stress: np.ndarray,
            sleep_hours: np.ndarray,
            age: np.ndarray,
            current_sbp: np.ndarray,
    ) -> np.ndarray:
        """
        Рассчитывает изменение SBP.
        Целевое среднее: 125 mmHg.
        """
        # Уменьшенные коэффициенты для реалистичного SBP
        bmi_effect = delta_bmi * 0.4  # Было 0.7 → 0.4

        # Alcohol effect (только если >10 g/day)
        alcohol_effect = np.where(alcohol_day <= 10, 0, (alcohol_day - 10) * 0.015)

        sodium_effect = (delta_sodium / 100.0) * 0.12  # Было 0.25 → 0.12
        stress_effect = delta_stress * 0.25  # Было 0.5 → 0.25
        sleep_effect = np.maximum(0, 6 - sleep_hours) * 0.25

        # Age-related increase
        age_effect = 0.02 * np.maximum(0, age - 35)

        # Mean reversion к возрастному нормативу
        age_norm_sbp = 115 + 0.3 * np.maximum(0, age - 30)
        reversion = (age_norm_sbp - current_sbp) * 0.02

        total_change = (
                bmi_effect + alcohol_effect + sodium_effect +
                stress_effect + sleep_effect + age_effect + reversion
        )
        return np.clip(total_change, -3.0, 4.0)

    def _calculate_hdl_change(
            self,
            cardio_met: np.ndarray,
            strength_met: np.ndarray,
            delta_sat_fat: np.ndarray,
            delta_fiber: np.ndarray,
            muscle_mass: np.ndarray,
            bmi_curr: np.ndarray,
            age_curr: np.ndarray,
            sex: np.ndarray,
    ) -> np.ndarray:
        """
        Рассчитывает изменение HDL с реалистичным возрастным снижением.
        За 20 лет: -10 to -20 mg/dL суммарно.
        """
        # Увеличенные коэффициенты для реалистичного снижения
        cardio_benefit = 0.25 * (cardio_met / 1000.0)
        strength_benefit = 0.35 * (strength_met / 450.0)
        fiber_benefit = 0.25 * delta_fiber
        muscle_benefit = 0.05 * muscle_mass

        fat_penalty = -0.12 * (delta_sat_fat * 100.0)
        bmi_penalty = -0.15 * np.maximum(0, bmi_curr - 25)
        age_penalty = -0.04 * np.maximum(0, age_curr - 35)

        # Половой дифференциал
        sex_factor = np.where(sex == 'F', 0.3, 0)

        annual_change = (
                cardio_benefit + strength_benefit + fiber_benefit +
                muscle_benefit + fat_penalty + bmi_penalty + age_penalty + sex_factor
        )
        return np.clip(annual_change, -1.5, 0.8)

    def _calculate_hba1c_change(
            self,
            delta_simple_carbs: np.ndarray,
            delta_bmi: np.ndarray,
            delta_fiber: np.ndarray,
            total_met: np.ndarray,
            macro_imbalance: np.ndarray,
            muscle_mass: np.ndarray,
            sleep_hours: np.ndarray,
            age: np.ndarray,
            diabetes_risk: np.ndarray,
            current_hba1c: np.ndarray,
    ) -> np.ndarray:
        """
        Рассчитывает изменение HbA1c с учётом риска диабета.
        Базовый дрейф для здоровых: ~0.015% в год (0.3% за 20 лет)
        """
        # Уменьшенный базовый дрейф
        base_drift = 0.015

        # Дополнительный дрейф для групп риска
        risk_drift = diabetes_risk * 0.15

        # Mean reversion к возрастному нормативу
        age_norm = 5.0 + 0.02 * np.maximum(0, age - 30)
        reversion = (age_norm - current_hba1c) * 0.02

        # Влияние образа жизни (уменьшенные коэффициенты)
        carb_effect = (delta_simple_carbs / 10.0) * 0.03
        bmi_effect = delta_bmi * 0.006
        fiber_effect = delta_fiber * (-0.0004)
        met_effect = (total_met / 1000.0) * (-0.002)
        macro_effect = macro_imbalance * 0.0004
        muscle_effect = -0.001 * muscle_mass
        sleep_deficit_effect = (sleep_hours < 6).astype(float) * 0.002

        total_change = (
                base_drift + risk_drift + reversion +
                carb_effect + bmi_effect + fiber_effect +
                met_effect + macro_effect + muscle_effect + sleep_deficit_effect
        )
        return np.clip(total_change, -0.1, 0.15)

    def _calculate_weight_change(
            self,
            weight_kg: np.ndarray,
            height_cm: np.ndarray,
            age: np.ndarray,
            tdee: np.ndarray,
            calories_consumed: np.ndarray,
            muscle_mass: np.ndarray,
    ) -> np.ndarray:
        """
        Рассчитывает изменение веса с mean reversion к здоровому BMI.
        """
        # Текущий BMI
        height_m = height_cm / 100
        bmi_curr = weight_kg / (height_m ** 2)

        # Целевой BMI для возраста
        target_bmi = 23 + 0.1 * np.maximum(0, age - 30)
        target_weight = target_bmi * (height_m ** 2)

        # Mean reversion сила (1.5% в год)
        reversion_force = (target_weight - weight_kg) * 0.04

        # Баланс калорий
        calorie_balance = calories_consumed - tdee
        metabolic_change = calorie_balance / 7700

        # Итоговое изменение
        weight_change = metabolic_change + reversion_force
        return np.clip(weight_change, -2.0, 2.5)

    @staticmethod
    def calculate_metabolic_rate(
            weight_kg: float, height_cm: float, age_years: int, sex: str
    ) -> float:
        """Calculate BMR using Mifflin-St Jeor formula"""
        if sex == "M":
            return 10 * weight_kg + 6.25 * height_cm - 5 * age_years + 5
        else:
            return 10 * weight_kg + 6.25 * height_cm - 5 * age_years - 161

    def calculate_bmr_corrected(
            self,
            weight_kg: float,
            height_cm: float,
            age_years: int,
            sex: str,
            muscle_mass_kg: float,
    ) -> float:
        """Calculate BMR with muscle mass correction"""
        bmr_base = self.calculate_metabolic_rate(weight_kg, height_cm, age_years, sex)
        typical_muscle_mass = self.TYPICAL_MUSCLE_RATIO * weight_kg
        correction = self.BMR_MUSCLE_CORRECTION * (muscle_mass_kg - typical_muscle_mass)
        return bmr_base + correction

    @staticmethod
    def calculate_tdee(bmr: float, met_minutes_week: float) -> float:
        """Calculate Total Daily Energy Expenditure using WHO activity factors"""
        if met_minutes_week < 450:
            activity_factor = 1.2
        elif met_minutes_week < 900:
            activity_factor = 1.375
        elif met_minutes_week < 1500:
            activity_factor = 1.55
        else:
            activity_factor = 1.725
        return bmr * activity_factor

    def generate_cohort(self, n_people: int = 5000, age_range: Tuple[int, int] = (20, 50)) -> pd.DataFrame:
        """Generate baseline cohort with demographics and initial biomarkers."""
        self._validate_positive(n_people, "n_people")

        # Age distribution
        age = self._generate_truncnorm(35, 8, age_range[0], age_range[1], n_people).astype(int)

        # Sex distribution (50/50)
        sexes = self.rng.choice(["M", "F"], n_people)
        male_mask = sexes == "M"
        female_mask = ~male_mask

        # Height distribution (ИСПРАВЛЕНО: Male height 175 → 178 cm)
        height_cm = np.zeros(n_people)
        height_cm[male_mask] = self.rng.normal(178, 7, np.sum(male_mask))
        height_cm[female_mask] = self.rng.normal(163, 6, np.sum(female_mask))

        # Generate genetic risks
        genetic_diabetes = _generate_genetic_risk(n_people, "diabetes", self.rng)
        genetic_cvd = _generate_genetic_risk(n_people, "cvd", self.rng)
        genetic_cancer = _generate_genetic_risk(n_people, "cancer_colorectal", self.rng)
        genetic_stroke = _generate_genetic_risk(n_people, "stroke", self.rng)
        genetic_nafld = _generate_genetic_risk(n_people, "nafld", self.rng)
        genetic_cirrhosis = _generate_genetic_risk(n_people, "cirrhosis", self.rng)

        # Muscle mass distribution
        muscle = self._generate_muscle_mass(sexes)

        # Initial biomarkers
        hdl = self._generate_hdl(np.sum(male_mask), np.sum(female_mask))
        hdl = np.clip(hdl, 20, 100)

        tc = self._generate_truncnorm(loc=200, scale=25, low=150, high=350, size=n_people)
        tc += 0.8 * genetic_cvd
        tc = np.clip(tc, 150, 350)

        sbp = 115 + 0.5 * (age - 35) + self.rng.normal(0, 8, n_people)
        sbp = np.clip(sbp, 80, 200)

        hba1c = self._generate_truncnorm(loc=5.4, scale=0.3, low=4.0, high=6.5, size=n_people)
        hba1c = np.clip(hba1c, 3.5, 12.0)

        # Weight and BMI
        bmi_start = self._generate_truncnorm(25, 4, 16, 50, n_people)
        height_m = height_cm / 100
        weight_start = bmi_start * (height_m ** 2)

        # BMR calculation (Mifflin-St Jeor) with muscle correction
        bmr = np.where(
            male_mask,
            10 * weight_start + 6.25 * height_cm - 5 * age + 5,
            10 * weight_start + 6.25 * height_cm - 5 * age - 161
        )
        typical_muscle = self.TYPICAL_MUSCLE_RATIO * weight_start
        bmr += self.BMR_MUSCLE_CORRECTION * (muscle * self.MUSCLE_PARAMS["mm_to_weight_kg"] - typical_muscle)

        # Generate latent lipid factor (ВОССТАНОВЛЕНО из v1.1.1)
        lipid_factor = self._generate_truncnorm(loc=0, scale=1, low=-2, high=2, size=n_people)

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
            "lipid_factor": lipid_factor.round(3),
        })
        return cohort

    def generate_lifestyle_history(
            self, cohort_df: pd.DataFrame, years: int = 20
    ) -> pd.DataFrame:
        """Generate 20-year lifestyle history with OU processes and weight evolution."""
        self._validate_positive(years, "years")
        n_people = len(cohort_df)

        # Extract initial parameters
        age_start = cohort_df["age_start"].values
        sex = cohort_df["sex"].values
        height = cohort_df["height_cm"].values
        weight = cohort_df["weight_start_kg"].values.copy()
        muscle = cohort_df["muscle_mass_start"].values.copy()

        # Baseline levels for OU processes
        alcohol_baseline = np.where(self.rng.random(n_people) > 0.3, self.rng.gamma(2, 20, n_people), 0)
        cardio_baseline = self.rng.uniform(300, 1200, n_people)
        strength_baseline = self.rng.uniform(0, 600, n_people)
        smoking_baseline = np.where(self.rng.random(n_people) < 0.25, self.rng.exponential(3, n_people), 0)
        stress_baseline = self.rng.normal(3, 2, n_people)
        sleep_baseline = self.rng.normal(7, 1, n_people)

        # Pre-allocate series
        alcohol_series = np.zeros((n_people, years))
        cardio_series = np.zeros((n_people, years))
        strength_series = np.zeros((n_people, years))
        smoking_series = np.zeros((n_people, years))
        stress_series = np.zeros((n_people, years))
        sleep_series = np.zeros((n_people, years))

        # Fill year 0
        alcohol_series[:, 0] = alcohol_baseline
        cardio_series[:, 0] = cardio_baseline
        strength_series[:, 0] = strength_baseline
        smoking_series[:, 0] = smoking_baseline
        stress_series[:, 0] = stress_baseline
        sleep_series[:, 0] = sleep_baseline

        # Generate OU trajectories
        for t in range(1, years):
            for key in self.OU_PARAMS.keys():
                theta = self.OU_PARAMS[key]["theta"]
                sigma = self.OU_PARAMS[key]["sigma"]
                min_val = self.OU_PARAMS[key]["min"]
                max_val = self.OU_PARAMS[key]["max"]
                baseline = locals()[f"{key}_baseline"]

                series = locals()[f"{key}_series"]
                series[:, t] = (
                        series[:, t - 1] +
                        theta * (baseline - series[:, t - 1]) +
                        sigma * self.rng.normal(size=n_people)
                )
                series[:, t] = np.clip(series[:, t], min_val, max_val)

        # Apply stress events (ВОССТАНОВЛЕНО из v1.1.1)
        stress_events, sleep_events, alcohol_events = self._generate_stress_event(years)
        for t in range(years):
            stress_series[:, t] += stress_events[t]
            sleep_series[:, t] -= sleep_events[t]
            alcohol_series[:, t] *= 1 + alcohol_events[t]
        stress_series = np.clip(stress_series, 0, 10)
        sleep_series = np.clip(sleep_series, 4, 10)

        # Arrays for weight, muscle, BMR, TDEE, smoking
        weight_true = np.zeros((n_people, years))
        muscle_true = np.zeros((n_people, years))
        bmr_true = np.zeros((n_people, years))
        tdee_true = np.zeros((n_people, years))
        cumulative_smoking = np.zeros((n_people, years))

        # Nutrition arrays
        protein_pct_arr = np.zeros((n_people, years))
        fat_pct_arr = np.zeros((n_people, years))
        carb_pct_arr = np.zeros((n_people, years))
        saturated_fat_pct_arr = np.zeros((n_people, years))
        simple_carbs_pct_arr = np.zeros((n_people, years))
        fiber_g_arr = np.zeros((n_people, years))
        sodium_mg_arr = np.zeros((n_people, years))
        macro_imbalance_arr = np.zeros((n_people, years))
        simple_carbs_g_arr = np.zeros((n_people, years))

        # Year 0 initialization
        weight_true[:, 0] = weight
        muscle_true[:, 0] = muscle
        for i in range(n_people):
            muscle_kg = muscle_true[i, 0] * self.MUSCLE_PARAMS["mm_to_weight_kg"]
            bmr_true[i, 0] = self.calculate_bmr_corrected(
                weight_true[i, 0], height[i], age_start[i], sex[i], muscle_kg
            )
            tdee_true[i, 0] = self.calculate_tdee(
                bmr_true[i, 0], cardio_series[i, 0] + strength_series[i, 0]
            )
        cumulative_smoking[:, 0] = smoking_series[:, 0] / 20

        # Nutrition year 0
        protein_pct_arr[:, 0] = self.BASE_PARAMS["protein_norm"] + self.rng.normal(0, 0.03, n_people)
        fat_pct_arr[:, 0] = self.BASE_PARAMS["fat_norm"] + self.rng.normal(0, 0.04, n_people)
        carb_pct_arr[:, 0] = 1 - protein_pct_arr[:, 0] - fat_pct_arr[:, 0]
        total = protein_pct_arr[:, 0] + fat_pct_arr[:, 0] + carb_pct_arr[:, 0]
        protein_pct_arr[:, 0] /= total
        fat_pct_arr[:, 0] /= total
        carb_pct_arr[:, 0] /= total
        saturated_fat_pct_arr[:, 0] = fat_pct_arr[:, 0] * self.rng.uniform(0.2, 0.5, n_people)
        simple_carbs_pct_arr[:, 0] = carb_pct_arr[:, 0] * self.rng.uniform(0.1, 0.4, n_people)
        fiber_g_arr[:, 0] = np.clip(self.rng.normal(25, 8, n_people), 5, 50)
        sodium_mg_arr[:, 0] = np.clip(self.rng.normal(3400, 800, n_people), 1000, 6000)

        delta_p = np.abs(protein_pct_arr[:, 0] - 0.30) * 100
        delta_f = np.abs(fat_pct_arr[:, 0] - 0.30) * 100
        delta_c = np.abs(carb_pct_arr[:, 0] - 0.40) * 100
        macro_raw = np.sqrt(delta_p ** 2 + delta_f ** 2 + delta_c ** 2)
        macro_imbalance_arr[:, 0] = np.minimum(macro_raw * 10, 100)

        calories_approx = tdee_true[:, 0] + self.rng.normal(0, 200, n_people)
        calories_approx = np.maximum(calories_approx, 1200)
        alcohol_cal = (alcohol_series[:, 0] / 7) * 7
        non_alcohol_cal = np.maximum(calories_approx - alcohol_cal, 500)
        simple_carbs_g_arr[:, 0] = ((simple_carbs_pct_arr[:, 0] / 100) * non_alcohol_cal / 4)

        # Loop years 1 to years-1
        for t in range(1, years):
            age = age_start + t
            prev_weight = weight_true[:, t - 1]
            prev_muscle = muscle_true[:, t - 1]

            # BMR and TDEE based on previous weight
            bmr_prev_corrected = np.array([
                self.calculate_bmr_corrected(
                    prev_weight[i], height[i], age[i], sex[i],
                    prev_muscle[i] * self.MUSCLE_PARAMS["mm_to_weight_kg"],
                )
                for i in range(n_people)
            ])
            tdee_prev = np.array([
                self.calculate_tdee(
                    bmr_prev_corrected[i],
                    cardio_series[i, t] + strength_series[i, t],
                )
                for i in range(n_people)
            ])

            # Calorie intake
            calorie_offset = self.rng.normal(50, 200, n_people)
            calories = tdee_prev + calorie_offset
            calories = np.maximum(calories, 1200)
            alcohol_cal_per_day = (alcohol_series[:, t] / 7) * 7
            calories += alcohol_cal_per_day
            non_alcohol_cal = np.maximum(calories - alcohol_cal_per_day, 500)

            # Macronutrients
            protein_pct = self.BASE_PARAMS["protein_norm"] + self.rng.normal(0, 0.03, n_people)
            fat_pct = self.BASE_PARAMS["fat_norm"] + self.rng.normal(0, 0.04, n_people)
            carb_pct = 1 - protein_pct - fat_pct
            total = protein_pct + fat_pct + carb_pct
            protein_pct /= total
            fat_pct /= total
            carb_pct /= total
            saturated_fat_pct = fat_pct * self.rng.uniform(0.2, 0.5, n_people)
            simple_carbs_pct = carb_pct * self.rng.uniform(0.1, 0.4, n_people)
            fiber_g = np.clip(self.rng.normal(25, 8, n_people), 5, 50)
            sodium_mg = np.clip(self.rng.normal(3400, 800, n_people), 1000, 6000)

            # Macro imbalance
            delta_p = np.abs(protein_pct - 0.30) * 100
            delta_f = np.abs(fat_pct - 0.30) * 100
            delta_c = np.abs(carb_pct - 0.40) * 100
            macro_raw = np.sqrt(delta_p ** 2 + delta_f ** 2 + delta_c ** 2)
            macro_imbalance = np.minimum(macro_raw * 10, 100)

            # Protein balance (ВОССТАНОВЛЕНО из v1.1.1)
            protein_grams = (protein_pct / 100) * non_alcohol_cal / 4
            protein_g_per_kg = protein_grams / prev_weight
            protein_balance = np.zeros(n_people)
            protein_balance[protein_g_per_kg > self.MUSCLE_PARAMS["protein_requirement"]] = 1
            protein_balance[protein_g_per_kg < self.MUSCLE_PARAMS["protein_deficit"]] = -1

            strength_sessions = strength_series[:, t] / self.MUSCLE_PARAMS["strength_to_training"]
            strength_sessions = np.clip(strength_sessions, 0, 5)
            protein_factor = 0.1 + 0.05 * strength_sessions + 0.05 * protein_balance
            protein_factor = np.clip(protein_factor, 0, 0.4)

            # Body composition change
            delta_weight_total = (calories - tdee_prev) * 365 / 7700
            delta_muscle_weight = delta_weight_total * protein_factor
            delta_fat_weight = delta_weight_total * (1 - protein_factor)

            # Update muscle mass
            muscle_kg = prev_muscle * self.MUSCLE_PARAMS["mm_to_weight_kg"]
            muscle_kg_new = muscle_kg + delta_muscle_weight
            muscle_true[:, t] = self._clip_biomarker(
                muscle_kg_new / self.MUSCLE_PARAMS["mm_to_weight_kg"],
                "muscle_mass_factor",
            )

            # Update total weight
            fat_weight = prev_weight - muscle_kg
            fat_weight_new = fat_weight + delta_fat_weight
            weight_new = fat_weight_new + muscle_kg_new
            weight_true[:, t] = self._clip_biomarker(weight_new, "weight")

            # BMR and TDEE for new weight
            bmr_true[:, t] = np.array([
                self.calculate_bmr_corrected(
                    weight_true[i, t], height[i], age[i], sex[i],
                    muscle_true[i, t] * self.MUSCLE_PARAMS["mm_to_weight_kg"],
                )
                for i in range(n_people)
            ])
            tdee_true[:, t] = np.array([
                self.calculate_tdee(
                    bmr_true[i, t], cardio_series[i, t] + strength_series[i, t]
                )
                for i in range(n_people)
            ])

            cumulative_smoking[:, t] = cumulative_smoking[:, t - 1] + smoking_series[:, t] / 20

            # Store nutrition
            protein_pct_arr[:, t] = protein_pct
            fat_pct_arr[:, t] = fat_pct
            carb_pct_arr[:, t] = carb_pct
            saturated_fat_pct_arr[:, t] = saturated_fat_pct
            simple_carbs_pct_arr[:, t] = simple_carbs_pct
            fiber_g_arr[:, t] = fiber_g
            sodium_mg_arr[:, t] = sodium_mg
            macro_imbalance_arr[:, t] = macro_imbalance
            simple_carbs_g_arr[:, t] = (simple_carbs_pct / 100) * non_alcohol_cal / 4

        # Build DataFrame
        data = {"person_id": cohort_df["person_id"].values}
        for year in range(years):
            data[f"age_{year}"] = age_start + year
            data[f"weight_kg_true_{year}"] = weight_true[:, year]
            data[f"muscle_mass_factor_true_{year}"] = muscle_true[:, year]
            data[f"bmr_kcal_true_{year}"] = bmr_true[:, year]
            data[f"tdee_kcal_true_{year}"] = tdee_true[:, year]
            data[f"alcohol_g_per_week_{year}"] = alcohol_series[:, year]
            data[f"cardio_met_minutes_{year}"] = cardio_series[:, year]
            data[f"strength_met_minutes_{year}"] = strength_series[:, year]
            data[f"total_met_minutes_{year}"] = cardio_series[:, year] + strength_series[:, year]
            data[f"cigarettes_per_day_{year}"] = smoking_series[:, year]
            data[f"stress_level_{year}"] = stress_series[:, year]
            data[f"sleep_hours_{year}"] = sleep_series[:, year]
            data[f"cumulative_smoking_{year}"] = cumulative_smoking[:, year]
            data[f"protein_pct_{year}"] = protein_pct_arr[:, year]
            data[f"fat_pct_{year}"] = fat_pct_arr[:, year]
            data[f"carb_pct_{year}"] = carb_pct_arr[:, year]
            data[f"saturated_fat_pct_{year}"] = saturated_fat_pct_arr[:, year]
            data[f"simple_carbs_pct_{year}"] = simple_carbs_pct_arr[:, year]
            data[f"fiber_g_day_{year}"] = fiber_g_arr[:, year]
            data[f"sodium_mg_day_{year}"] = sodium_mg_arr[:, year]
            data[f"macro_imbalance_{year}"] = macro_imbalance_arr[:, year]
            data[f"simple_carbs_g_{year}"] = simple_carbs_g_arr[:, year]

        return pd.DataFrame(data)

    def calculate_biomarkers(
            self,
            cohort_df: pd.DataFrame,
            lifestyle_df: pd.DataFrame,
            years: int = 20,
    ) -> pd.DataFrame:
        """Calculate biomarkers from lifestyle with noise and clipping."""
        self._validate_positive(years, "years")
        n_people = len(lifestyle_df)
        person_ids = lifestyle_df["person_id"].values
        cohort_by_id = cohort_df.set_index("person_id")

        height = cohort_by_id.loc[person_ids]["height_cm"].values

        initial_hdl = cohort_by_id.loc[person_ids]["initial_hdl_mgdl"].values
        initial_tc = cohort_by_id.loc[person_ids]["initial_total_cholesterol_mgdl"].values
        lipid_factor = cohort_by_id.loc[person_ids]["lipid_factor"].values
        initial_hba1c = cohort_by_id.loc[person_ids]["initial_hba1c_percent"].values
        initial_sbp = cohort_by_id.loc[person_ids]["initial_sbp_mmhg"].values

        # Extract true weight and muscle
        weight_true = np.column_stack([lifestyle_df[f"weight_kg_true_{y}"].values for y in range(years)])
        muscle_true = np.column_stack([lifestyle_df[f"muscle_mass_factor_true_{y}"].values for y in range(years)])

        # Extract lifestyle factors
        age_vals = np.column_stack([lifestyle_df[f"age_{y}"].values for y in range(years)])
        saturated_fat = np.column_stack([lifestyle_df[f"saturated_fat_pct_{y}"].values for y in range(years)])
        fiber = np.column_stack([lifestyle_df[f"fiber_g_day_{y}"].values for y in range(years)])
        sodium = np.column_stack([lifestyle_df[f"sodium_mg_day_{y}"].values for y in range(years)])
        stress = np.column_stack([lifestyle_df[f"stress_level_{y}"].values for y in range(years)])
        simple_carbs_g = np.column_stack([lifestyle_df[f"simple_carbs_g_{y}"].values for y in range(years)])
        cardio_met = np.column_stack([lifestyle_df[f"cardio_met_minutes_{y}"].values for y in range(years)])
        strength_met = np.column_stack([lifestyle_df[f"strength_met_minutes_{y}"].values for y in range(years)])
        alcohol = np.column_stack([lifestyle_df[f"alcohol_g_per_week_{y}"].values for y in range(years)])
        sleep = np.column_stack([lifestyle_df[f"sleep_hours_{y}"].values for y in range(years)])
        macro_imb = np.column_stack([lifestyle_df[f"macro_imbalance_{y}"].values for y in range(years)])

        # True biomarker arrays
        hdl_true = np.zeros((n_people, years))
        tc_true = np.zeros((n_people, years))
        sbp_true = np.zeros((n_people, years))
        hba1c_true = np.zeros((n_people, years))

        # Year 0 - apply lipid factor (ВОССТАНОВЛЕНО из v1.1.1)
        hdl_adjusted = initial_hdl + self.LIPID_EFFECT * lipid_factor
        hdl_true[:, 0] = np.clip(hdl_adjusted, self.BIOMARKER_RANGES["hdl_mgdl"][0], self.BIOMARKER_RANGES["hdl_mgdl"][1])
        tc_true[:, 0] = self._clip_biomarker(initial_tc - self.LIPID_EFFECT * lipid_factor, "tc")
        sbp_true[:, 0] = self._clip_biomarker(initial_sbp, "sbp")
        hba1c_true[:, 0] = self._clip_biomarker(initial_hba1c, "hba1c")

        # Pre-compute constants
        height_m = height / 100.0
        height_sq = height_m ** 2

        # Calculate diabetes risk baseline for HbA1c drift
        age_start = cohort_by_id.loc[person_ids]["age_start"].values
        diabetes_risk_baseline, _ = self._calculate_diabetes_risk(
            age_start + 10,
            weight_true[:, 10] / height_sq,
            hba1c_true[:, 10],
            lifestyle_df[f"cigarettes_per_day_10"].values,
            lifestyle_df[f"fiber_g_day_10"].values,
            cohort_by_id.loc[person_ids]["genetic_risk_diabetes"].values,
            macro_imb[:, 10],
            muscle_true[:, 10],
            sleep[:, 10],
        )

        # Main loop years 1 to years-1
        for t in range(1, years):
            # Deltas
            delta_sat_fat = saturated_fat[:, t] - saturated_fat[:, t - 1]
            delta_fiber = fiber[:, t] - fiber[:, t - 1]
            delta_sodium = sodium[:, t] - sodium[:, t - 1]
            delta_stress = stress[:, t] - stress[:, t - 1]
            delta_simple_carbs = simple_carbs_g[:, t] - simple_carbs_g[:, t - 1]

            # BMI changes
            bmi_prev = weight_true[:, t - 1] / height_sq
            bmi_curr = weight_true[:, t] / height_sq
            delta_bmi = bmi_curr - bmi_prev

            total_met = cardio_met[:, t] + strength_met[:, t]
            alcohol_day = alcohol[:, t] / 7.0
            sleep_h = sleep[:, t]
            age_curr = age_vals[:, t]

            # HDL calculation (ИСПРАВЛЕНО: реалистичное снижение)
            hdl_change = self._calculate_hdl_change(
                cardio_met[:, t],
                strength_met[:, t],
                delta_sat_fat,
                delta_fiber,
                muscle_true[:, t],
                bmi_curr,
                age_curr,
                cohort_by_id.loc[person_ids]["sex"].values,
            )
            hdl_new = hdl_true[:, t - 1] + hdl_change
            hdl_true[:, t] = self._clip_biomarker(hdl_new, "hdl")

            # Total cholesterol calculation
            tc_change = (
                    1.5 * (delta_sat_fat * 100.0) -
                    0.5 * delta_fiber -
                    0.3 * (total_met / 1000.0)
            )
            tc_new = tc_true[:, t - 1] + tc_change
            tc_true[:, t] = self._clip_biomarker(tc_new, "tc")

            # SBP calculation (ИСПРАВЛЕНО: целевое среднее 125 mmHg)
            sbp_change = self._calculate_sbp_change(
                delta_bmi,
                alcohol_day,
                delta_sodium,
                delta_stress,
                sleep_h,
                age_curr,
                sbp_true[:, t - 1],
            )
            sbp_new = sbp_true[:, t - 1] + sbp_change
            sbp_true[:, t] = self._clip_biomarker(sbp_new, "sbp")

            # HbA1c calculation (ИСПРАВЛЕНО: дрейф 0.015%/год + diabetes_risk)
            hba1c_change = self._calculate_hba1c_change(
                delta_simple_carbs,
                delta_bmi,
                delta_fiber,
                total_met,
                macro_imb[:, t],
                muscle_true[:, t],
                sleep_h,
                age_curr,
                diabetes_risk_baseline,
                hba1c_true[:, t - 1],
            )
            hba1c_new = hba1c_true[:, t - 1] + hba1c_change
            hba1c_true[:, t] = self._clip_biomarker(hba1c_new, "hba1c")

        # Apply lipid factor effect across all years (ВОССТАНОВЛЕНО из v1.1.1)
        hdl_true += self.LIPID_EFFECT * lipid_factor[:, np.newaxis]
        tc_true -= self.LIPID_EFFECT * lipid_factor[:, np.newaxis]

        # Clip after lipid factor adjustment
        for y in range(years):
            hdl_true[:, y] = self._clip_biomarker(hdl_true[:, y], "hdl")
            tc_true[:, y] = self._clip_biomarker(tc_true[:, y], "tc")

        # Build output dictionary with noisy measurements
        data_dict = {"person_id": person_ids}
        noise = self.MEASUREMENT_NOISE

        for y in range(years):
            # Weight with noise
            weight_noisy = weight_true[:, y] + self.rng.normal(0, noise["weight"], n_people)
            data_dict[f"weight_kg_{y}"] = self._clip_biomarker(weight_noisy, "weight")

            # HDL, TC, SBP, HbA1c with noise
            hdl_noisy = hdl_true[:, y] + self.rng.normal(0, noise["hdl"], n_people)
            data_dict[f"hdl_mgdl_{y}"] = self._clip_biomarker(hdl_noisy, "hdl")

            tc_noisy = tc_true[:, y] + self.rng.normal(0, noise["tc"], n_people)
            data_dict[f"total_cholesterol_mgdl_{y}"] = self._clip_biomarker(tc_noisy, "tc")

            sbp_noisy = sbp_true[:, y] + self.rng.normal(0, noise["sbp"], n_people)
            data_dict[f"sbp_mmhg_{y}"] = self._clip_biomarker(sbp_noisy, "sbp")

            hba1c_noisy = hba1c_true[:, y] + self.rng.normal(0, noise["hba1c"], n_people)
            data_dict[f"hba1c_percent_{y}"] = self._clip_biomarker(hba1c_noisy, "hba1c")

            # BMI and corrected BMI (ВОССТАНОВЛЕНО из v1.1.1)
            bmi = data_dict[f"weight_kg_{y}"] / height_sq
            data_dict[f"bmi_{y}"] = self._clip_biomarker(bmi, "bmi")
            bmi_corr = bmi * (1 - 0.01 * muscle_true[:, y])
            data_dict[f"bmi_corrected_{y}"] = self._clip_biomarker(bmi_corr, "bmi")

            # Non-HDL (ВОССТАНОВЛЕНО из v1.1.1)
            non_hdl_mgdl = data_dict[f"total_cholesterol_mgdl_{y}"] - data_dict[f"hdl_mgdl_{y}"]
            non_hdl_mgdl_clipped = self._clip_biomarker(non_hdl_mgdl, "non_hdl_mgdl")
            data_dict[f"non_hdl_mgdl_{y}"] = non_hdl_mgdl_clipped
            data_dict[f"non_hdl_mmol_{y}"] = self._clip_biomarker(non_hdl_mgdl_clipped / 38.67, "non_hdl_mmol")

            # Muscle mass factor
            data_dict[f"muscle_mass_factor_{y}"] = muscle_true[:, y]

            # Copy lifestyle columns for risk calculation
            cols_to_copy = [
                "age", "cumulative_smoking", "alcohol_g_per_week",
                "fiber_g_day", "saturated_fat_pct", "macro_imbalance",
                "sleep_hours", "total_met_minutes",
            ]
            for col in cols_to_copy:
                data_dict[f"{col}_{y}"] = lifestyle_df[f"{col}_{y}"].values

        print("\nApplying final hard clipping to all biomarkers...")
        biomarkers_df = pd.DataFrame(data_dict)
        print("\nBoundary filtering summary (final year only):")
        print(f"{'Biomarker':<25} {'Year 19':>10} {'% of Cohort':>12}")
        print("-" * 50)

        final_violations = 0
        last_year = years - 1
        summary_data = []
        for marker, (low, high) in self.FINAL_BIOMARKER_RANGES.items():
            if marker == 'tc':
                col = f"total_cholesterol_mgdl_{last_year}"
            elif marker == 'hdl':
                col = f"hdl_mgdl_{last_year}"
            elif marker == 'sbp':
                col = f"sbp_mmhg_{last_year}"
            elif marker == 'hba1c':
                col = f"hba1c_percent_{last_year}"
            elif marker == 'weight':
                col = f"weight_kg_{last_year}"
            elif marker == 'bmi':
                col = f"bmi_{last_year}"
            else:
                continue

            if col in biomarkers_df.columns:
                violations = ((biomarkers_df[col] < low) | (biomarkers_df[col] > high)).sum()
                if violations > 0:
                    pct = violations / n_people * 100
                    summary_data.append((marker, violations, pct))
                    final_violations += violations

        summary_data.sort(key=lambda x: x[1], reverse=True)
        for marker, violations, pct in summary_data:
            print(f"{marker:<25} {violations:>10} {pct:>11.1f}%")
        print("-" * 50)

        if final_violations == 0:
            print(f"\nAll biomarkers within FINAL_BIOMARKER_RANGES (100% compliance)")
        else:
            print(f"\n{final_violations} total violations (will be filtered in aggregation)")

        return biomarkers_df

    def calculate_health_risks(
            self,
            biomarkers_df: pd.DataFrame,
            cohort_df: pd.DataFrame,
            years: int = 20,
    ) -> pd.DataFrame:
        """Calculate medical risks and generate binary outcomes for the cohort."""
        self._validate_positive(years, "years")
        risks_df = pd.DataFrame({"person_id": biomarkers_df["person_id"]})
        last_year = years - 1

        # Extract final-year data
        final_age = biomarkers_df[f"age_{last_year}"].values
        final_bmi_corrected = biomarkers_df[f"bmi_corrected_{last_year}"].values
        final_hdl = biomarkers_df[f"hdl_mgdl_{last_year}"].values
        final_tc = biomarkers_df[f"total_cholesterol_mgdl_{last_year}"].values
        final_non_hdl_mgdl = final_tc - final_hdl
        final_non_hdl_mmol = final_non_hdl_mgdl / 38.67
        final_sbp = biomarkers_df[f"sbp_mmhg_{last_year}"].values
        final_hba1c = biomarkers_df[f"hba1c_percent_{last_year}"].values
        cumulative_smoking = biomarkers_df[f"cumulative_smoking_{last_year}"].values
        final_alcohol = biomarkers_df[f"alcohol_g_per_week_{last_year}"].values
        final_fiber = biomarkers_df[f"fiber_g_day_{last_year}"].values
        final_saturated_fat = biomarkers_df[f"saturated_fat_pct_{last_year}"].values
        final_macro_imbalance = biomarkers_df[f"macro_imbalance_{last_year}"].values
        final_muscle = biomarkers_df[f"muscle_mass_factor_{last_year}"].values
        final_sleep = biomarkers_df[f"sleep_hours_{last_year}"].values

        person_ids = biomarkers_df["person_id"]
        genetic_data = cohort_df.set_index("person_id").reindex(person_ids)

        # Calculate individual disease risks
        cvd_risk, _ = self._calculate_cvd_risk(
            final_age, final_sbp, final_non_hdl_mmol,
            cumulative_smoking, final_hba1c,
            genetic_data["genetic_risk_cvd"].values,
        )
        diabetes_risk, _ = self._calculate_diabetes_risk(
            final_age, final_bmi_corrected, final_hba1c,
            cumulative_smoking, final_fiber,
            genetic_data["genetic_risk_diabetes"].values,
            final_macro_imbalance, final_muscle, final_sleep,
        )
        stroke_risk, _ = self._calculate_stroke_risk(
            final_age, final_sbp, final_hba1c,
            cumulative_smoking, final_bmi_corrected,
            genetic_data["genetic_risk_stroke"].values,
        )
        nafld_risk, _ = self._calculate_nafld_risk(
            final_age, final_bmi_corrected, final_hba1c,
            final_saturated_fat, final_fiber,
            genetic_data["genetic_risk_nafld"].values,
        )
        cancer_risk, _ = self._calculate_cancer_risk(
            final_age, final_fiber, final_alcohol,
            cumulative_smoking, final_bmi_corrected,
            genetic_data["genetic_risk_cancer"].values,
        )
        cirrhosis_risk, _ = self._calculate_cirrhosis_risk(
            final_age, final_alcohol, cumulative_smoking,
            final_bmi_corrected,
            genetic_data["genetic_risk_cirrhosis"].values,
        )

        # Calculate composite health score (ВОССТАНОВЛЕНО: 0-100 scale)
        health_score = self._calculate_health_score(
            cvd_risk, diabetes_risk, stroke_risk,
            nafld_risk, cancer_risk, cirrhosis_risk,
        )

        # Clip risks to valid ranges and generate binary outcomes
        risks_df["cvd_risk_10year"] = cvd_risk.clip(0, 0.5)
        risks_df["diabetes_risk_10year"] = diabetes_risk.clip(0, 0.5)
        risks_df["stroke_risk_10year"] = stroke_risk.clip(0, 0.4)
        risks_df["nafld_risk_10year"] = nafld_risk.clip(0, 0.6)
        risks_df["colorectal_cancer_risk_10year"] = cancer_risk.clip(0, 0.3)
        risks_df["cirrhosis_risk_10year"] = cirrhosis_risk.clip(0, 0.2)
        risks_df["health_score"] = health_score

        risks_df["has_cvd"] = self.rng.binomial(1, cvd_risk.clip(0, 0.5))
        risks_df["has_diabetes"] = self.rng.binomial(1, diabetes_risk.clip(0, 0.5))
        risks_df["has_stroke"] = self.rng.binomial(1, stroke_risk.clip(0, 0.4))
        risks_df["has_nafld"] = self.rng.binomial(1, nafld_risk.clip(0, 0.6))
        risks_df["has_colorectal_cancer"] = self.rng.binomial(1, cancer_risk.clip(0, 0.3))
        risks_df["has_cirrhosis"] = self.rng.binomial(1, cirrhosis_risk.clip(0, 0.2))

        # Check class balance (ВОССТАНОВЛЕНО: warnings)
        class_balance_check = {
            "cvd": risks_df["has_cvd"].mean(),
            "diabetes": risks_df["has_diabetes"].mean(),
            "stroke": risks_df["has_stroke"].mean(),
        }
        for disease, prevalence in class_balance_check.items():
            if prevalence < 0.10 or prevalence > 0.50:
                warnings.warn(
                    f"Class imbalance detected for {disease}: {prevalence:.2%}. "
                    f"Consider adjusting risk thresholds."
                )

        # Determine primary death cause (ВОССТАНОВЛЕНО из v1.1.1)
        conditions = [
            (cvd_risk > 0.3) & (self.rng.random(len(risks_df)) < 0.15),
            (diabetes_risk > 0.25) & (self.rng.random(len(risks_df)) < 0.08),
            (stroke_risk > 0.2) & (self.rng.random(len(risks_df)) < 0.1),
            (cirrhosis_risk > 0.15) & (self.rng.random(len(risks_df)) < 0.05),
        ]
        choices = ["Cardiovascular disease", "Diabetes complications", "Stroke", "Liver failure"]
        risks_df["primary_death_cause"] = np.select(conditions, choices, default="Other")

        # Estimate event age based on health score (ВОССТАНОВЛЕНО из v1.1.1)
        base_age = 60
        risks_df["estimated_event_age"] = base_age - 20 * (1 - risks_df["health_score"] / 100)
        risks_df["sex"] = genetic_data["sex"].values

        return risks_df

    def _generate_mar_missing_values(
            self,
            df: pd.DataFrame,
            age_col: str = "age_end",
            risk_col: str = "cvd_risk_10year",
    ) -> Tuple[pd.DataFrame, dict]:
        """Generate MAR missing values (2-15% missingness)."""
        df_copy = df.copy()

        # ИЗМЕНЕНО: age threshold 30 → 45 согласно фактическим данным
        if age_col in df_copy.columns and risk_col in df_copy.columns:
            age = df_copy[age_col].values
            cvd_risk = df_copy[risk_col].values
            p_missing_base = 0.05 + 0.08 * (age < 45) - 0.05 * (cvd_risk > 0.2)
            p_missing = np.clip(p_missing_base, 0.02, 0.15)
        else:
            p_missing = np.full(len(df_copy), 0.07)

        cols_with_missing = [
            "final_sbp_mmhg", "final_hdl_mgdl", "final_total_cholesterol_mgdl",
            "final_hba1c_percent", "avg_alcohol_g_per_week", "avg_total_met_minutes",
        ]
        missing_info = {}
        for col in cols_with_missing:
            if col in df_copy.columns:
                mask = self.rng.random(len(df_copy)) < p_missing
                df_copy.loc[mask, col] = np.nan
                missing_info[col] = int(mask.sum())

        return df_copy, missing_info

    def create_aggregated_dataset(
            self,
            cohort_df: pd.DataFrame,
            biomarkers_df: pd.DataFrame,
            risks_df: pd.DataFrame,
            years: int = 20,
            apply_mar: bool = True,
            filter_boundary: bool = True,
            target_n_people: int = 5000,
    ) -> Tuple[pd.DataFrame, np.ndarray, int]:
        """
        Aggregate data into one row per patient with optional boundary filtering.

        Returns:
            tuple: (aggregated_df, filtered_mask, n_original)
        """
        self._validate_positive(years, "years")
        n_original = len(cohort_df)
        filtered_mask = np.zeros(n_original, dtype=bool)

        # Boundary filtering for final-year biomarkers
        if filter_boundary:
            print("\nFiltering boundary artifacts...")
            final_ranges = {
                'final_bmi': (16, 50),
                'final_sbp_mmhg': (80, 200),
                'final_hdl_mgdl': (20, 100),
                'final_total_cholesterol_mgdl': (150, 350),
                'final_hba1c_percent': (3.5, 12.0),
            }
            last_year = years - 1
            boundary_stats = {}

            for col, (low, high) in final_ranges.items():
                if col == 'final_bmi':
                    biomarker_col = f"bmi_{last_year}"
                elif col == 'final_sbp_mmhg':
                    biomarker_col = f"sbp_mmhg_{last_year}"
                elif col == 'final_hdl_mgdl':
                    biomarker_col = f"hdl_mgdl_{last_year}"
                elif col == 'final_total_cholesterol_mgdl':
                    biomarker_col = f"total_cholesterol_mgdl_{last_year}"
                elif col == 'final_hba1c_percent':
                    biomarker_col = f"hba1c_percent_{last_year}"
                else:
                    continue

                if biomarker_col in biomarkers_df.columns:
                    out_of_range = (
                            (biomarkers_df[biomarker_col] < low) |
                            (biomarkers_df[biomarker_col] > high)
                    )
                    filtered_mask |= out_of_range
                    boundary_stats[col] = {
                        'below_low': int((biomarkers_df[biomarker_col] < low).sum()),
                        'above_high': int((biomarkers_df[biomarker_col] > high).sum()),
                        'total': int(out_of_range.sum()),
                        'pct': out_of_range.mean() * 100
                    }
                    print(f"  {col}: {out_of_range.sum()} rows out of range ({out_of_range.mean() * 100:.2f}%)")

            # Check weight separately
            weight_col = f"weight_kg_{last_year}"
            if weight_col in biomarkers_df.columns:
                weight_out_of_range = (
                        (biomarkers_df[weight_col] < 40) |
                        (biomarkers_df[weight_col] > 200)
                )
                filtered_mask |= weight_out_of_range
                print(
                    f"  weight_kg: {weight_out_of_range.sum()} rows out of range ({weight_out_of_range.mean() * 100:.2f}%)")

            n_filtered = filtered_mask.sum()
            filter_pct = n_filtered / n_original * 100
            print(f"\nBoundary filtering: {n_filtered} rows ({filter_pct:.2f}%) marked for removal")

        # Build aggregated dataset
        aggregated = cohort_df.copy()
        lifestyle_cols = [
            "calories_day", "protein_pct", "fat_pct", "carb_pct",
            "saturated_fat_pct", "simple_carbs_pct", "fiber_g_day",
            "sodium_mg_day", "alcohol_g_per_week", "total_met_minutes",
            "cigarettes_per_day", "stress_level", "sleep_hours",
            "muscle_mass_factor", "macro_imbalance",
        ]

        for col in lifestyle_cols:
            year_cols = [f"{col}_{year}" for year in range(years)]
            existing_cols = [c for c in year_cols if c in biomarkers_df.columns]
            if existing_cols:
                aggregated[f"avg_{col}"] = biomarkers_df[existing_cols].mean(axis=1)
                aggregated[f"std_{col}"] = biomarkers_df[existing_cols].std(axis=1)

        for col in ["alcohol_g_per_week", "total_met_minutes", "cigarettes_per_day"]:
            first_col = f"{col}_0"
            last_col = f"{col}_{years - 1}"
            if first_col in biomarkers_df.columns and last_col in biomarkers_df.columns:
                aggregated[f"{col}_start"] = biomarkers_df[first_col]
                aggregated[f"{col}_end"] = biomarkers_df[last_col]
                aggregated[f"{col}_change"] = biomarkers_df[last_col] - biomarkers_df[first_col]
                aggregated[f"{col}_change_pct"] = (
                        (biomarkers_df[last_col] - biomarkers_df[first_col])
                        / (biomarkers_df[first_col] + 1e-10) * 100
                )

        # Final biomarkers
        aggregated["final_bmi"] = biomarkers_df[f"bmi_{years - 1}"]
        aggregated["final_bmi_corrected"] = biomarkers_df[f"bmi_corrected_{years - 1}"]
        aggregated["final_hdl_mgdl"] = biomarkers_df[f"hdl_mgdl_{years - 1}"]
        aggregated["final_total_cholesterol_mgdl"] = biomarkers_df[f"total_cholesterol_mgdl_{years - 1}"]
        aggregated["final_non_hdl_mgdl"] = biomarkers_df[f"non_hdl_mgdl_{years - 1}"]
        aggregated["final_non_hdl_mmol"] = biomarkers_df[f"non_hdl_mmol_{years - 1}"]
        aggregated["final_sbp_mmhg"] = biomarkers_df[f"sbp_mmhg_{years - 1}"]
        aggregated["final_hba1c_percent"] = biomarkers_df[f"hba1c_percent_{years - 1}"]
        aggregated["final_muscle_mass"] = biomarkers_df[f"muscle_mass_factor_{years - 1}"]
        aggregated['cumulative_smoking_end'] = biomarkers_df[f'cumulative_smoking_{years - 1}']

        # Merge with risks
        aggregated = aggregated.merge(risks_df, on="person_id", how="left", suffixes=('', '_risk'))
        aggregated["age_end"] = aggregated["age_start"] + years

        # Apply boundary filtering
        if filter_boundary and filtered_mask.sum() > 0:
            n_before = len(aggregated)
            aggregated = aggregated[~filtered_mask].reset_index(drop=True)
            n_after = len(aggregated)
            print(f"\nAfter boundary filtering: {n_after} patients ({n_after / n_before * 100:.1f}% retained)")

            if n_after < target_n_people * 0.9:
                print(f"Warning: Sample size ({n_after}) below target ({target_n_people})")
                print(f"Recommendation: Generate with n_people={int(target_n_people / (1 - filter_pct / 100))}")

        # Final clipping for aggregated biomarkers
        print("\nApplying final hard clipping to aggregated biomarkers...")
        for col, (low, high) in final_ranges.items():
            if col in aggregated.columns:
                before_clip = ((aggregated[col] < low) | (aggregated[col] > high)).sum()
                aggregated[col] = aggregated[col].clip(low, high)
                after_clip = ((aggregated[col] < low) | (aggregated[col] > high)).sum()
                print(f"  {col}: {before_clip} -> {after_clip} values outside range")

        # Generate MAR missing values
        if apply_mar:
            aggregated, missing_info = self._generate_mar_missing_values(aggregated)
            if missing_info:
                print("\nMAR missing values generated:")
                for col, count in missing_info.items():
                    print(f"   {col}: {count} missing ({count / len(aggregated) * 100:.1f}%)")

        print(f"\nAggregated Dataset Summary")
        print(f"  Original cohort: {n_original} patients")
        print(f"  After boundary filtering: {len(aggregated)} patients")
        print(f"  Features: {aggregated.shape[1]} columns")
        print(f"  Age range: [{aggregated['age_start'].min()}, {aggregated['age_start'].max()}] years")

        # ВОССТАНОВЛЕНО: Возврат tuple (df, mask, n_original)
        return aggregated, filtered_mask, n_original

    def save(
            self,
            df: pd.DataFrame,
            name: str,
            output_dir: Optional[str] = None,
            apply_mar: bool = False,
            generation_params: Optional[dict] = None,
    ) -> str:
        """Save DataFrame as CSV with accompanying metadata JSON."""
        if output_dir is None:
            output_dir = f"data/synthetic_v{__version__}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        df_copy = df.copy()
        if apply_mar and "final_sbp_mmhg" in df_copy.columns and not df_copy["final_sbp_mmhg"].isna().any():
            df_copy, missing_info = self._generate_mar_missing_values(df_copy)
            if missing_info:
                print("Missing values generated (MAR) during save:")
                for col, count in missing_info.items():
                    print(f"   {col}: {count} missing ({count / len(df_copy) * 100:.1f}%)")

        missing_report = df_copy.isnull().sum()
        missing_report = missing_report[missing_report > 0]

        filepath = output_path / f"{name}.csv"
        df_copy.to_csv(filepath, index=False, encoding="utf-8")

        # ВОССТАНОВЛЕНО: Расширенные метаданные
        metadata = {
            "version": __version__,
            "title": f"Comprehensive Health Risk Dataset - {name}",
            "id": f"health-risk-{name.lower().replace('_', '-')}",
            "licenses": [{"name": "CC0-1.0"}],
            "description": (
                f"Synthetic dataset for medical risk prediction. "
                f"Contains {len(df_copy)} records, generated {datetime.now().strftime('%Y-%m-%d')}. "
                f"WARNING: NOT FOR CLINICAL USE."
            ),
            "keywords": ["health", "lifestyle", "risk prediction", "synthetic data", "missing values"],
            "generated_at": datetime.now().isoformat(),
            "citation": "Data generated for coursework on 'Medical Risk Prediction'",
            "generation_parameters": generation_params or {},
            "filtering_info": {
                "boundary_filtering_applied": True,
                "patients_retained": len(df_copy),
                "retention_rate": generation_params.get("retention_rate", "N/A") if generation_params else "N/A",
            } if generation_params else {},
            "missing_values_info": {
                "keep_missing": True,
                "columns_with_missing": list(missing_report.index),
                "missing_percentage": "2-15% (MAR)" if (apply_mar or missing_report.any()) else "0%",
                "missing_counts": missing_report.to_dict(),
            },
        }

        with open(output_path / f"{name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\n{name} saved: {filepath}")
        print(f"Size: {df_copy.shape[0]} rows x {df_copy.shape[1]} columns")
        return str(filepath)


def main():
    """Main entry point for data generation."""
    print("-" * 50)
    print(f"SyntheticHealthSimulator v{__version__}")
    print("MEDICAL DATA GENERATOR LAUNCH")
    print("WARNING: NOT FOR CLINICAL USE - EDUCATIONAL PURPOSES ONLY")
    print("-" * 50)

    generator = DataGenerator(seed=42, diabetes_threshold=16.5)
    years = 20
    target_n_people = 5000

    # Oversampling coefficient to compensate for boundary filtering losses
    boundary_coefficient = 2.41
    n_people = int(target_n_people * boundary_coefficient)

    print(f"\nGeneration parameters:")
    print(f"  Target number of people: {target_n_people}")
    print(f"  Number of people for clipping: {n_people}")
    print(f"  Observation period: {years} years")
    print(f"  Random seed: {generator.seed}")
    print(f"  Diabetes threshold: {generator.diabetes_threshold} (target prevalence ~15%)")
    print(f"  BMR correction: {generator.BMR_MUSCLE_CORRECTION} kcal/kg/day")
    print(f"  Hard clipping: enabled for all biomarkers")

    # 1/5 Generating baseline cohort...
    print(f"\n1/5 Generating baseline cohort...")
    cohort = generator.generate_cohort(n_people)

    # 2/5 Generating lifestyle history for 20 years...
    print(f"\n2/5 Generating lifestyle history for {years} years...")
    lifestyle = generator.generate_lifestyle_history(cohort, years=years)

    # 3/5 Calculating biomarkers with hard clipping...
    print(f"\n3/5 Calculating biomarkers with hard clipping...")
    biomarkers = generator.calculate_biomarkers(cohort, lifestyle, years=years)

    # 4/5 Calculating medical risks...
    print(f"\n4/5 Calculating medical risks...")
    risks = generator.calculate_health_risks(biomarkers, cohort, years=years)

    print(
        f"Class balance check: {{'cvd': {risks['has_cvd'].mean():.4f}, 'diabetes': {risks['has_diabetes'].mean():.4f}, 'stroke': {risks['has_stroke'].mean():.4f}}}")

    # 5/5 Creating aggregated dataset with boundary filtering...
    print(f"\n5/5 Creating aggregated dataset with boundary filtering...")
    aggregated, filtered_mask, n_original = generator.create_aggregated_dataset(
        cohort, biomarkers, risks,
        years=years,
        apply_mar=True,
        filter_boundary=True,
        target_n_people=target_n_people,
    )

    print(f"\nApplying boundary filter to all datasets...")
    n_filtered = filtered_mask.sum()
    n_retained = len(aggregated)

    cohort = cohort[~filtered_mask].reset_index(drop=True)
    lifestyle = lifestyle[~filtered_mask].reset_index(drop=True)
    biomarkers = biomarkers[~filtered_mask].reset_index(drop=True)
    risks = risks[~filtered_mask].reset_index(drop=True)

    if n_retained > target_n_people:
        excess = n_retained - target_n_people
        print(f"\nTrimming {excess} excess patients to match target ({target_n_people})...")
        trim_idx = generator.rng.choice(n_retained, size=target_n_people, replace=False)
        trim_idx = np.sort(trim_idx)  # Сохранить порядок

        cohort = cohort.iloc[trim_idx].reset_index(drop=True)
        lifestyle = lifestyle.iloc[trim_idx].reset_index(drop=True)
        biomarkers = biomarkers.iloc[trim_idx].reset_index(drop=True)
        risks = risks.iloc[trim_idx].reset_index(drop=True)
        aggregated = aggregated.iloc[trim_idx].reset_index(drop=True)

        n_retained = target_n_people
        print(f"  Final dataset: {n_retained} patients (exact match)")

    print(f"  Filtered datasets: {n_filtered} rows removed, {n_retained} rows retained")

    generation_params = {
        "target_n_people": target_n_people,
        "n_people_generated": n_people,
        "n_people_after_filtering": n_retained,
        "boundary_coefficient": boundary_coefficient,
        "filter_loss_rate": f"{n_filtered / n_original * 100:.1f}%",
        "retention_rate": f"{n_retained / n_original * 100:.1f}%",
        "years_observation": years,
        "seed": generator.seed,
    }

    generator.save(cohort, f"01_cohort_baseline_v{__version__}", generation_params=generation_params)
    generator.save(lifestyle, f"02_lifestyle_history_v{__version__}", generation_params=generation_params)
    generator.save(biomarkers, f"03_biomarkers_history_v{__version__}", generation_params=generation_params)
    generator.save(risks, f"04_health_risks_v{__version__}", generation_params=generation_params)
    generator.save(aggregated, f"05_aggregated_dataset_with_missing_v{__version__}", apply_mar=False,
                   generation_params=generation_params)

    # Risk Distribution Statistics
    print("\n" + "-" * 50)
    print("RISK DISTRIBUTION STATISTICS:")
    print("-" * 50)
    risk_stats = {
        "CVD": risks["cvd_risk_10year"].mean(),
        "Diabetes": risks["diabetes_risk_10year"].mean(),
        "Stroke": risks["stroke_risk_10year"].mean(),
        "NAFLD": risks["nafld_risk_10year"].mean(),
    }
    for disease, risk in risk_stats.items():
        print(f"  • {disease}: average risk {risk:.2%}")

    print("\nCLASS BALANCE:")
    class_balance = {
        "CVD": risks["has_cvd"].mean(),
        "Diabetes": risks["has_diabetes"].mean(),
        "Stroke": risks["has_stroke"].mean(),
    }
    for disease, prevalence in class_balance.items():
        print(f"  • {disease}: {prevalence:.2%}")

    print("\n" + "-" * 50)
    print("GENERATION COMPLETED SUCCESSFULLY!")
    print("REMINDER: Data is synthetic, NOT for clinical use")
    print("-" * 50)


if __name__ == "__main__":
    main()
