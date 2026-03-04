# ============================================================================
# WARNING: SyntheticHealthSimulator - NOT FOR CLINICAL USE
# ============================================================================
# This code generates synthetic medical data based on
# mechanistic models. Intended for educational purposes only
# and for training ML models. Do NOT use for clinical diagnosis!

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

__version__ = "1.0.2"
__version_info__ = (1, 0, 2)


class DataGenerator:
    """
    Synthetic Medical Data Generator
    WARNING: NOT FOR CLINICAL USE
    """

    def __init__(
            self,
            seed: int = 42,
            diabetes_threshold: float = 18.7,
            cvd_intercept: float = -3.8,
            stroke_intercept: float = -3.0,
            nafld_intercept: float = -2.5,
            cancer_intercept: float = -1.5,
            cirrhosis_intercept: float = -2.0,
    ):
        """
        Generator initialization
        Args:
            seed: Random seed for reproducibility
            diabetes_threshold: Threshold for diabetes (15-20)
                15 = ~37% prevalence, 17 = ~25%, 18 = ~18%, 18.7 = ~15% (recommended)
            cvd_intercept: Intercept for CVD risk (SCORE2‑like), default -3.8
            stroke_intercept: Intercept for stroke risk, default -3.0
            nafld_intercept: Intercept for NAFLD risk, default -2.5
            cancer_intercept: Intercept for colorectal cancer risk, default -1.5
            cirrhosis_intercept: Intercept for cirrhosis risk, default -2.0
        """
        self.current_year = datetime.now().year
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Calibration parameters (exposed for class balance tuning)
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
        self.BMR_MUSCLE_CORRECTION = 8.5  # kcal/kg/day
        self.TYPICAL_MUSCLE_RATIO = 0.3  # typical muscle mass = 30% of weight

        self.MUSCLE_PARAMS = {
            "strength_to_training": 150,  # MET‑min per strength session
            "protein_requirement": 1.6,  # g/kg for positive balance
            "protein_deficit": 0.8,  # g/kg for negative balance
            "sleep_deficit_hours": 6,
            "max_muscle": 20,
            "min_muscle": 0,
            "mm_to_weight_kg": 0.5,  # 1 index unit = 0.5 kg
        }

        # Biomarker validity ranges (reference section 8.3)
        self.BIOMARKER_RANGES = {
            "hdl": (20, 100),
            "tc": (150, 350),
            "sbp": (80, 200),
            "hba1c": (3.5, 12.0),
            "weight": (40, 200),
            "bmi": (16, 50),
            "non_hdl_mgdl": (50, 232),  # 232 = 6.0 mmol/L
            "non_hdl_mmol": (2.0, 6.0),
            "muscle_mass_factor": (0, 20),
        }

        # Validate inputs
        if not (15 <= diabetes_threshold <= 25):
            warnings.warn(f"diabetes_threshold={diabetes_threshold} outside typical range [15,25]")
        if n_people := getattr(self, '_n_people', None):
            self._validate_positive(n_people, "n_people")
        # years will be validated in methods

    @staticmethod
    def _validate_positive(value: Union[int, float], name: str) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    @staticmethod
    def _to_array(val: Union[float, int, np.ndarray]) -> np.ndarray:
        return np.asarray(val)

    def _clip_biomarker(self, values: np.ndarray, marker_type: str) -> np.ndarray:
        """Hard clipping for biomarkers according to reference section 8.3"""
        if marker_type in self.BIOMARKER_RANGES:
            low, high = self.BIOMARKER_RANGES[marker_type]
            return np.clip(values, low, high)
        return values

    # Truncated normal helpers (reduce duplication)
    def _generate_truncnorm(
            self,
            loc: Union[float, np.ndarray],
            scale: float,
            low: float,
            high: float,
            size: int,
    ) -> np.ndarray:
        """
        Generate values from a truncated normal distribution.
        Parameters:
            loc : float or array_like – mean of the underlying normal
            scale : float – standard deviation
            low : float – lower bound
            high : float – upper bound
            size : int – number of samples
        Returns:
            np.ndarray of size samples
        """
        a = (low - loc) / scale
        b = (high - loc) / scale
        return truncnorm.rvs(a, b, loc=loc, scale=scale, size=size, random_state=self.rng)

    def _generate_hdl(self, n_m: int, n_f: int) -> np.ndarray:
        """Generate HDL cholesterol values for males and females."""
        hdl_m = self._generate_truncnorm(loc=50, scale=10, low=20, high=100, size=n_m)
        hdl_f = self._generate_truncnorm(loc=60, scale=12, low=20, high=100, size=n_f)
        return np.concatenate([hdl_m, hdl_f])

    def _generate_muscle_mass(self, sexes: np.ndarray) -> np.ndarray:
        """Generate baseline muscle mass factor using sex‑specific truncated normals."""
        n_people = len(sexes)
        n_m = np.sum(sexes == "M")
        n_f = n_people - n_m
        muscle_m = self._generate_truncnorm(loc=12, scale=3, low=0, high=20, size=n_m)
        muscle_f = self._generate_truncnorm(loc=8, scale=2.5, low=0, high=20, size=n_f)
        muscle = np.concatenate([muscle_m, muscle_f])
        # reshuffle to match the original random order (sexes already shuffled later)
        return muscle

    # Risk calculation methods (with tunable intercepts)
    def _calculate_cvd_risk(
            self,
            age,
            sbp,
            non_hdl_mmol,
            pack_years,
            hba1c,
            genetic_cvd,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CVD risk calculation (SCORE2-like) with non-HDL and pack-years"""
        # Target prevalence: 15-20% for 20-year cohort, adjust intercept if needed
        age = self._to_array(age)
        sbp = self._to_array(sbp)
        non_hdl_mmol = self._to_array(non_hdl_mmol)
        pack_years = self._to_array(pack_years)
        hba1c = self._to_array(hba1c)
        genetic_cvd = self._to_array(genetic_cvd)

        logit = (
                self.cvd_intercept
                + (age - 40) * 0.05
                + (sbp - 120) * 0.03
                + (non_hdl_mmol - 3.5) * 0.3
                + pack_years * 0.08  # INTENTIONALLY REDUCED
                + (hba1c > 6.5).astype(float) * 0.9
                + genetic_cvd * 0.4
        )
        risk = 1 / (1 + np.exp(-logit))
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
        """Diabetes risk calculation (FINDRISC-like) with additional factors
        Uses corrected BMI (bmi_corrected) – protective effect of muscle mass."""
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
        points += (bmi_corrected >= 25).astype(float) * 1 + (bmi_corrected >= 30).astype(float) * 2

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

        # INTENTIONALLY CALIBRATED: Diabetes threshold 18.7 for additional risk factors in extended model.
        risk = 1 / (1 + np.exp(-0.4 * (points - self.diabetes_threshold)))
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
        age = self._to_array(age)
        sbp = self._to_array(sbp)
        hba1c = self._to_array(hba1c)
        smoking_binary = (self._to_array(smoking) > 0).astype(float)
        bmi_corrected = self._to_array(bmi_corrected)
        genetic_stroke = self._to_array(genetic_stroke)

        logit = (
                (age - 50) * 0.03
                + (sbp - 120) * 0.015
                + (hba1c - 5.5) * 0.3
                + smoking_binary * 0.15
                + (bmi_corrected - 25) * 0.03
                + genetic_stroke * 0.4
                + self.stroke_intercept
        )
        risk = 1 / (1 + np.exp(-logit))
        return risk, logit

    def _calculate_nafld_risk(
            self,
            bmi_corrected,
            hba1c,
            saturated_fat,
            fiber,
            genetic_nafld,
    ) -> Tuple[np.ndarray, np.ndarray]:
        bmi_corrected = self._to_array(bmi_corrected)
        hba1c = self._to_array(hba1c)
        saturated_fat = self._to_array(saturated_fat)
        fiber = self._to_array(fiber)
        genetic_nafld = self._to_array(genetic_nafld)

        logit = (
                (bmi_corrected - 25) * 0.04
                + (hba1c - 5.5) * 0.25
                + (saturated_fat - 0.3) * 0.4
                + (fiber < 25).astype(float) * 0.3
                + genetic_nafld * 0.4
                + self.nafld_intercept
        )
        risk = 1 / (1 + np.exp(-logit))
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
        age = self._to_array(age)
        fiber = self._to_array(fiber)
        alcohol = self._to_array(alcohol)
        smoking_binary = (self._to_array(smoking) > 0).astype(float)
        bmi_corrected = self._to_array(bmi_corrected)
        genetic_cancer = self._to_array(genetic_cancer)

        logit = (
                (age - 50) * 0.04
                + (fiber < 20).astype(float) * 0.6
                + (alcohol > 140).astype(float) * 0.4
                + smoking_binary * 0.5
                + (bmi_corrected > 30).astype(float) * 0.3
                + genetic_cancer * 0.7
                + self.cancer_intercept
        )
        risk = 1 / (1 + np.exp(-logit))
        return risk, logit

    def _calculate_cirrhosis_risk(
            self,
            alcohol,
            smoking,
            bmi_corrected,
            genetic_cirrhosis,
    ) -> Tuple[np.ndarray, np.ndarray]:
        alcohol = self._to_array(alcohol)
        smoking_binary = (self._to_array(smoking) > 0).astype(float)
        bmi_corrected = self._to_array(bmi_corrected)
        genetic_cirrhosis = self._to_array(genetic_cirrhosis)

        logit = (
                np.log1p(alcohol) * 0.4
                + smoking_binary * 0.2
                + (bmi_corrected > 30).astype(float) * 0.3
                + genetic_cirrhosis * 0.6
                + self.cirrhosis_intercept
        )
        risk = 1 / (1 + np.exp(-logit))
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

    def calculate_health_risks(
            self,
            biomarkers_df: pd.DataFrame,
            cohort_df: pd.DataFrame,
            years: int = 20,
    ) -> pd.DataFrame:
        """Medical risk calculation for the cohort"""
        self._validate_positive(years, "years")
        risks_df = pd.DataFrame({"person_id": biomarkers_df["person_id"]})
        last_year = years - 1

        # Extract final‑year data
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

        cvd_risk, _ = self._calculate_cvd_risk(
            final_age,
            final_sbp,
            final_non_hdl_mmol,
            cumulative_smoking,
            final_hba1c,
            genetic_data["genetic_risk_cvd"].values,
        )

        diabetes_risk, _ = self._calculate_diabetes_risk(
            final_age,
            final_bmi_corrected,
            final_hba1c,
            cumulative_smoking,
            final_fiber,
            genetic_data["genetic_risk_diabetes"].values,
            final_macro_imbalance,
            final_muscle,
            final_sleep,
        )

        stroke_risk, _ = self._calculate_stroke_risk(
            final_age,
            final_sbp,
            final_hba1c,
            cumulative_smoking,
            final_bmi_corrected,
            genetic_data["genetic_risk_stroke"].values,
        )

        nafld_risk, _ = self._calculate_nafld_risk(
            final_bmi_corrected,
            final_hba1c,
            final_saturated_fat,
            final_fiber,
            genetic_data["genetic_risk_nafld"].values,
        )

        cancer_risk, _ = self._calculate_cancer_risk(
            final_age,
            final_fiber,
            final_alcohol,
            cumulative_smoking,
            final_bmi_corrected,
            genetic_data["genetic_risk_cancer"].values,
        )

        cirrhosis_risk, _ = self._calculate_cirrhosis_risk(
            final_alcohol,
            cumulative_smoking,
            final_bmi_corrected,
            genetic_data["genetic_risk_cirrhosis"].values,
        )

        health_score = self._calculate_health_score(
            cvd_risk,
            diabetes_risk,
            stroke_risk,
            nafld_risk,
            cancer_risk,
            cirrhosis_risk,
        )

        # Clip risks and generate binary outcomes
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

        # Class balance warnings
        class_balance_check = {
            "cvd": risks_df["has_cvd"].mean(),
            "diabetes": risks_df["has_diabetes"].mean(),
            "stroke": risks_df["has_stroke"].mean(),
        }

        print(f"Class balance check: {class_balance_check}")
        for disease, prevalence in class_balance_check.items():
            if prevalence < 0.10 or prevalence > 0.50:
                warnings.warn(
                    f"Class imbalance detected for {disease}: {prevalence:.2%}. "
                    f"Consider adjusting risk thresholds."
                )

        # Primary death cause
        conditions = [
            (cvd_risk > 0.3) & (self.rng.random(len(risks_df)) < 0.15),
            (diabetes_risk > 0.25) & (self.rng.random(len(risks_df)) < 0.08),
            (stroke_risk > 0.2) & (self.rng.random(len(risks_df)) < 0.1),
            (cirrhosis_risk > 0.15) & (self.rng.random(len(risks_df)) < 0.05),
        ]

        choices = [
            "Cardiovascular disease",
            "Diabetes complications",
            "Stroke",
            "Liver failure",
        ]

        risks_df["primary_death_cause"] = np.select(conditions, choices, default="Other")

        base_age = 60
        risks_df["estimated_event_age"] = base_age - 20 * (1 - risks_df["health_score"] / 100)
        risks_df["sex"] = genetic_data["sex"].values
        return risks_df

    # BMR / TDEE helpers
    @staticmethod
    def calculate_metabolic_rate(
            weight_kg: float, height_cm: float, age_years: int, sex: str
    ) -> float:
        """Mifflin‑St Jeor formula without muscle mass correction."""
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
        """BMR with muscle mass correction."""
        bmr_base = self.calculate_metabolic_rate(weight_kg, height_cm, age_years, sex)
        typical_muscle_mass = self.TYPICAL_MUSCLE_RATIO * weight_kg
        correction = self.BMR_MUSCLE_CORRECTION * (muscle_mass_kg - typical_muscle_mass)
        return bmr_base + correction

    @staticmethod
    def calculate_tdee(bmr: float, met_minutes_week: float) -> float:
        """Total Daily Energy Expenditure (WHO classification)."""
        if met_minutes_week < 450:
            activity_factor = 1.2
        elif met_minutes_week < 900:
            activity_factor = 1.375
        elif met_minutes_week < 1500:
            activity_factor = 1.55
        else:
            activity_factor = 1.725
        return bmr * activity_factor

    # Genetic risk generation
    def _generate_genetic_risk(self, n: int, disease: str, rng) -> np.ndarray:
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

    # Cohort generation
    def generate_cohort(self, n_people: int = 5000) -> pd.DataFrame:
        """Generate baseline cohort."""
        self._validate_positive(n_people, "n_people")

        ages = self._generate_truncnorm(loc=35, scale=8, low=20, high=50, size=n_people).astype(int)
        sexes = self.rng.choice(["M", "F"], n_people, p=[0.48, 0.52])

        genetic_diabetes = self._generate_genetic_risk(n_people, "diabetes", self.rng)
        genetic_cvd = self._generate_genetic_risk(n_people, "cvd", self.rng)
        genetic_cancer = self._generate_genetic_risk(n_people, "cancer_colorectal", self.rng)
        genetic_stroke = self._generate_genetic_risk(n_people, "stroke", self.rng)
        genetic_nafld = self._generate_genetic_risk(n_people, "nafld", self.rng)
        genetic_cirrhosis = self._generate_genetic_risk(n_people, "cirrhosis", self.rng)

        # Generate height and weight based on sex (FIXED)
        height_cm = np.zeros(n_people)
        weight_kg = np.zeros(n_people)

        # Male
        male_mask = sexes == "M"
        height_cm[male_mask] = self.rng.normal(175, 7, np.sum(male_mask))
        weight_kg[male_mask] = self.rng.normal(80, 12, np.sum(male_mask))

        # Female
        female_mask = sexes == "F"
        height_cm[female_mask] = self.rng.normal(162, 6, np.sum(female_mask))
        weight_kg[female_mask] = self.rng.normal(65, 10, np.sum(female_mask))

        # No need to shuffle because arrays are already aligned with sexes
        # However, we should still shuffle all arrays together to break any ordering patterns
        # But shuffling is optional because data is already independent.
        # To keep consistency with previous versions, we can shuffle but ensure all arrays are shuffled identically.
        shuffle_idx = self.rng.permutation(n_people)
        ages = ages[shuffle_idx]
        sexes = sexes[shuffle_idx]
        height_cm = height_cm[shuffle_idx]
        weight_kg = weight_kg[shuffle_idx]
        genetic_diabetes = genetic_diabetes[shuffle_idx]
        genetic_cvd = genetic_cvd[shuffle_idx]
        genetic_cancer = genetic_cancer[shuffle_idx]
        genetic_stroke = genetic_stroke[shuffle_idx]
        genetic_nafld = genetic_nafld[shuffle_idx]
        genetic_cirrhosis = genetic_cirrhosis[shuffle_idx]

        bmi_baseline = weight_kg / ((height_cm / 100) ** 2)
        bmi_baseline = self._clip_biomarker(bmi_baseline, "bmi")

        hdl_baseline = self._generate_hdl(np.sum(male_mask), np.sum(female_mask))[shuffle_idx]
        tc_baseline = self._generate_truncnorm(loc=200, scale=25, low=150, high=350, size=n_people)

        # NOTE: Genetic risk affects ONLY baseline TC, not annual change (section 2.4)
        tc_baseline = tc_baseline + 0.8 * genetic_cvd
        tc_baseline = self._clip_biomarker(tc_baseline, "tc")  # Ensure within [150,350]

        hba1c_baseline = self._generate_truncnorm(loc=5.4, scale=0.3, low=4.0, high=6.5, size=n_people)
        hba1c_baseline = self._clip_biomarker(hba1c_baseline, "hba1c")

        mu_sbp = 120 + 0.5 * (ages - 35)
        sbp_baseline = self._generate_truncnorm(loc=mu_sbp, scale=8, low=90, high=160, size=n_people)
        sbp_baseline = self._clip_biomarker(sbp_baseline, "sbp")

        muscle_baseline = self._generate_muscle_mass(sexes)

        # BMR calculation with muscle mass correction (reference section 1.1)
        bmr_baseline_corrected = np.array(
            [
                self.calculate_bmr_corrected(
                    weight_kg[i],
                    height_cm[i],
                    ages[i],
                    sexes[i],
                    muscle_baseline[i] * self.MUSCLE_PARAMS["mm_to_weight_kg"],
                )
                for i in range(n_people)
            ]
        )

        data = {
            "person_id": range(1, n_people + 1),
            "age_start": ages,
            "sex": sexes,
            "height_cm": height_cm,
            "weight_start_kg": weight_kg,
            "bmi_start": bmi_baseline,
            "muscle_mass_start": muscle_baseline,
            "bmr_start_kcal": bmr_baseline_corrected,
            "base_calories": np.where(sexes == "M", 2500, 2000),
            "genetic_risk_diabetes": genetic_diabetes,
            "genetic_risk_cvd": genetic_cvd,
            "genetic_risk_cancer": genetic_cancer,
            "genetic_risk_stroke": genetic_stroke,
            "genetic_risk_nafld": genetic_nafld,
            "genetic_risk_cirrhosis": genetic_cirrhosis,
            "initial_hdl_mgdl": hdl_baseline,
            "initial_tc_mgdl": tc_baseline,
            "initial_hba1c_percent": hba1c_baseline,
            "initial_sbp_mmhg": sbp_baseline,
        }
        return pd.DataFrame(data)

    # Stress event generator
    def _generate_stress_event(self, years: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Random stress event generation.
        NOTE: Duration extended to 1 year with decay (section 4.2).
        """
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

    # Lifestyle history (OU processes)
    def generate_lifestyle_history(
            self, cohort_df: pd.DataFrame, years: int = 20
    ) -> pd.DataFrame:
        """Lifestyle history with OU processes, weight and muscle evolution."""
        self._validate_positive(years, "years")
        n_people = len(cohort_df)

        # Extract initial parameters
        age_start = cohort_df["age_start"].values
        sex = cohort_df["sex"].values
        height = cohort_df["height_cm"].values
        weight = cohort_df["weight_start_kg"].values.copy()
        muscle = cohort_df["muscle_mass_start"].values.copy()

        # Baseline levels for OU processes
        alc_baseline = np.where(self.rng.random(n_people) > 0.3, self.rng.gamma(2, 20, n_people), 0)
        cardio_baseline = self.rng.uniform(300, 1200, n_people)
        strength_baseline = self.rng.uniform(0, 600, n_people)
        smoke_baseline = np.where(self.rng.random(n_people) < 0.25, self.rng.exponential(3, n_people), 0)
        stress_baseline = self.rng.normal(3, 2, n_people)
        sleep_baseline = self.rng.normal(7, 1, n_people)

        # Pre‑allocate series (n_people × years)
        alcohol_series = np.zeros((n_people, years))
        cardio_series = np.zeros((n_people, years))
        strength_series = np.zeros((n_people, years))
        smoke_series = np.zeros((n_people, years))
        stress_series = np.zeros((n_people, years))
        sleep_series = np.zeros((n_people, years))

        # Fill year 0
        alcohol_series[:, 0] = alc_baseline
        cardio_series[:, 0] = cardio_baseline
        strength_series[:, 0] = strength_baseline
        smoke_series[:, 0] = smoke_baseline
        stress_series[:, 0] = stress_baseline
        sleep_series[:, 0] = sleep_baseline

        # OU parameters (common for all factors)
        theta = {
            "alcohol": 0.8,
            "cardio": 0.2,
            "strength": 0.25,
            "smoke": 0.3,
            "stress": 0.2,
            "sleep": 0.2,
        }

        sigma = {
            "alcohol": 20.0,
            "cardio": 0.2,
            "strength": 0.25,
            "smoke": 0.3,
            "stress": 1.0,
            "sleep": 0.8,
        }

        min_vals = {
            "alcohol": 0,
            "cardio": 0,
            "strength": 0,
            "smoke": 0,
            "stress": 0,
            "sleep": 4,
        }

        max_vals = {
            "alcohol": 500,
            "cardio": 3000,
            "strength": 1500,
            "smoke": 40,
            "stress": 10,
            "sleep": 10,
        }

        baselines = {
            "alcohol": alc_baseline,
            "cardio": cardio_baseline,
            "strength": strength_baseline,
            "smoke": smoke_baseline,
            "stress": stress_baseline,
            "sleep": sleep_baseline,
        }

        series = {
            "alcohol": alcohol_series,
            "cardio": cardio_series,
            "strength": strength_series,
            "smoke": smoke_series,
            "stress": stress_series,
            "sleep": sleep_series,
        }

        # Loop over years
        for t in range(1, years):
            for key in theta.keys():
                series[key][:, t] = (
                        series[key][:, t - 1]
                        + theta[key] * (baselines[key] - series[key][:, t - 1])
                        + sigma[key] * self.rng.normal(size=n_people)
                )
                series[key][:, t] = np.clip(series[key][:, t], min_vals[key], max_vals[key])

        # Apply stress events (common for all)
        stress_events, sleep_events, alcohol_events = self._generate_stress_event(years)
        for t in range(years):
            stress_series[:, t] += stress_events[t]
            sleep_series[:, t] -= sleep_events[t]
            alcohol_series[:, t] *= 1 + alcohol_events[t]
        stress_series = np.clip(stress_series, 0, 10)
        sleep_series = np.clip(sleep_series, 4, 10)

        # Arrays for true weight, muscle, BMR, TDEE, cumulative smoking
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

        # Year 0
        weight_true[:, 0] = weight
        muscle_true[:, 0] = muscle
        for i in range(n_people):
            muscle_kg = muscle_true[i, 0] * self.MUSCLE_PARAMS["mm_to_weight_kg"]
            bmr_true[i, 0] = self.calculate_bmr_corrected(weight_true[i, 0], height[i], age_start[i], sex[i], muscle_kg)
            tdee_true[i, 0] = self.calculate_tdee(bmr_true[i, 0], cardio_series[i, 0] + strength_series[i, 0])
        cumulative_smoking[:, 0] = smoke_series[:, 0] / 20

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

        # Loop years 1..years-1
        for t in range(1, years):
            age = age_start + t
            prev_weight = weight_true[:, t - 1]
            prev_muscle = muscle_true[:, t - 1]

            # BMR and TDEE based on previous weight
            bmr_prev_corrected = np.array(
                [
                    self.calculate_bmr_corrected(
                        prev_weight[i],
                        height[i],
                        age[i],
                        sex[i],
                        prev_muscle[i] * self.MUSCLE_PARAMS["mm_to_weight_kg"],
                    )
                    for i in range(n_people)
                ]
            )

            tdee_prev = np.array(
                [
                    self.calculate_tdee(
                        bmr_prev_corrected[i],
                        cardio_series[i, t] + strength_series[i, t],
                    )
                    for i in range(n_people)
                ]
            )

            # Calorie intake
            calorie_offset = self.rng.normal(0, 200, n_people)
            calories = tdee_prev + calorie_offset
            calories = np.maximum(calories, 1200)

            # Calories from alcohol
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

            # Protein in g/kg
            protein_grams = (protein_pct / 100) * non_alcohol_cal / 4
            protein_g_per_kg = protein_grams / prev_weight

            # NOTE: Protein balance categories (section 1.3.2):
            # >1.6 g/kg = positive (+1), 0.8-1.6 g/kg = neutral (0), <0.8 g/kg = negative (-1)
            protein_balance = np.zeros(n_people)
            protein_balance[protein_g_per_kg > self.MUSCLE_PARAMS["protein_requirement"]] = 1
            protein_balance[protein_g_per_kg < self.MUSCLE_PARAMS["protein_deficit"]] = -1

            strength_sessions = strength_series[:, t] / self.MUSCLE_PARAMS["strength_to_training"]
            strength_sessions = np.clip(strength_sessions, 0, 5)

            # ProteinFactor
            protein_factor = 0.1 + 0.05 * strength_sessions + 0.05 * protein_balance
            protein_factor = np.clip(protein_factor, 0, 0.4)

            # Body composition change calculation
            delta_weight_total = (calories - tdee_prev) * 365 / 7700

            # Split into fat and muscle
            delta_muscle_weight = delta_weight_total * protein_factor
            delta_fat_weight = delta_weight_total * (1 - protein_factor)

            # Update muscle mass (1 unit of index = 0.5 kg)
            muscle_kg = prev_muscle * self.MUSCLE_PARAMS["mm_to_weight_kg"]
            muscle_kg_new = muscle_kg + delta_muscle_weight
            muscle_true[:, t] = self._clip_biomarker(
                muscle_kg_new / self.MUSCLE_PARAMS["mm_to_weight_kg"],
                "muscle_mass_factor",
            )

            # Total weight = fat mass + muscle mass + other (bones, organs)
            fat_weight = prev_weight - muscle_kg
            fat_weight_new = fat_weight + delta_fat_weight
            weight_new = fat_weight_new + muscle_kg_new
            weight_true[:, t] = self._clip_biomarker(weight_new, "weight")

            # BMR and TDEE for new weight
            bmr_true[:, t] = np.array(
                [
                    self.calculate_bmr_corrected(
                        weight_true[i, t],
                        height[i],
                        age[i],
                        sex[i],
                        muscle_true[i, t] * self.MUSCLE_PARAMS["mm_to_weight_kg"],
                    )
                    for i in range(n_people)
                ]
            )

            tdee_true[:, t] = np.array(
                [
                    self.calculate_tdee(bmr_true[i, t], cardio_series[i, t] + strength_series[i, t])
                    for i in range(n_people)
                ]
            )

            cumulative_smoking[:, t] = cumulative_smoking[:, t - 1] + smoke_series[:, t] / 20

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
            data[f"cigarettes_per_day_{year}"] = smoke_series[:, year]
            data[f"stress_level_{year}"] = stress_series[:, year]
            data[f"sleep_hours_{year}"] = sleep_series[:, year]
            data[f"cumulative_smoking_{year}"] = cumulative_smoking[:, year]

            # Nutrition parameters
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

    # Biomarker calculation
    def calculate_biomarkers(
            self, cohort_df: pd.DataFrame, lifestyle_df: pd.DataFrame, years: int = 20
    ) -> pd.DataFrame:
        """Vectorised biomarker calculation from lifestyle."""
        self._validate_positive(years, "years")
        n_people = len(lifestyle_df)
        person_ids = lifestyle_df["person_id"].values
        cohort_by_id = cohort_df.set_index("person_id")
        height = cohort_by_id.loc[person_ids]["height_cm"].values

        # Initial biomarker values
        initial_hdl = cohort_by_id.loc[person_ids]["initial_hdl_mgdl"].values
        initial_tc = cohort_by_id.loc[person_ids]["initial_tc_mgdl"].values
        initial_hba1c = cohort_by_id.loc[person_ids]["initial_hba1c_percent"].values
        initial_sbp = cohort_by_id.loc[person_ids]["initial_sbp_mmhg"].values

        # Extract true weight and muscle (n_people × years)
        weight_true = np.column_stack([lifestyle_df[f"weight_kg_true_{y}"].values for y in range(years)])
        muscle_true = np.column_stack([lifestyle_df[f"muscle_mass_factor_true_{y}"].values for y in range(years)])

        # Extract lifestyle factors (n_people × years)
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

        # Year 0
        hdl_true[:, 0] = self._clip_biomarker(initial_hdl, "hdl")
        tc_true[:, 0] = self._clip_biomarker(initial_tc, "tc")
        sbp_true[:, 0] = self._clip_biomarker(initial_sbp, "sbp")
        hba1c_true[:, 0] = self._clip_biomarker(initial_hba1c, "hba1c")

        # Pre‑compute constants that do not depend on t (except height)
        height_m = height / 100.0
        height_sq = height_m ** 2

        # Main loop years 1..years-1
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

            # HDL
            age_curr = age_vals[:, t]
            hdl_change = (
                    0.1 * (cardio_met[:, t] / 1000.0)
                    + 0.15 * (strength_met[:, t] / 450.0)
                    - 0.05 * (delta_sat_fat * 100.0)
                    + 0.1 * delta_fiber
                    + 0.02 * muscle_true[:, t]
                    - 0.06 * (bmi_curr - 25)
                    - 0.03 * (age_curr - 35)
            )

            hdl_new = hdl_true[:, t - 1] + hdl_change
            hdl_true[:, t] = self._clip_biomarker(hdl_new, "hdl")

            # Total cholesterol
            tc_change = (
                    1.5 * (delta_sat_fat * 100.0)
                    - 0.5 * delta_fiber
                    - 0.3 * (total_met / 1000.0)
            )
            tc_new = tc_true[:, t - 1] + tc_change
            tc_true[:, t] = self._clip_biomarker(tc_new, "tc")

            # SBP
            alcohol_effect = np.where(alcohol_day <= 10, 0, (alcohol_day - 10) * 0.03)
            sleep_effect = np.maximum(0, 6 - sleep_h) * 0.5
            sbp_change = (
                    delta_bmi * 0.7
                    + alcohol_effect
                    + (delta_sodium / 100.0) * 0.25
                    + delta_stress * 0.5
                    + sleep_effect
            )
            sbp_new = sbp_true[:, t - 1] + sbp_change
            sbp_true[:, t] = self._clip_biomarker(sbp_new, "sbp")

            # HbA1c
            macro_effect = macro_imb[:, t] * 0.002
            muscle_effect = -0.005 * muscle_true[:, t]
            sleep_deficit_effect = (sleep_h < 6).astype(float) * 0.01
            hba1c_change = (
                    (delta_simple_carbs / 10.0) * 0.15
                    + delta_bmi * 0.03
                    + delta_fiber * (-0.002)
                    + (total_met / 1000.0) * (-0.01)
                    + macro_effect
                    + muscle_effect
                    + sleep_deficit_effect
            )
            hba1c_new = hba1c_true[:, t - 1] + hba1c_change
            hba1c_true[:, t] = self._clip_biomarker(hba1c_new, "hba1c")

        # Build output dictionary with noisy measurements
        data_dict = {"person_id": person_ids}

        # Noise parameters
        noise = self.MEASUREMENT_NOISE

        for y in range(years):
            # Weight (noise added)
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

            # BMI and corrected BMI
            bmi = data_dict[f"weight_kg_{y}"] / height_sq
            data_dict[f"bmi_{y}"] = self._clip_biomarker(bmi, "bmi")
            bmi_corr = bmi * (1 - 0.01 * muscle_true[:, y])
            data_dict[f"bmi_corrected_{y}"] = self._clip_biomarker(bmi_corr, "bmi")

            # Non‑HDL
            non_hdl_mgdl = data_dict[f"total_cholesterol_mgdl_{y}"] - data_dict[f"hdl_mgdl_{y}"]
            non_hdl_mgdl_clipped = self._clip_biomarker(non_hdl_mgdl, "non_hdl_mgdl")
            data_dict[f"non_hdl_mgdl_{y}"] = non_hdl_mgdl_clipped
            data_dict[f"non_hdl_mmol_{y}"] = self._clip_biomarker(non_hdl_mgdl_clipped / 38.67, "non_hdl_mmol")

            # Muscle mass factor (true value, no noise)
            data_dict[f"muscle_mass_factor_{y}"] = muscle_true[:, y]

        # Copy columns from lifestyle_df that are needed for risk calculation
        # but not yet present (e.g., cumulative_smoking, alcohol, fiber, etc.)
        cols_to_copy = [
            "age",
            "cumulative_smoking",
            "alcohol_g_per_week",
            "fiber_g_day",
            "saturated_fat_pct",
            "macro_imbalance",
            "sleep_hours",
            "total_met_minutes",
        ]
        for col in cols_to_copy:
            for y in range(years):
                data_dict[f"{col}_{y}"] = lifestyle_df[f"{col}_{y}"].values

        # Convert dictionary to DataFrame
        biomarkers_df = pd.DataFrame(data_dict)
        return biomarkers_df

    # MAR missing values
    def _generate_mar_missing_values(
            self,
            df: pd.DataFrame,
            age_col: str = "age_end",
            risk_col: str = "cvd_risk_10year",
    ) -> Tuple[pd.DataFrame, dict]:
        """Generate MAR missing values (2‑15% missingness)."""
        df_copy = df.copy()
        if age_col in df_copy.columns and risk_col in df_copy.columns:
            age = df_copy[age_col].values
            cvd_risk = df_copy[risk_col].values
            p_missing_base = 0.05 + 0.1 * (age < 30) - 0.05 * (cvd_risk > 0.2)
            p_missing = np.clip(p_missing_base, 0.02, 0.15)
        else:
            p_missing = np.full(len(df_copy), 0.07)

        cols_with_missing = [
            "final_sbp_mmhg",
            "final_hdl_mgdl",
            "final_total_cholesterol_mgdl",
            "final_hba1c_percent",
            "avg_alcohol_g_per_week",
            "avg_total_met_minutes",
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
    ) -> pd.DataFrame:
        """Aggregate data into one row per patient, optionally add MAR missing values."""
        self._validate_positive(years, "years")
        aggregated = cohort_df.copy()
        lifestyle_cols = [
            "calories_day",
            "protein_pct",
            "fat_pct",
            "carb_pct",
            "saturated_fat_pct",
            "simple_carbs_pct",
            "fiber_g_day",
            "sodium_mg_day",
            "alcohol_g_per_week",
            "total_met_minutes",
            "cigarettes_per_day",
            "stress_level",
            "sleep_hours",
            "muscle_mass_factor",
            "macro_imbalance",
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
                        / (biomarkers_df[first_col] + 1e-10)
                        * 100
                )

        aggregated["final_bmi"] = biomarkers_df[f"bmi_{years - 1}"]
        aggregated["final_bmi_corrected"] = biomarkers_df[f"bmi_corrected_{years - 1}"]
        aggregated["final_hdl_mgdl"] = biomarkers_df[f"hdl_mgdl_{years - 1}"]
        aggregated["final_total_cholesterol_mgdl"] = biomarkers_df[f"total_cholesterol_mgdl_{years - 1}"]
        aggregated["final_non_hdl_mgdl"] = biomarkers_df[f"non_hdl_mgdl_{years - 1}"]
        aggregated["final_non_hdl_mmol"] = biomarkers_df[f"non_hdl_mmol_{years - 1}"]
        aggregated["final_sbp_mmhg"] = biomarkers_df[f"sbp_mmhg_{years - 1}"]
        aggregated["final_hba1c_percent"] = biomarkers_df[f"hba1c_percent_{years - 1}"]
        aggregated["final_muscle_mass"] = biomarkers_df[f"muscle_mass_factor_{years - 1}"]
        aggregated = aggregated.merge(risks_df, on="person_id", how="left", suffixes=('', '_risk'))
        aggregated["age_end"] = aggregated["age_start"] + years

        if apply_mar:
            aggregated, missing_info = self._generate_mar_missing_values(aggregated)
            if missing_info:
                print("MAR missing values generated:")
                for col, count in missing_info.items():
                    print(f"   • {col}: {count} missing ({count / len(aggregated) * 100:.1f}%)")
        return aggregated

    # Save to CSV with metadata
    def save(
            self,
            df: pd.DataFrame,
            name: str,
            output_dir: Optional[str] = None,
            apply_mar: bool = False,
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
                    print(f"   • {col}: {count} missing ({count / len(df_copy) * 100:.1f}%)")

        missing_report = df_copy.isnull().sum()
        missing_report = missing_report[missing_report > 0]
        filepath = output_path / f"{name}.csv"
        df_copy.to_csv(filepath, index=False, encoding="utf-8")

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
            "keywords": [
                "health",
                "lifestyle",
                "risk prediction",
                "synthetic data",
                "missing values",
            ],
            "generated_at": datetime.now().isoformat(),
            "citation": "Data generated for coursework on 'Medical Risk Prediction'",
            "missing_values_info": {
                "keep_missing": True,
                "columns_with_missing": list(missing_report.index),
                "missing_percentage": "2-15% (MAR)" if (apply_mar or missing_report.any()) else "0%",
                "missing_counts": missing_report.to_dict(),
            },
        }
        with open(output_path / f"{name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"{name} saved: {filepath}")
        print(f"Size: {df_copy.shape[0]} rows × {df_copy.shape[1]} columns")
        return str(filepath)


def main():
    # ============================================================================
    # WARNING: SyntheticHealthSimulator - NOT FOR CLINICAL USE (section 7.4.3)
    # ============================================================================
    print("=" * 70)
    print(f"SyntheticHealthSimulator v{__version__}")
    print("MEDICAL DATA GENERATOR LAUNCH")
    print("WARNING: NOT FOR CLINICAL USE - EDUCATIONAL PURPOSES ONLY")
    print("=" * 70)

    generator = DataGenerator(seed=42, diabetes_threshold=18.7)
    n_people = 5000
    years = 20

    print(f"\nGeneration parameters:")
    print(f"  • Number of people: {n_people}")
    print(f"  • Observation period: {years} years")
    print(f"  • Random seed: {generator.seed}")
    print(f"  • Diabetes threshold: {generator.diabetes_threshold} (target prevalence ~15%)")
    print(f"  • BMR correction: {generator.BMR_MUSCLE_CORRECTION} kcal/kg/day (section 1.1)")
    print(f"  • Hard clipping: enabled for all biomarkers (section 7.4.5)")

    print(f"\n1/5 Generating baseline cohort...")
    cohort = generator.generate_cohort(n_people)
    generator.save(cohort, f"01_cohort_baseline_v{__version__}")

    print(f"\n2/5 Generating lifestyle history for {years} years...")
    lifestyle = generator.generate_lifestyle_history(cohort, years=years)
    generator.save(lifestyle, f"02_lifestyle_history_v{__version__}")

    print(f"\n3/5 Calculating biomarkers with hard clipping...")
    biomarkers = generator.calculate_biomarkers(cohort, lifestyle, years=years)
    generator.save(biomarkers, f"03_biomarkers_history_v{__version__}")

    print(f"\n4/5 Calculating medical risks...")
    risks = generator.calculate_health_risks(biomarkers, cohort, years=years)
    generator.save(risks, f"04_health_risks_v{__version__}")

    print(f"\n5/5 Creating aggregated dataset with MAR missing values...")
    aggregated = generator.create_aggregated_dataset(
        cohort, biomarkers, risks, years=years, apply_mar=True
    )
    generator.save(
        aggregated,
        f"05_aggregated_dataset_with_missing_v{__version__}",
        apply_mar=False,
    )

    print("\n" + "=" * 70)
    print("RISK DISTRIBUTION STATISTICS:")
    print("=" * 70)
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

    print("\n" + "=" * 70)
    print("GENERATION COMPLETED SUCCESSFULLY!")
    print("REMINDER: Data is synthetic, NOT for clinical use")
    print("=" * 70)


if __name__ == "__main__":
    main()
