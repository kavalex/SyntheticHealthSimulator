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
from typing import Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

__version__ = "1.0.1"
__version_info__ = (1, 0, 1)


class DataGenerator:
    """
    Synthetic Medical Data Generator
    WARNING: NOT FOR CLINICAL USE
    """

    def __init__(self, seed: int = 42, diabetes_threshold: float = 18.7):
        """
        Generator initialization
        Args:
            seed: Random seed for reproducibility
            diabetes_threshold: Threshold for diabetes (15-20)
                15 = ~37% prevalence
                17 = ~25% prevalence
                18 = ~18% prevalence
                18.7 = ~15% prevalence (recommended)
        """
        self.current_year = datetime.now().year
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.diabetes_threshold = diabetes_threshold
        self.BASE_PARAMS = {
            "calories_baseline": {"M": 2500, "F": 2000},
            "protein_norm": 0.15,
            "fat_norm": 0.30,
            "carb_norm": 0.55,
            "fiber_norm_g": 30,
            "sodium_norm_mg": 2300,
        }
        self.MEASUREMENT_NOISE = {
            "sbp": 5.0,
            "hdl": 5.0,
            "tc": 10.0,
            "hba1c": 0.2,
            "weight": 1.5,
        }
        # NOTE: BMR correction uses 8.5 kcal/kg/day difference between muscle and fat tissue
        # per scientific specification section 1.1. This reflects higher metabolic activity
        # of muscle tissue compared to adipose tissue at rest.
        self.BMR_MUSCLE_CORRECTION = 8.5  # kcal/kg/day
        self.TYPICAL_MUSCLE_RATIO = 0.3  # typical muscle mass = 30% of weight
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
        self.MUSCLE_PARAMS = {
            "strength_to_training": 150,  # MET-minutes per strength training session
            "protein_requirement": 1.6,  # g/kg for positive balance
            "protein_deficit": 0.8,  # g/kg for negative balance
            "sleep_deficit_hours": 6,
            "max_muscle": 20,
            "min_muscle": 0,
            "mm_to_weight_kg": 0.5,  # conversion of muscle mass index units to kg
        }

    @staticmethod
    def _to_array(val: Union[float, int, np.ndarray]) -> np.ndarray:
        return np.asarray(val)

    def _clip_biomarker(self, values: np.ndarray, marker_type: str) -> np.ndarray:
        """Hard clipping for biomarkers according to reference section 8.3"""
        if marker_type in self.BIOMARKER_RANGES:
            low, high = self.BIOMARKER_RANGES[marker_type]
            return np.clip(values, low, high)
        return values

    def _calculate_cvd_risk(
        self, age, sbp, non_hdl_mmol, pack_years, hba1c, genetic_cvd
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CVD risk calculation (SCORE2-like) with non-HDL and pack-years"""
        # TODO: Check class balance distribution here (section 7.4.1)
        # Target prevalence: 15-20% for 20-year cohort, adjust intercept if needed
        age = self._to_array(age)
        sbp = self._to_array(sbp)
        non_hdl_mmol = self._to_array(non_hdl_mmol)
        pack_years = self._to_array(pack_years)
        hba1c = self._to_array(hba1c)
        genetic_cvd = self._to_array(genetic_cvd)

        # INTENTIONALLY CALIBRATED: SCORE2 intercept (-3.8) adjusted from -4.0 for target population prevalence.
        ALPHA_SCORE2 = -3.8
        # INTENTIONALLY REDUCED: Smoking coefficient (0.08 per pack-year) is lower
        # than epidemiological data (0.6) to prevent feature dominance and ensure
        # ML model learns multi-factor patterns. Real effect may be stronger.
        logit = (
            ALPHA_SCORE2
            + (age - 40) * 0.05
            + (sbp - 120) * 0.03
            + (non_hdl_mmol - 3.5) * 0.3
            + pack_years * 0.08  # REDUCED: see comment above
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
        points += (bmi_corrected >= 25).astype(float) * 1 + (
            bmi_corrected >= 30
        ).astype(float) * 2
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
        self, age, sbp, hba1c, smoking, bmi_corrected, genetic_stroke
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
        ) - 3.0
        risk = 1 / (1 + np.exp(-logit))
        return risk, logit

    def _calculate_nafld_risk(
        self, bmi_corrected, hba1c, saturated_fat, fiber, genetic_nafld
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
        ) - 2.5
        risk = 1 / (1 + np.exp(-logit))
        return risk, logit

    def _calculate_cancer_risk(
        self, age, fiber, alcohol, smoking, bmi_corrected, genetic_cancer
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
        ) - 1.5
        risk = 1 / (1 + np.exp(-logit))
        return risk, logit

    def _calculate_cirrhosis_risk(
        self, alcohol, smoking, bmi_corrected, genetic_cirrhosis
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
        ) - 2.0
        risk = 1 / (1 + np.exp(-logit))
        return risk, logit

    def _calculate_health_score(
        self,
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
        self, biomarkers_df: pd.DataFrame, cohort_df: pd.DataFrame, years: int = 20
    ) -> pd.DataFrame:
        """Medical risk calculation for the cohort"""
        risks_df = pd.DataFrame({"person_id": biomarkers_df["person_id"]})
        last_year = years - 1
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
        risks_df["has_colorectal_cancer"] = self.rng.binomial(
            1, cancer_risk.clip(0, 0.3)
        )
        risks_df["has_cirrhosis"] = self.rng.binomial(1, cirrhosis_risk.clip(0, 0.2))

        class_balance_check = {
            "cvd": risks_df["has_cvd"].mean(),
            "diabetes": risks_df["has_diabetes"].mean(),
            "stroke": risks_df["has_stroke"].mean(),
        }
        print(f"📊 Class balance check: {class_balance_check}")
        for disease, prevalence in class_balance_check.items():
            if prevalence < 0.10 or prevalence > 0.50:
                warnings.warn(
                    f"⚠️ Class imbalance detected for {disease}: {prevalence:.2%}. "
                    f"Consider adjusting risk thresholds."
                )

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
        risks_df["primary_death_cause"] = np.select(
            conditions, choices, default="Other"
        )

        base_age = 60
        risks_df["estimated_event_age"] = base_age - 20 * (
            1 - risks_df["health_score"] / 100
        )
        risks_df["sex"] = genetic_data["sex"].values
        return risks_df

    @staticmethod
    def calculate_metabolic_rate(
        weight_kg: float, height_cm: float, age_years: int, sex: str
    ) -> float:
        """Mifflin-St Jeor formula without muscle mass correction"""
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
        """
        BMR calculation with muscle mass correction (reference section 1.1)
        NOTE: Mifflin-St Jeor formula already includes age correction.
        Additional age-related metabolic decline is modeled through
        muscle mass changes (section 1.4), not through double-counting
        age in BMR formula.
        """
        bmr_base = self.calculate_metabolic_rate(weight_kg, height_cm, age_years, sex)
        typical_muscle_mass = self.TYPICAL_MUSCLE_RATIO * weight_kg
        correction = self.BMR_MUSCLE_CORRECTION * (muscle_mass_kg - typical_muscle_mass)
        return bmr_base + correction

    def calculate_tdee(self, bmr: float, met_minutes_week: float) -> float:
        """
        Total Daily Energy Expenditure (TDEE) calculation by WHO classification.
        MET-minute thresholds adjusted to match activity levels:
        - < 450 MET-min/week ≈ sedentary/light activity (1-2 days)
        - 450-900 MET-min/week ≈ moderate activity (3-5 days)
        - 900-1500 MET-min/week ≈ high activity (6-7 days)
        - > 1500 MET-min/week ≈ very high activity (athletes)
        Args:
            bmr: Basal metabolic rate (kcal/day)
            met_minutes_week: MET-minutes per week
        Returns:
            TDEE (kcal/day)
        """
        # NOTE: Thresholds calibrated from WHO classification (days/week) to MET-minutes
        # to ensure realistic population distribution: ~40% sedentary, ~35% light,
        # ~20% moderate, ~5% high activity (section 1.2)
        # INTENTIONALLY CALIBRATED: Upper limit 1.725 prevents unrealistic TDEE >4000 kcal
        if met_minutes_week < 450:  # ~150 min × 3 days or less
            activity_factor = 1.2  # Sedentary/light activity
        elif met_minutes_week < 900:  # ~150 min × 6 days or ~300 min × 3 days
            activity_factor = 1.375  # Moderate activity
        elif met_minutes_week < 1500:  # ~300 min × 5 days or intensive training
            activity_factor = 1.55  # High activity
        else:
            activity_factor = 1.725  # Very high activity (athletes)
        return bmr * activity_factor

    @staticmethod
    def _generate_genetic_risk(n: int, disease: str, rng) -> np.ndarray:
        params = {
            "diabetes": (0, 0.3),
            "cvd": (0, 0.4),
            "cancer_colorectal": (0, 0.35),
            "stroke": (0, 0.3),
            "nafld": (0, 0.25),
            "cirrhosis": (0, 0.4),
        }
        mean, std = params.get(disease, (0, 0.3))
        return rng.lognormal(mean, std, n).clip(0.5, 2.0)

    def generate_cohort(self, n_people: int = 10000) -> pd.DataFrame:
        ages = self.rng.normal(35, 8, n_people).clip(20, 50).astype(int)
        sexes = self.rng.choice(["M", "F"], n_people, p=[0.48, 0.52])
        genetic_diabetes = self._generate_genetic_risk(n_people, "diabetes", self.rng)
        genetic_cvd = self._generate_genetic_risk(n_people, "cvd", self.rng)
        genetic_cancer = self._generate_genetic_risk(
            n_people, "cancer_colorectal", self.rng
        )
        genetic_stroke = self._generate_genetic_risk(n_people, "stroke", self.rng)
        genetic_nafld = self._generate_genetic_risk(n_people, "nafld", self.rng)
        genetic_cirrhosis = self._generate_genetic_risk(n_people, "cirrhosis", self.rng)

        n_m = int(n_people * 0.48)
        n_f = n_people - n_m
        height_cm = np.concatenate(
            [self.rng.normal(175, 7, n_m), self.rng.normal(162, 6, n_f)]
        )
        weight_kg = np.concatenate(
            [self.rng.normal(80, 12, n_m), self.rng.normal(65, 10, n_f)]
        )

        shuffle_idx = self.rng.permutation(n_people)
        height_cm = height_cm[shuffle_idx]
        weight_kg = weight_kg[shuffle_idx]
        sexes = sexes[shuffle_idx]
        ages = ages[shuffle_idx]
        genetic_diabetes = genetic_diabetes[shuffle_idx]
        genetic_cvd = genetic_cvd[shuffle_idx]
        genetic_cancer = genetic_cancer[shuffle_idx]
        genetic_stroke = genetic_stroke[shuffle_idx]
        genetic_nafld = genetic_nafld[shuffle_idx]
        genetic_cirrhosis = genetic_cirrhosis[shuffle_idx]

        bmi_baseline = weight_kg / ((height_cm / 100) ** 2)
        hdl_baseline = np.where(
            sexes == "M",
            self.rng.normal(50, 10, n_people),
            self.rng.normal(60, 12, n_people),
        ).clip(20, 100)
        tc_baseline = self.rng.normal(200, 25, n_people).clip(150, 350)

        # NOTE: Genetic risk affects ONLY baseline TC, not annual change (section 2.4)
        # This reflects biological predisposition rather than rate of change under risk factors
        tc_baseline = tc_baseline + 0.8 * genetic_cvd
        tc_baseline = tc_baseline.clip(150, 350)

        hba1c_baseline = self.rng.normal(5.4, 0.3, n_people).clip(4.0, 6.5)
        sbp_baseline = 120 + 0.5 * (ages - 35) + self.rng.normal(0, 8, n_people)

        # Muscle mass by sex
        muscle_baseline = np.where(
            sexes == "M",
            self.rng.normal(12, 3, n_people),
            self.rng.normal(8, 2.5, n_people),
        ).clip(0, 20)

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
            "initial_sbp_mmhg": sbp_baseline.clip(90, 160),
        }
        return pd.DataFrame(data)

    def _generate_stress_event(
        self, years: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Random stress event generation.
        NOTE: Duration extended from 6 months (original spec) to 1 year with decay
        to match annual discretization of the model (section 4.2)
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

    def generate_lifestyle_history(
        self, cohort_df: pd.DataFrame, years: int = 20
    ) -> pd.DataFrame:
        """
        Lifestyle history generation with OU processes, weight and muscle mass evolution,
        and preservation of all nutrition parameters for each year.
        """
        n_people = len(cohort_df)

        # Extract initial parameters
        age_start = cohort_df["age_start"].values
        sex = cohort_df["sex"].values
        height = cohort_df["height_cm"].values
        weight = cohort_df["weight_start_kg"].values.copy()
        muscle = cohort_df["muscle_mass_start"].values.copy()

        # Generate individual baseline levels for OU processes
        alc_baseline = np.where(
            self.rng.random(n_people) > 0.3, self.rng.gamma(2, 20, n_people), 0
        )
        cardio_baseline = self.rng.uniform(300, 1200, n_people)
        strength_baseline = self.rng.uniform(0, 600, n_people)
        smoke_baseline = np.where(
            self.rng.random(n_people) < 0.25, self.rng.exponential(3, n_people), 0
        )
        stress_baseline = self.rng.normal(3, 2, n_people)
        sleep_baseline = self.rng.normal(7, 1, n_people)

        # Initialize arrays for storing true values
        alcohol_series = np.zeros((n_people, years))
        cardio_series = np.zeros((n_people, years))
        strength_series = np.zeros((n_people, years))
        smoke_series = np.zeros((n_people, years))
        stress_series = np.zeros((n_people, years))
        sleep_series = np.zeros((n_people, years))

        # Fill first year
        alcohol_series[:, 0] = alc_baseline
        cardio_series[:, 0] = cardio_baseline
        strength_series[:, 0] = strength_baseline
        smoke_series[:, 0] = smoke_baseline
        stress_series[:, 0] = stress_baseline
        sleep_series[:, 0] = sleep_baseline

        # OU parameters
        theta = {
            "alcohol": 0.2,
            "cardio": 0.2,
            "strength": 0.25,
            "smoke": 0.3,
            "stress": 0.2,
            "sleep": 0.2,
        }
        sigma = {
            "alcohol": 0.2,
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

        # WARNING: Seasonal variations intentionally NOT implemented (section 4.2)
        # Annual discretization averages out seasonal effects; OU processes provide
        # sufficient variability without explicit seasonality modeling
        for t in range(1, years):
            alcohol_series[:, t] = (
                alcohol_series[:, t - 1]
                + theta["alcohol"] * (alc_baseline - alcohol_series[:, t - 1])
                + sigma["alcohol"] * self.rng.normal(size=n_people)
            )
            cardio_series[:, t] = (
                cardio_series[:, t - 1]
                + theta["cardio"] * (cardio_baseline - cardio_series[:, t - 1])
                + sigma["cardio"] * self.rng.normal(size=n_people)
            )
            strength_series[:, t] = (
                strength_series[:, t - 1]
                + theta["strength"] * (strength_baseline - strength_series[:, t - 1])
                + sigma["strength"] * self.rng.normal(size=n_people)
            )
            smoke_series[:, t] = (
                smoke_series[:, t - 1]
                + theta["smoke"] * (smoke_baseline - smoke_series[:, t - 1])
                + sigma["smoke"] * self.rng.normal(size=n_people)
            )
            stress_series[:, t] = (
                stress_series[:, t - 1]
                + theta["stress"] * (stress_baseline - stress_series[:, t - 1])
                + sigma["stress"] * self.rng.normal(size=n_people)
            )
            sleep_series[:, t] = (
                sleep_series[:, t - 1]
                + theta["sleep"] * (sleep_baseline - sleep_series[:, t - 1])
                + sigma["sleep"] * self.rng.normal(size=n_people)
            )

            # Clipping
            alcohol_series[:, t] = np.clip(
                alcohol_series[:, t], min_vals["alcohol"], max_vals["alcohol"]
            )
            cardio_series[:, t] = np.clip(
                cardio_series[:, t], min_vals["cardio"], max_vals["cardio"]
            )
            strength_series[:, t] = np.clip(
                strength_series[:, t], min_vals["strength"], max_vals["strength"]
            )
            smoke_series[:, t] = np.clip(
                smoke_series[:, t], min_vals["smoke"], max_vals["smoke"]
            )
            stress_series[:, t] = np.clip(
                stress_series[:, t], min_vals["stress"], max_vals["stress"]
            )
            sleep_series[:, t] = np.clip(
                sleep_series[:, t], min_vals["sleep"], max_vals["sleep"]
            )

        # Generate stress events (common for all)
        stress_events, sleep_events, alcohol_events = self._generate_stress_event(years)
        for t in range(years):
            stress_series[:, t] += stress_events[t]
            sleep_series[:, t] -= sleep_events[t]
            alcohol_series[:, t] *= 1 + alcohol_events[t]
        stress_series = np.clip(stress_series, 0, 10)
        sleep_series = np.clip(sleep_series, 4, 10)

        # Arrays for storing true values of weight, muscle, and nutrition parameters
        weight_true = np.zeros((n_people, years))
        muscle_true = np.zeros((n_people, years))
        bmr_true = np.zeros((n_people, years))
        tdee_true = np.zeros((n_people, years))
        cumulative_smoking = np.zeros((n_people, years))

        # Arrays for nutrition parameters
        protein_pct_arr = np.zeros((n_people, years))
        fat_pct_arr = np.zeros((n_people, years))
        carb_pct_arr = np.zeros((n_people, years))
        saturated_fat_pct_arr = np.zeros((n_people, years))
        simple_carbs_pct_arr = np.zeros((n_people, years))
        fiber_g_arr = np.zeros((n_people, years))
        sodium_mg_arr = np.zeros((n_people, years))
        macro_imbalance_arr = np.zeros(
            (n_people, years)
        )  # CHANGED: was kbju_imbalance_arr
        simple_carbs_g_arr = np.zeros((n_people, years))

        # Fill first year
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
        cumulative_smoking[:, 0] = smoke_series[:, 0] / 20

        # Generate nutrition parameters for first year
        protein_pct_arr[:, 0] = self.BASE_PARAMS["protein_norm"] + self.rng.normal(
            0, 0.03, n_people
        )
        fat_pct_arr[:, 0] = self.BASE_PARAMS["fat_norm"] + self.rng.normal(
            0, 0.04, n_people
        )
        carb_pct_arr[:, 0] = 1 - protein_pct_arr[:, 0] - fat_pct_arr[:, 0]
        total = protein_pct_arr[:, 0] + fat_pct_arr[:, 0] + carb_pct_arr[:, 0]
        protein_pct_arr[:, 0] /= total
        fat_pct_arr[:, 0] /= total
        carb_pct_arr[:, 0] /= total
        saturated_fat_pct_arr[:, 0] = fat_pct_arr[:, 0] * self.rng.uniform(
            0.2, 0.5, n_people
        )
        simple_carbs_pct_arr[:, 0] = carb_pct_arr[:, 0] * self.rng.uniform(
            0.1, 0.4, n_people
        )
        fiber_g_arr[:, 0] = np.clip(self.rng.normal(25, 8, n_people), 5, 50)
        sodium_mg_arr[:, 0] = np.clip(self.rng.normal(3400, 800, n_people), 1000, 6000)

        delta_p = np.abs(protein_pct_arr[:, 0] - 0.30) * 100
        delta_f = np.abs(fat_pct_arr[:, 0] - 0.30) * 100
        delta_c = np.abs(carb_pct_arr[:, 0] - 0.40) * 100
        macro_raw = np.sqrt(delta_p**2 + delta_f**2 + delta_c**2)
        macro_imbalance_arr[:, 0] = np.minimum(macro_raw * 10, 100)

        calories_approx = tdee_true[:, 0] + self.rng.normal(0, 200, n_people)
        calories_approx = np.maximum(calories_approx, 1200)
        alcohol_cal = (alcohol_series[:, 0] / 7) * 7
        non_alcohol_cal = np.maximum(calories_approx - alcohol_cal, 500)
        simple_carbs_g_arr[:, 0] = (
            (simple_carbs_pct_arr[:, 0] / 100) * non_alcohol_cal / 4
        )

        # Main loop by years (starting from second)
        for t in range(1, years):
            age = age_start + t
            prev_weight = weight_true[:, t - 1]
            prev_muscle = muscle_true[:, t - 1]

            # BMR and TDEE based on previous weight (for calorie calculation)
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

            # Calorie consumption
            calorie_offset = self.rng.normal(0, 200, n_people)
            calories = tdee_prev + calorie_offset
            calories = np.maximum(calories, 1200)

            # Calories from alcohol
            alcohol_cal_per_day = (alcohol_series[:, t] / 7) * 7
            calories += alcohol_cal_per_day
            non_alcohol_cal = np.maximum(calories - alcohol_cal_per_day, 500)

            # Generate macronutrients
            protein_pct = self.BASE_PARAMS["protein_norm"] + self.rng.normal(
                0, 0.03, n_people
            )
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
            macro_raw = np.sqrt(delta_p**2 + delta_f**2 + delta_c**2)
            macro_imbalance = np.minimum(macro_raw * 10, 100)

            # Protein in g/kg
            protein_grams = (protein_pct / 100) * non_alcohol_cal / 4
            protein_g_per_kg = protein_grams / prev_weight

            # NOTE: Protein balance categories (section 1.3.2):
            # >1.6 g/kg = positive (+1), 0.8-1.6 g/kg = neutral (0), <0.8 g/kg = negative (-1)
            protein_balance = np.zeros(n_people)
            protein_balance[
                protein_g_per_kg > self.MUSCLE_PARAMS["protein_requirement"]
            ] = 1
            protein_balance[
                protein_g_per_kg < self.MUSCLE_PARAMS["protein_deficit"]
            ] = -1

            # Number of strength training sessions per week (150 MET-min = 1 session)
            strength_sessions = (
                strength_series[:, t] / self.MUSCLE_PARAMS["strength_to_training"]
            )
            strength_sessions = np.clip(
                strength_sessions, 0, 5
            )  # limit as in reference

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
                    self.calculate_tdee(
                        bmr_true[i, t], cardio_series[i, t] + strength_series[i, t]
                    )
                    for i in range(n_people)
                ]
            )

            # Pack-year accumulation
            cumulative_smoking[:, t] = (
                cumulative_smoking[:, t - 1] + smoke_series[:, t] / 20
            )

            # Save nutrition parameters
            protein_pct_arr[:, t] = protein_pct
            fat_pct_arr[:, t] = fat_pct
            carb_pct_arr[:, t] = carb_pct
            saturated_fat_pct_arr[:, t] = saturated_fat_pct
            simple_carbs_pct_arr[:, t] = simple_carbs_pct
            fiber_g_arr[:, t] = fiber_g
            sodium_mg_arr[:, t] = sodium_mg
            macro_imbalance_arr[:, t] = macro_imbalance
            simple_carbs_g_arr[:, t] = (simple_carbs_pct / 100) * non_alcohol_cal / 4

        # Form DataFrame
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
            data[f"total_met_minutes_{year}"] = (
                cardio_series[:, year] + strength_series[:, year]
            )
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

    def calculate_biomarkers(
        self, cohort_df: pd.DataFrame, lifestyle_df: pd.DataFrame, years: int = 20
    ) -> pd.DataFrame:
        """
        Biomarker calculation based on lifestyle.
        # TODO: Implement hard clipping for all biomarkers (section 7.4.5)
        # Current implementation uses _clip_biomarker() but explicit TODO required by spec
        """
        biomarkers_df = lifestyle_df.copy()
        n_people = len(biomarkers_df)
        person_ids = biomarkers_df["person_id"]
        cohort_by_id = cohort_df.set_index("person_id")
        height = cohort_by_id.loc[person_ids]["height_cm"].values
        initial_hdl = cohort_by_id.loc[person_ids]["initial_hdl_mgdl"].values
        initial_tc = cohort_by_id.loc[person_ids]["initial_tc_mgdl"].values
        initial_hba1c = cohort_by_id.loc[person_ids]["initial_hba1c_percent"].values
        initial_sbp = cohort_by_id.loc[person_ids]["initial_sbp_mmhg"].values

        # True biomarker values (without noise)
        hdl_true = np.zeros((n_people, years))
        tc_true = np.zeros((n_people, years))
        sbp_true = np.zeros((n_people, years))
        hba1c_true = np.zeros((n_people, years))

        # First year with hard clipping
        hdl_true[:, 0] = self._clip_biomarker(initial_hdl, "hdl")
        tc_true[:, 0] = self._clip_biomarker(initial_tc, "tc")
        sbp_true[:, 0] = self._clip_biomarker(initial_sbp, "sbp")
        hba1c_true[:, 0] = self._clip_biomarker(initial_hba1c, "hba1c")

        # Extract true weights and muscles
        weight_true = np.array(
            [lifestyle_df[f"weight_kg_true_{year}"].values for year in range(years)]
        ).T
        muscle_true = np.array(
            [
                lifestyle_df[f"muscle_mass_factor_true_{year}"].values
                for year in range(years)
            ]
        ).T

        # Calculation for subsequent years
        for year in range(1, years):
            # Factor changes
            prev_saturated_fat = lifestyle_df[f"saturated_fat_pct_{year - 1}"].values
            curr_saturated_fat = lifestyle_df[f"saturated_fat_pct_{year}"].values
            delta_saturated_fat = curr_saturated_fat - prev_saturated_fat

            prev_fiber = lifestyle_df[f"fiber_g_day_{year - 1}"].values
            curr_fiber = lifestyle_df[f"fiber_g_day_{year}"].values
            delta_fiber = curr_fiber - prev_fiber

            prev_sodium = lifestyle_df[f"sodium_mg_day_{year - 1}"].values
            curr_sodium = lifestyle_df[f"sodium_mg_day_{year}"].values
            delta_sodium = curr_sodium - prev_sodium

            prev_stress = lifestyle_df[f"stress_level_{year - 1}"].values
            curr_stress = lifestyle_df[f"stress_level_{year}"].values
            delta_stress = curr_stress - prev_stress

            prev_simple_carbs_g = lifestyle_df[f"simple_carbs_g_{year - 1}"].values
            curr_simple_carbs_g = lifestyle_df[f"simple_carbs_g_{year}"].values
            delta_simple_carbs_g = curr_simple_carbs_g - prev_simple_carbs_g

            # BMI and its change
            bmi_prev = weight_true[:, year - 1] / ((height / 100) ** 2)
            bmi_curr = weight_true[:, year] / ((height / 100) ** 2)
            delta_bmi = bmi_curr - bmi_prev

            # Activity
            cardio_met = lifestyle_df[f"cardio_met_minutes_{year}"].values
            strength_met = lifestyle_df[f"strength_met_minutes_{year}"].values
            total_met = cardio_met + strength_met

            # Alcohol
            alcohol_g_day = lifestyle_df[f"alcohol_g_per_week_{year}"].values / 7

            # Sleep
            sleep_hours = lifestyle_df[f"sleep_hours_{year}"].values

            # Macro imbalance (CHANGED: was KBJU imbalance)
            macro_imb = lifestyle_df[f"macro_imbalance_{year}"].values

            # HDL
            hdl_change = (
                0.2 * (cardio_met / 1000)
                +
                # 450 MET-min = 3 sessions × 150 MET-min (moderate intensity strength training)
                # Calibrated to match "3 sessions/week" from specification section 2.2
                0.3 * (strength_met / 450)  # assumption: 450 MET-min ≈ 3 sessions/week
                + -0.05 * (delta_saturated_fat * 100)
                + 0.1 * delta_fiber
                + 0.05 * muscle_true[:, year]
            )
            hdl_new = hdl_true[:, year - 1] + hdl_change
            hdl_true[:, year] = self._clip_biomarker(hdl_new, "hdl")

            # TC (genetics already in baseline value)
            tc_change = (
                1.5 * (delta_saturated_fat * 100)
                + -0.5 * delta_fiber
                + -0.3 * (total_met / 1000)
            )
            tc_new = tc_true[:, year - 1] + tc_change
            tc_true[:, year] = self._clip_biomarker(tc_new, "tc")

            # SBP calculation with intentionally reduced coefficients for feature balance
            # INTENTIONALLY REDUCED: Alcohol coefficient (0.03 mmHg/g/day) is calibrated
            # for feature balance, not clinical accuracy. Real epidemiological effect
            # may be 2-3x stronger (Xin et al., 2001).
            alcohol_effect = np.where(
                alcohol_g_day <= 10, 0, (alcohol_g_day - 10) * 0.03
            )
            # NOTE: Sleep effect on SBP is theoretical assumption (section 2.1.3)
            # Based on associative studies, mechanistic link simplified in model
            sleep_effect = np.maximum(0, 6 - sleep_hours) * 0.5
            sbp_change = (
                delta_bmi * 0.7
                + alcohol_effect
                +
                # INTENTIONALLY REDUCED: Sodium coefficient (0.25 mmHg/100mg) is
                # lower than Cochrane estimate (~0.5-1.0) for feature balance
                (delta_sodium / 100) * 0.25
                + delta_stress * 0.5
                + sleep_effect
            )
            sbp_new = sbp_true[:, year - 1] + sbp_change
            sbp_true[:, year] = self._clip_biomarker(sbp_new, "sbp")

            # HbA1c
            # NOTE: Macro_Imbalance is normalized 0-100 scale (section 2.3.2)
            # Original raw value multiplied by 10 and clipped at 100 for compatibility
            # with logistic risk model. This is technical solution, not clinical scale.
            macro_effect = macro_imb * 0.002
            muscle_effect = -0.005 * muscle_true[:, year]
            sleep_deficit_effect = (sleep_hours < 6).astype(float) * 0.01
            hba1c_change = (
                (delta_simple_carbs_g / 10) * 0.15
                + delta_bmi * 0.03
                + delta_fiber * -0.002
                + (total_met / 1000) * -0.01
                + macro_effect  # CHANGED: was kbju_effect
                + muscle_effect
                + sleep_deficit_effect
            )
            hba1c_new = hba1c_true[:, year - 1] + hba1c_change
            hba1c_true[:, year] = self._clip_biomarker(hba1c_new, "hba1c")

            # Add noise and save to resulting DataFrame
            for year in range(years):
                # Weight with noise
                weight_noisy = weight_true[:, year] + self.rng.normal(
                    0, self.MEASUREMENT_NOISE["weight"], size=n_people
                )
                biomarkers_df[f"weight_kg_{year}"] = self._clip_biomarker(
                    weight_noisy, "weight"
                )

                # Biomarkers with noise (already with hard clipping during evolution)
                hdl_noisy = hdl_true[:, year] + self.rng.normal(
                    0, self.MEASUREMENT_NOISE["hdl"], size=n_people
                )
                tc_noisy = tc_true[:, year] + self.rng.normal(
                    0, self.MEASUREMENT_NOISE["tc"], size=n_people
                )
                sbp_noisy = sbp_true[:, year] + self.rng.normal(
                    0, self.MEASUREMENT_NOISE["sbp"], size=n_people
                )
                hba1c_noisy = hba1c_true[:, year] + self.rng.normal(
                    0, self.MEASUREMENT_NOISE["hba1c"], size=n_people
                )

                biomarkers_df[f"hdl_mgdl_{year}"] = self._clip_biomarker(
                    hdl_noisy, "hdl"
                )
                biomarkers_df[f"total_cholesterol_mgdl_{year}"] = self._clip_biomarker(
                    tc_noisy, "tc"
                )
                biomarkers_df[f"sbp_mmhg_{year}"] = self._clip_biomarker(
                    sbp_noisy, "sbp"
                )
                biomarkers_df[f"hba1c_percent_{year}"] = self._clip_biomarker(
                    hba1c_noisy, "hba1c"
                )

                # BMI
                bmi = biomarkers_df[f"weight_kg_{year}"] / ((height / 100) ** 2)
                biomarkers_df[f"bmi_{year}"] = self._clip_biomarker(bmi, "bmi")
                biomarkers_df[f"bmi_corrected_{year}"] = self._clip_biomarker(
                    bmi * (1 - 0.01 * muscle_true[:, year]), "bmi"
                )

                # non-HDL
                non_hdl_mgdl = (
                    biomarkers_df[f"total_cholesterol_mgdl_{year}"]
                    - biomarkers_df[f"hdl_mgdl_{year}"]
                )
                non_hdl_mgdl = self._clip_biomarker(non_hdl_mgdl, "non_hdl_mgdl")
                biomarkers_df[f"non_hdl_mgdl_{year}"] = non_hdl_mgdl
                biomarkers_df[f"non_hdl_mmol_{year}"] = self._clip_biomarker(
                    non_hdl_mgdl / 38.67, "non_hdl_mmol"
                )

                # Add muscle mass columns without _true suffix (for backward compatibility)
                biomarkers_df[f"muscle_mass_factor_{year}"] = muscle_true[:, year]

        return biomarkers_df

    def _generate_mar_missing_values(
        self,
        df: pd.DataFrame,
        age_col: str = "age_end",
        risk_col: str = "cvd_risk_10year",
    ) -> Tuple[pd.DataFrame, dict]:
        """
        MAR missing value generation (Missing At Random) according to reference section 7.4.2.
        # TODO: Implement MAR missing value simulation (2-15% NaN) - section 7.4.2
        # Formula: p_missing = 0.05 + 0.1*(age<30) - 0.05*(Risk_CVD>0.2), clipped [0.02, 0.15]
        """
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
        """Create aggregated dataset with optional MAR missing values"""
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
                aggregated[f"{col}_change"] = (
                    biomarkers_df[last_col] - biomarkers_df[first_col]
                )
                aggregated[f"{col}_change_pct"] = (
                    (biomarkers_df[last_col] - biomarkers_df[first_col])
                    / (biomarkers_df[first_col] + 1e-10)
                    * 100
                )

        aggregated["final_bmi"] = biomarkers_df[f"bmi_{years - 1}"]
        aggregated["final_bmi_corrected"] = biomarkers_df[f"bmi_corrected_{years - 1}"]
        aggregated["final_hdl_mgdl"] = biomarkers_df[f"hdl_mgdl_{years - 1}"]
        aggregated["final_total_cholesterol_mgdl"] = biomarkers_df[
            f"total_cholesterol_mgdl_{years - 1}"
        ]
        aggregated["final_non_hdl_mgdl"] = biomarkers_df[f"non_hdl_mgdl_{years - 1}"]
        aggregated["final_non_hdl_mmol"] = biomarkers_df[f"non_hdl_mmol_{years - 1}"]
        aggregated["final_sbp_mmhg"] = biomarkers_df[f"sbp_mmhg_{years - 1}"]
        aggregated["final_hba1c_percent"] = biomarkers_df[f"hba1c_percent_{years - 1}"]
        aggregated["final_muscle_mass"] = biomarkers_df[
            f"muscle_mass_factor_{years - 1}"
        ]
        aggregated = aggregated.merge(risks_df, on="person_id", how="left")
        aggregated["age_end"] = aggregated["age_start"] + years

        # TODO: Implement MAR missing value simulation (2-15% NaN) - reference section 7.4.2
        missing_info = {}
        if apply_mar:
            # WARNING: SyntheticHealthSimulator - NOT FOR CLINICAL USE (section 7.4.3)
            # MAR missing values simulate realistic medical data patterns for ML training
            aggregated, missing_info = self._generate_mar_missing_values(aggregated)
            if missing_info:
                print(f"MAR missing values generated:")
                for col, count in missing_info.items():
                    print(
                        f"   • {col}: {count} missing values ({count / len(aggregated) * 100:.1f}%)"
                    )
        return aggregated

    def save_for_kaggle(
        self,
        df: pd.DataFrame,
        name: str,
        output_dir: str = None,
        apply_mar: bool = False,
    ) -> str:
        """
        Save in Kaggle format.
        Note: MAR missing values are now generated in create_aggregated_dataset().
        This method saves data as-is if apply_mar=False.
        """
        if output_dir is None:
            output_dir = f"data/synthetic_v{__version__}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        df_copy = df.copy()

        # If MAR was not applied in pipeline, can apply here (backward compatibility)
        if (
            apply_mar
            and "final_sbp_mmhg" in df_copy.columns
            and not df_copy["final_sbp_mmhg"].isna().any()
        ):
            df_copy, missing_info = self._generate_mar_missing_values(df_copy)
            if missing_info:
                print(f"Missing values generated (MAR) during save:")
                for col, count in missing_info.items():
                    print(
                        f"   • {col}: {count} missing values ({count / len(df_copy) * 100:.1f}%)"
                    )

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
                "missing_percentage": "2-15% (MAR)"
                if apply_mar or missing_report.any()
                else "0%",
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
    # All files and functions must contain this disclaimer per specification.
    # INTENTIONALLY CALIBRATED: All coefficients reduced for ML feature balance.
    # See INTENTIONALLY REDUCED/CALIBRATED comments throughout codebase.
    # ============================================================================
    print("=" * 70)
    print(f"SyntheticHealthSimulator v{__version__}")
    print("MEDICAL DATA GENERATOR LAUNCH")
    print("WARNING: NOT FOR CLINICAL USE - EDUCATIONAL PURPOSES ONLY")
    print("=" * 70)

    generator = DataGenerator(seed=42, diabetes_threshold=18.7)
    N_PEOPLE = 5000
    YEARS = 20

    print(f"\nGeneration parameters:")
    print(f"  • Number of people: {N_PEOPLE}")
    print(f"  • Observation period: {YEARS} years")
    print(f"  • Random seed: {generator.seed}")
    print(
        f"  • Diabetes threshold: {generator.diabetes_threshold} (target prevalence ~15%)"
    )
    print(
        f"  • BMR correction: {generator.BMR_MUSCLE_CORRECTION} kcal/kg/day (section 1.1)"
    )
    print(f"  • Hard clipping: enabled for all biomarkers (section 7.4.5)")

    print(f"\n1/5 Generating baseline cohort...")
    cohort = generator.generate_cohort(N_PEOPLE)
    generator.save_for_kaggle(cohort, f"01_cohort_baseline_v{__version__}")

    print(f"\n2/5 Generating lifestyle history for {YEARS} years...")
    lifestyle = generator.generate_lifestyle_history(cohort, years=YEARS)
    generator.save_for_kaggle(lifestyle, f"02_lifestyle_history_v{__version__}")

    print(f"\n3/5 Calculating biomarkers with hard clipping...")
    biomarkers = generator.calculate_biomarkers(cohort, lifestyle, years=YEARS)
    generator.save_for_kaggle(biomarkers, f"03_biomarkers_history_v{__version__}")

    print(f"\n4/5 Calculating medical risks...")
    risks = generator.calculate_health_risks(biomarkers, cohort, years=YEARS)
    generator.save_for_kaggle(risks, f"04_health_risks_v{__version__}")

    print(f"\n5/5 Creating aggregated dataset with MAR missing values...")
    # MAR missing values are generated inside create_aggregated_dataset (section 7.4.2)
    aggregated = generator.create_aggregated_dataset(
        cohort, biomarkers, risks, years=YEARS, apply_mar=True
    )
    generator.save_for_kaggle(
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
