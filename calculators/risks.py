"""
Health risk calculation module.
Calculates disease risks and generates binary outcomes.
"""
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import SimulationConfig


class RiskCalculator:
    """
    Calculates medical risks and generates binary outcomes.
    """

    def __init__(
            self,
            config: SimulationConfig,
            rng: np.random.Generator,
    ):
        self.config = config
        self.rng = rng

        self.diabetes_threshold = config.diabetes_threshold
        self.risk_intercepts = config.risk_intercepts

    def calculate(
            self,
            biomarkers_df: pd.DataFrame,
            cohort_df: pd.DataFrame,
            years: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Calculate medical risks and generate binary outcomes.
        """
        years = years or self.config.years
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
        final_alcohol = biomarkers_df[f"alcohol_g_per_week_{last_year}"].values
        final_fiber = biomarkers_df[f"fiber_g_day_{last_year}"].values
        final_saturated_fat = biomarkers_df[f"saturated_fat_pct_{last_year}"].values
        final_macro_imbalance = biomarkers_df[f"macro_imbalance_{last_year}"].values
        final_muscle = biomarkers_df[f"muscle_mass_factor_{last_year}"].values
        final_sleep = biomarkers_df[f"sleep_hours_{last_year}"].values
        cumulative_smoking = biomarkers_df[f"cumulative_smoking_{last_year}"].values

        person_ids = biomarkers_df["person_id"]
        genetic_data = cohort_df.set_index("person_id").reindex(person_ids)

        # Calculate individual disease risks
        cvd_risk, _ = self._calculate_cvd_risk(
            final_age, final_sbp, final_non_hdl_mmol,
            cumulative_smoking, final_hba1c,
            genetic_data["genetic_risk_cvd"].values
        )

        diabetes_risk, _ = self._calculate_diabetes_risk(
            final_age, final_bmi_corrected, final_hba1c,
            cumulative_smoking, final_fiber,
            genetic_data["genetic_risk_diabetes"].values,
            final_macro_imbalance, final_muscle, final_sleep
        )

        stroke_risk, _ = self._calculate_stroke_risk(
            final_age, final_sbp, final_hba1c,
            cumulative_smoking, final_bmi_corrected,
            genetic_data["genetic_risk_stroke"].values
        )

        nafld_risk, _ = self._calculate_nafld_risk(
            final_age, final_bmi_corrected, final_hba1c,
            final_saturated_fat, final_fiber,
            genetic_data["genetic_risk_nafld"].values
        )

        cancer_risk, _ = self._calculate_cancer_risk(
            final_age, final_fiber, final_alcohol,
            cumulative_smoking, final_bmi_corrected,
            genetic_data["genetic_risk_cancer"].values
        )

        cirrhosis_risk, _ = self._calculate_cirrhosis_risk(
            final_age, final_alcohol, cumulative_smoking,
            final_bmi_corrected,
            genetic_data["genetic_risk_cirrhosis"].values
        )

        # Calculate composite health score
        health_score = self._calculate_health_score(
            cvd_risk, diabetes_risk, stroke_risk,
            nafld_risk, cancer_risk, cirrhosis_risk
        )

        # Clip risks and generate binary outcomes
        risks_df["cvd_risk_10year"] = cvd_risk.clip(0, 0.5)
        risks_df["diabetes_risk_10year"] = diabetes_risk.clip(0, 0.5)
        risks_df["stroke_risk_10year"] = stroke_risk.clip(0, 0.4)
        risks_df["nafld_risk_10year"] = nafld_risk.clip(0, 0.6)
        risks_df["colorectal_cancer_risk_10year"] = cancer_risk.clip(0.05, 0.35)
        risks_df["cirrhosis_risk_10year"] = cirrhosis_risk.clip(0.02, 0.25)
        risks_df["health_score"] = health_score

        risks_df["has_cvd"] = self.rng.binomial(1, cvd_risk.clip(0, 0.5))
        risks_df["has_diabetes"] = self.rng.binomial(1, diabetes_risk.clip(0, 0.5))
        risks_df["has_stroke"] = self.rng.binomial(1, stroke_risk.clip(0, 0.4))
        risks_df["has_nafld"] = self.rng.binomial(1, nafld_risk.clip(0, 0.6))
        risks_df["has_colorectal_cancer"] = self.rng.binomial(1, cancer_risk.clip(0, 0.3))
        risks_df["has_cirrhosis"] = self.rng.binomial(1, cirrhosis_risk.clip(0, 0.2))

        # Check class balance
        self._check_class_balance(risks_df)

        # Determine primary death cause
        risks_df["primary_death_cause"] = self._determine_death_cause(
            cvd_risk, diabetes_risk, stroke_risk, cirrhosis_risk
        )

        # Estimate event age
        base_age = 60
        risks_df["estimated_event_age"] = base_age - 20 * (1 - risks_df["health_score"] / 100)
        risks_df["sex"] = genetic_data["sex"].values

        return risks_df

    def _calculate_cvd_risk(
            self,
            age: np.ndarray,
            sbp: np.ndarray,
            non_hdl_mmol: np.ndarray,
            pack_years: np.ndarray,
            hba1c: np.ndarray,
            genetic_cvd: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CVD risk calculation (SCORE2-like)."""
        logit = (
                self.risk_intercepts["cvd"]
                + 0.03 * (age - 40)
                + 0.02 * (sbp - 120)
                + 0.2 * (non_hdl_mmol - 3.5)
                + 0.05 * pack_years
                + 0.6 * (hba1c >= 6.5).astype(float)
                + 0.25 * genetic_cvd
        )
        risk = 1 / (1 + np.exp(-logit))
        return np.clip(risk, 0, 0.5), logit

    def _calculate_diabetes_risk(
            self,
            age: np.ndarray,
            bmi_corrected: np.ndarray,
            hba1c: np.ndarray,
            smoking: np.ndarray,
            fiber: np.ndarray,
            genetic_diabetes: np.ndarray,
            macro_imbalance: np.ndarray,
            muscle_mass: np.ndarray,
            sleep_hours: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Diabetes risk calculation (FINDRISC-like)."""
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
        points += (macro_imbalance > 2).astype(float) * 2
        points += (muscle_mass < 5).astype(float) * 1
        points += (sleep_hours < 6).astype(float) * 2
        points += (genetic_diabetes > 1.5).astype(float) * 1.5

        risk = 1 / (1 + np.exp(-0.4 * (points - self.diabetes_threshold)))
        return np.clip(risk, 0, 0.5), points

    def _calculate_stroke_risk(
            self,
            age: np.ndarray,
            sbp: np.ndarray,
            hba1c: np.ndarray,
            smoking: np.ndarray,
            bmi_corrected: np.ndarray,
            genetic_stroke: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stroke risk calculation."""
        smoking_binary = (smoking > 0).astype(float)
        logit = (
                self.risk_intercepts["stroke"]
                + 0.03 * (age - 50)
                + 0.015 * (sbp - 120)
                + 0.3 * (hba1c - 5.5)
                + 0.15 * smoking_binary
                + 0.03 * (bmi_corrected - 25)
                + 0.4 * genetic_stroke
        )
        risk = 1 / (1 + np.exp(-logit))
        return np.clip(risk, 0, 0.4), logit

    def _calculate_nafld_risk(
            self,
            age: np.ndarray,
            bmi_corrected: np.ndarray,
            hba1c: np.ndarray,
            saturated_fat: np.ndarray,
            fiber: np.ndarray,
            genetic_nafld: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """NAFLD risk calculation."""
        logit = (
                self.risk_intercepts["nafld"]
                + 0.02 * (age - 50)
                + 0.04 * (bmi_corrected - 25)
                + 0.25 * (hba1c - 5.5)
                + 0.4 * (saturated_fat - 0.3)
                + 0.3 * (fiber < 25).astype(float)
                + 0.4 * genetic_nafld
        )
        risk = 1 / (1 + np.exp(-logit))
        return np.clip(risk, 0, 0.5), logit

    def _calculate_cancer_risk(
            self,
            age: np.ndarray,
            fiber: np.ndarray,
            alcohol: np.ndarray,
            smoking: np.ndarray,
            bmi_corrected: np.ndarray,
            genetic_cancer: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Colorectal cancer risk calculation."""
        smoking_binary = (smoking > 0).astype(float)
        random_noise = self.rng.normal(0, 0.5, len(age))

        logit = (
                self.risk_intercepts["cancer"]
                + 0.06 * (age - 50)
                + 0.8 * (fiber < 20).astype(float)
                + 0.6 * (alcohol > 140).astype(float)
                + 0.7 * smoking_binary
                + 0.5 * (bmi_corrected > 30).astype(float)
                + 0.9 * genetic_cancer
                + random_noise
        )
        risk = 1 / (1 + np.exp(-logit))
        return np.clip(risk, 0.05, 0.35), logit

    def _calculate_cirrhosis_risk(
            self,
            age: np.ndarray,
            alcohol: np.ndarray,
            smoking: np.ndarray,
            bmi_corrected: np.ndarray,
            genetic_cirrhosis: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cirrhosis risk calculation."""
        smoking_binary = (smoking > 0).astype(float)
        random_noise = self.rng.normal(0, 0.4, len(alcohol))
        alcohol_effect = 0.8 * np.log1p(alcohol)

        logit = (
                self.risk_intercepts["cirrhosis"]
                + alcohol_effect
                + 0.015 * (age - 50)
                + 0.4 * smoking_binary
                + 0.5 * (bmi_corrected > 30).astype(float)
                + 0.8 * genetic_cirrhosis
                + random_noise
        )
        risk = 1 / (1 + np.exp(-logit))
        return np.clip(risk, 0.02, 0.25), logit

    @staticmethod
    def _calculate_health_score(
            cvd_risk: np.ndarray,
            diabetes_risk: np.ndarray,
            stroke_risk: np.ndarray,
            nafld_risk: np.ndarray,
            cancer_risk: np.ndarray,
            cirrhosis_risk: np.ndarray,
    ) -> np.ndarray:
        """Calculate composite health score (0-100 scale)."""
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

    def _check_class_balance(self, risks_df: pd.DataFrame) -> None:
        """Check and warn about class imbalance."""
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

    def _determine_death_cause(
            self,
            cvd_risk: np.ndarray,
            diabetes_risk: np.ndarray,
            stroke_risk: np.ndarray,
            cirrhosis_risk: np.ndarray,
    ) -> np.ndarray:
        """Determine primary death cause based on risks."""
        n = len(cvd_risk)
        conditions = [
            (cvd_risk > 0.3) & (self.rng.random(n) < 0.15),
            (diabetes_risk > 0.25) & (self.rng.random(n) < 0.08),
            (stroke_risk > 0.2) & (self.rng.random(n) < 0.1),
            (cirrhosis_risk > 0.15) & (self.rng.random(n) < 0.05),
        ]
        choices = ["Cardiovascular disease", "Diabetes complications", "Stroke", "Liver failure"]
        return np.select(conditions, choices, default="Other")
