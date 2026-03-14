"""
Biomarker calculation module.
Calculates annual biomarker measurements from lifestyle data.
"""
from typing import Optional

import numpy as np
import pandas as pd

from config import SimulationConfig


class BiomarkerCalculator:
    """
    Calculates biomarkers from lifestyle trajectories.
    """

    def __init__(
            self,
            config: SimulationConfig,
            rng: np.random.Generator,
    ):
        self.config = config
        self.rng = rng

        self.lipid_effect = config.lipid_effect
        self.measurement_noise = config.measurement_noise
        self.biomarker_ranges = config.biomarker_ranges
        self.final_biomarker_ranges = config.final_biomarker_ranges

    def calculate(
            self,
            cohort_df: pd.DataFrame,
            lifestyle_df: pd.DataFrame,
            years: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Calculate biomarkers from lifestyle with noise and clipping.
        """
        years = years or self.config.years
        n_people = len(lifestyle_df)
        person_ids = lifestyle_df["person_id"].values

        cohort_by_id = cohort_df.set_index("person_id")

        # Extract baseline data
        height = cohort_by_id.loc[person_ids]["height_cm"].values
        initial_hdl = cohort_by_id.loc[person_ids]["initial_hdl_mgdl"].values
        initial_tc = cohort_by_id.loc[person_ids]["initial_total_cholesterol_mgdl"].values
        lipid_factor_hdl = cohort_by_id.loc[person_ids]["lipid_factor_hdl"].values
        lipid_factor_tc = cohort_by_id.loc[person_ids]["lipid_factor_tc"].values
        initial_hba1c = cohort_by_id.loc[person_ids]["initial_hba1c_percent"].values
        initial_sbp = cohort_by_id.loc[person_ids]["initial_sbp_mmhg"].values

        # Genetic offsets (persistent across all years)
        genetic_hdl_offset = self.lipid_effect * lipid_factor_hdl
        genetic_tc_offset = -self.lipid_effect * lipid_factor_tc

        # Extract true weight and muscle
        weight_true = np.column_stack([
            lifestyle_df[f"weight_kg_true_{y}"].values for y in range(years)
        ])
        muscle_true = np.column_stack([
            lifestyle_df[f"muscle_mass_factor_true_{y}"].values for y in range(years)
        ])

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

        # Year 0 - apply lipid factor
        hdl_baseline = initial_hdl + genetic_hdl_offset
        tc_baseline = initial_tc + genetic_tc_offset
        hdl_true[:, 0] = self._clip_biomarker(hdl_baseline, "hdl")
        tc_true[:, 0] = self._clip_biomarker(tc_baseline, "tc")
        sbp_true[:, 0] = self._clip_biomarker(initial_sbp, "sbp")
        hba1c_true[:, 0] = self._clip_biomarker(initial_hba1c, "hba1c")

        # Pre-compute constants
        height_m = height / 100.0
        height_sq = height_m ** 2

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
            sex = cohort_by_id.loc[person_ids]["sex"].values

            # HDL calculation
            hdl_change = self._calculate_hdl_change(
                cardio_met[:, t], strength_met[:, t],
                delta_sat_fat, delta_fiber,
                muscle_true[:, t], bmi_curr, age_curr, sex
            )
            hdl_without_genetic = hdl_true[:, t - 1] - genetic_hdl_offset
            hdl_new_lifestyle = hdl_without_genetic + hdl_change
            hdl_true[:, t] = self._clip_biomarker(hdl_new_lifestyle + genetic_hdl_offset, "hdl")

            # Total cholesterol calculation
            tc_change = (
                    1.5 * (delta_sat_fat * 100.0)
                    - 0.5 * delta_fiber
                    - 0.3 * (total_met / 1000.0)
            )
            tc_without_genetic = tc_true[:, t - 1] - genetic_tc_offset
            tc_new_lifestyle = tc_without_genetic + tc_change
            tc_true[:, t] = self._clip_biomarker(tc_new_lifestyle + genetic_tc_offset, "tc")

            # SBP calculation
            sbp_change = self._calculate_sbp_change(
                delta_bmi, alcohol_day, delta_sodium,
                delta_stress, sleep_h, age_curr, sbp_true[:, t - 1],
                bmi_curr
            )
            sbp_true[:, t] = self._clip_biomarker(sbp_true[:, t - 1] + sbp_change, "sbp")

            # HbA1c calculation
            hba1c_change = self._calculate_hba1c_change(
                delta_simple_carbs, delta_bmi, delta_fiber,
                total_met, macro_imb[:, t], muscle_true[:, t],
                sleep_h, age_curr, hba1c_true[:, t - 1]
            )
            hba1c_true[:, t] = self._clip_biomarker(hba1c_true[:, t - 1] + hba1c_change, "hba1c")

        # Build output dictionary with noisy measurements
        data_dict = {"person_id": person_ids}
        noise = self.measurement_noise

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

            # BMI and corrected BMI
            bmi = data_dict[f"weight_kg_{y}"] / height_sq
            data_dict[f"bmi_{y}"] = self._clip_biomarker(bmi, "bmi")
            bmi_corr = bmi * (1 - 0.01 * muscle_true[:, y])
            data_dict[f"bmi_corrected_{y}"] = self._clip_biomarker(bmi_corr, "bmi")

            # Non-HDL
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

        return pd.DataFrame(data_dict)

    def _clip_biomarker(self, values: np.ndarray, marker_type: str) -> np.ndarray:
        """Clip biomarker values to physiological ranges."""
        marker_map = {
            "hdl": "hdl_mgdl",
            "tc": "total_cholesterol_mgdl",
            "sbp": "sbp_mmhg",
            "hba1c": "hba1c_percent",
            "weight": "weight_kg",
            "bmi": "bmi",
            "non_hdl_mgdl": "non_hdl_mgdl",
            "non_hdl_mmol": "non_hdl_mmol",
        }
        key = marker_map.get(marker_type, marker_type)
        if key in self.biomarker_ranges:
            low, high = self.biomarker_ranges[key]
            return np.clip(values, low, high)
        return values

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
        """Calculate annual HDL change with linear BMI effect."""
        cardio_benefit = 0.25 * (cardio_met / 1000.0)
        strength_benefit = 0.35 * (strength_met / 450.0)
        fiber_benefit = 0.25 * delta_fiber
        muscle_benefit = 0.05 * muscle_mass
        fat_penalty = -0.12 * (delta_sat_fat * 100.0)

        # ИЗМЕНЕНО: линейный эффект BMI вместо порогового
        # Было: -0.3 * np.maximum(0, bmi_curr - 25)
        bmi_penalty = -0.15 * (bmi_curr - 22)  # линейный от 22, не только >25

        # Сниженный возрастной штраф для компенсации: -0.06 вместо -0.08
        age_penalty = -0.06 * np.maximum(0, age_curr - 35)
        sex_factor = np.where(sex == 'F', 0.3, 0)

        annual_change = (
                cardio_benefit + strength_benefit + fiber_benefit +
                muscle_benefit + fat_penalty + bmi_penalty + age_penalty + sex_factor
        )
        return np.clip(annual_change, -1.5, 0.8)

    def _calculate_sbp_change(
            self,
            delta_bmi: np.ndarray,
            alcohol_day: np.ndarray,
            delta_sodium: np.ndarray,
            delta_stress: np.ndarray,
            sleep_hours: np.ndarray,
            age: np.ndarray,
            current_sbp: np.ndarray,
            bmi_curr: np.ndarray,  # ← ДОБАВЛЕНО
    ) -> np.ndarray:
        """Calculate annual SBP change with stabilized age effect and absolute BMI influence."""
        # Изменение BMI (краткосрочное)
        bmi_delta_effect = delta_bmi * 0.4

        # Абсолютный BMI (долгосрочное давление) — НОВОЕ
        bmi_absolute_effect = 0.3 * np.maximum(0, bmi_curr - 25)

        alcohol_effect = np.where(alcohol_day <= 10, 0, (alcohol_day - 10) * 0.015)
        sodium_effect = (delta_sodium / 100.0) * 0.12
        stress_effect = delta_stress * 0.25
        sleep_effect = np.maximum(0, 6 - sleep_hours) * 0.25

        # Сниженный возрастной эффект: 0.35 вместо 0.8
        age_effect = 0.35 * np.maximum(0, age - 35)

        # Норма SBP для возраста
        age_norm_sbp = 115 + 0.5 * np.maximum(0, age - 30)

        # Усиленная mean reversion: 0.06 вместо 0.03
        reversion = (age_norm_sbp - current_sbp) * 0.06

        total_change = (
                bmi_delta_effect + bmi_absolute_effect + alcohol_effect +
                sodium_effect + stress_effect + sleep_effect + age_effect + reversion
        )
        return np.clip(total_change, -4.0, 5.0)

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
            current_hba1c: np.ndarray,
    ) -> np.ndarray:
        """Calculate annual HbA1c change."""
        base_drift = 0.025
        age_norm = 5.0 + 0.02 * np.maximum(0, age - 30)
        reversion = (age_norm - current_hba1c) * 0.02
        carb_effect = (delta_simple_carbs / 10.0) * 0.03
        bmi_effect = delta_bmi * 0.006
        fiber_effect = delta_fiber * (-0.0004)
        met_effect = (total_met / 1000.0) * (-0.002)
        macro_effect = macro_imbalance * 0.004
        muscle_effect = -0.001 * muscle_mass
        sleep_deficit_effect = (sleep_hours < 6).astype(float) * 0.002

        total_change = (
                base_drift + reversion + carb_effect + bmi_effect +
                fiber_effect + met_effect + macro_effect + muscle_effect + sleep_deficit_effect
        )
        return np.clip(total_change, -0.1, 0.15)
