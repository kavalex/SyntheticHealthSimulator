"""
Lifestyle trajectory simulation module.
Generates 20-year lifestyle histories using OU processes.
"""
from typing import Optional, Dict

import numpy as np
import pandas as pd

from config import SimulationConfig
from utils.helpers import clip_biomarker, calculate_metabolic_rate, calculate_tdee


class LifestyleSimulator:
    """
    Simulates lifestyle trajectories and weight evolution.
    """

    def __init__(
            self,
            config: SimulationConfig,
            rng: np.random.Generator,
    ):
        self.config = config
        self.rng = rng

        # OU parameters
        self.ou_params = config.ou_params

        # Calorie parameters
        self.calorie_params = config.calorie_params

        # Muscle parameters
        self.muscle_params = config.muscle_params

        # Base metabolic parameters
        self.base_calories = config.base_calories
        self.macro_norms = config.macro_norms

        # BMR/TDEE parameters
        self.bmr_correction = config.muscle_params["bmr_correction"]
        self.typical_ratio = config.muscle_params["typical_ratio"]
        self.mm_to_weight_kg = config.muscle_params["mm_to_weight_kg"]

        # Biomarker ranges for clipping
        self.biomarker_ranges = config.biomarker_ranges

    def simulate(
            self,
            cohort_df: pd.DataFrame,
            years: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate lifestyle history with OU processes and weight evolution.

        Args:
            cohort_df: Baseline cohort DataFrame
            years: Number of years to simulate (default from config)

        Returns:
            DataFrame with yearly lifestyle data
        """
        years = years or self.config.years
        n_people = len(cohort_df)

        # Extract initial parameters
        age_start = cohort_df["age_start"].values
        sex = cohort_df["sex"].values
        height = cohort_df["height_cm"].values
        weight = cohort_df["weight_start_kg"].values.copy()
        muscle = cohort_df["muscle_mass_start"].values.copy()

        # Generate OU baselines
        baselines = self._generate_ou_baselines(n_people)

        # Generate OU trajectories
        trajectories = self._generate_ou_trajectories(baselines, years, n_people)

        # Apply stress events
        trajectories = self._apply_stress_events(trajectories, years)

        # Simulate weight evolution
        weight_data = self._simulate_weight_evolution(
            cohort_df, trajectories, years
        )

        # Build DataFrame
        data = {"person_id": cohort_df["person_id"].values}

        for year in range(years):
            data[f"age_{year}"] = age_start + year
            data[f"weight_kg_true_{year}"] = weight_data["weight"][:, year]
            data[f"muscle_mass_factor_true_{year}"] = weight_data["muscle"][:, year]
            data[f"bmr_kcal_true_{year}"] = weight_data["bmr"][:, year]
            data[f"tdee_kcal_true_{year}"] = weight_data["tdee"][:, year]

            # Lifestyle factors
            for key in ["alcohol", "cardio", "strength", "smoking", "stress", "sleep"]:
                series = trajectories[key]
                if key == "alcohol":
                    data[f"alcohol_g_per_week_{year}"] = series[:, year]
                elif key == "cardio":
                    data[f"cardio_met_minutes_{year}"] = series[:, year]
                elif key == "strength":
                    data[f"strength_met_minutes_{year}"] = series[:, year]
                elif key == "smoking":
                    data[f"cigarettes_per_day_{year}"] = series[:, year]
                    data[f"cumulative_smoking_{year}"] = weight_data["cumulative_smoking"][:, year]
                elif key == "stress":
                    data[f"stress_level_{year}"] = series[:, year]
                elif key == "sleep":
                    data[f"sleep_hours_{year}"] = series[:, year]

            # Nutrition
            data[f"protein_pct_{year}"] = weight_data["nutrition"]["protein"][:, year]
            data[f"fat_pct_{year}"] = weight_data["nutrition"]["fat"][:, year]
            data[f"carb_pct_{year}"] = weight_data["nutrition"]["carb"][:, year]
            data[f"saturated_fat_pct_{year}"] = weight_data["nutrition"]["sat_fat"][:, year]
            data[f"simple_carbs_pct_{year}"] = weight_data["nutrition"]["simple_carbs"][:, year]
            data[f"fiber_g_day_{year}"] = weight_data["nutrition"]["fiber"][:, year]
            data[f"sodium_mg_day_{year}"] = weight_data["nutrition"]["sodium"][:, year]
            data[f"macro_imbalance_{year}"] = weight_data["nutrition"]["macro_imb"][:, year]
            data[f"simple_carbs_g_{year}"] = weight_data["nutrition"]["simple_carbs_g"][:, year]

            # ✅ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: total_met_minutes
            data[f"total_met_minutes_{year}"] = (
                    trajectories["cardio"][:, year] + trajectories["strength"][:, year]
            )

        return pd.DataFrame(data)

    def _generate_ou_baselines(self, n: int) -> Dict[str, np.ndarray]:
        """Generate baseline levels for OU processes."""
        return {
            "alcohol": np.where(self.rng.random(n) > 0.3, self.rng.gamma(2, 20, n), 0),
            "cardio": self.rng.uniform(300, 1200, n),
            "strength": self.rng.uniform(0, 600, n),
            "smoking": np.where(self.rng.random(n) < 0.25, self.rng.exponential(3, n), 0),
            "stress": self.rng.normal(3, 2, n),
            "sleep": self.rng.normal(7, 1, n),
        }

    def _generate_ou_trajectories(
            self,
            baselines: Dict[str, np.ndarray],
            years: int,
            n: int,
    ) -> Dict[str, np.ndarray]:
        """Generate OU process trajectories."""
        trajectories = {
            key: np.zeros((n, years))
            for key in self.ou_params.keys()
        }

        for key in self.ou_params.keys():
            trajectories[key][:, 0] = baselines[key]

        for t in range(1, years):
            for key, params in self.ou_params.items():
                theta = params["theta"]
                sigma = params["sigma"]
                min_val = params["min"]
                max_val = params["max"]

                prev = trajectories[key][:, t - 1]
                baseline = baselines[key]

                new_val = prev + theta * (baseline - prev) + sigma * self.rng.normal(size=n)
                trajectories[key][:, t] = np.clip(new_val, min_val, max_val)

        return trajectories

    def _apply_stress_events(
            self,
            trajectories: Dict[str, np.ndarray],
            years: int
    ) -> Dict[str, np.ndarray]:
        """Apply random stress events with decay."""
        n = trajectories["stress"].shape[0]

        stress_boost = np.zeros(years)

        sleep_drop = np.zeros(years)
        alcohol_boost = np.zeros(years)

        event_years = [y for y in range(years) if self.rng.random() < 0.4]

        for ey in event_years:
            stress_boost[ey] += 3
            if ey + 1 < years:
                stress_boost[ey + 1] += 1.5
            sleep_drop[ey] += 1
            alcohol_boost[ey] += 0.2

        trajectories["stress"] += stress_boost
        trajectories["sleep"] -= sleep_drop
        trajectories["alcohol"] *= 1 + alcohol_boost

        trajectories["stress"] = np.clip(trajectories["stress"], 0, 10)
        trajectories["sleep"] = np.clip(trajectories["sleep"], 4, 10)

        return trajectories

    def _simulate_weight_evolution(
            self,
            cohort_df: pd.DataFrame,
            trajectories: Dict[str, np.ndarray],
            years: int,
    ) -> Dict:
        """Simulate weight, muscle, and nutrition evolution with enhanced mean reversion."""
        n = len(cohort_df)

        age_start = cohort_df["age_start"].values
        sex = cohort_df["sex"].values
        height = cohort_df["height_cm"].values
        height_m = height / 100.0
        height_sq = height_m ** 2  # ← нужно для BMI вычислений

        weight_true = np.zeros((n, years))
        muscle_true = np.zeros((n, years))
        bmr_true = np.zeros((n, years))
        tdee_true = np.zeros((n, years))
        cumulative_smoking = np.zeros((n, years))

        nutrition = {
            "protein": np.zeros((n, years)),
            "fat": np.zeros((n, years)),
            "carb": np.zeros((n, years)),
            "sat_fat": np.zeros((n, years)),
            "simple_carbs": np.zeros((n, years)),
            "fiber": np.zeros((n, years)),
            "sodium": np.zeros((n, years)),
            "macro_imb": np.zeros((n, years)),
            "simple_carbs_g": np.zeros((n, years)),
        }

        weight_true[:, 0] = cohort_df["weight_start_kg"].values
        muscle_true[:, 0] = cohort_df["muscle_mass_start"].values

        # Year 0 BMR/TDEE
        for i in range(n):
            muscle_kg = muscle_true[i, 0] * self.mm_to_weight_kg
            bmr_true[i, 0] = calculate_metabolic_rate(
                weight_true[i, 0], height[i], age_start[i], sex[i]
            )
            bmr_true[i, 0] += self.bmr_correction * (
                    muscle_kg - self.typical_ratio * weight_true[i, 0]
            )
            tdee_true[i, 0] = calculate_tdee(
                bmr_true[i, 0],
                trajectories["cardio"][i, 0] + trajectories["strength"][i, 0]
            )

        # Year 0 nutrition
        nutrition["protein"][:, 0] = self.macro_norms["protein"] + self.rng.normal(0, 0.03, n)
        nutrition["fat"][:, 0] = self.macro_norms["fat"] + self.rng.normal(0, 0.04, n)
        nutrition["carb"][:, 0] = 1 - nutrition["protein"][:, 0] - nutrition["fat"][:, 0]

        total = nutrition["protein"][:, 0] + nutrition["fat"][:, 0] + nutrition["carb"][:, 0]
        nutrition["protein"][:, 0] /= total
        nutrition["fat"][:, 0] /= total
        nutrition["carb"][:, 0] /= total

        nutrition["sat_fat"][:, 0] = nutrition["fat"][:, 0] * self.rng.uniform(0.2, 0.5, n)
        nutrition["simple_carbs"][:, 0] = nutrition["carb"][:, 0] * self.rng.uniform(0.1, 0.4, n)
        nutrition["fiber"][:, 0] = np.clip(self.rng.normal(25, 8, n), 5, 50)
        nutrition["sodium"][:, 0] = np.clip(self.rng.normal(3400, 800, n), 1000, 6000)

        delta_p = np.abs(nutrition["protein"][:, 0] - 0.30) * 100
        delta_f = np.abs(nutrition["fat"][:, 0] - 0.30) * 100
        delta_c = np.abs(nutrition["carb"][:, 0] - 0.40) * 100
        nutrition["macro_imb"][:, 0] = np.sqrt(delta_p ** 2 + delta_f ** 2 + delta_c ** 2)

        cumulative_smoking[:, 0] = trajectories["smoking"][:, 0] / 20

        # Subsequent years
        for t in range(1, years):
            age = age_start + t

            # Текущий BMI для target calculation
            bmi_current = weight_true[:, t - 1] / height_sq

            # Целевой BMI с возрастным трендом (плавный рост ~0.1 кг/м²/год после 30)
            target_bmi = 23 + 0.1 * np.maximum(0, age - 30)
            target_weight = target_bmi * height_sq

            # Случайный дрифт + mean reversion к целевому весу
            weight_drift = self.rng.normal(0, 0.5, n)
            reversion_force = (target_weight - weight_true[:, t - 1]) * 0.025  # коэффициент реверсии

            # Аппетит-регуляция: снижение потребления при высоком BMI
            appetite_suppression = np.maximum(0, bmi_current - 27) * 0.3  # кг/год торможения

            weight_change = weight_drift + reversion_force - appetite_suppression

            weight_true[:, t] = weight_true[:, t - 1] + weight_change
            weight_true[:, t] = clip_biomarker(
                weight_true[:, t], "weight_kg", self.biomarker_ranges
            )

            # Muscle drift
            muscle_drift = self.rng.normal(0, 0.1, n)
            muscle_true[:, t] = muscle_true[:, t - 1] + muscle_drift
            muscle_true[:, t] = clip_biomarker(
                muscle_true[:, t], "muscle_mass_factor", self.biomarker_ranges
            )

            # BMR/TDEE for year t
            for i in range(n):
                muscle_kg = muscle_true[i, t] * self.mm_to_weight_kg
                bmr_true[i, t] = calculate_metabolic_rate(
                    weight_true[i, t], height[i], age[i], sex[i]
                )
                bmr_true[i, t] += self.bmr_correction * (
                        muscle_kg - self.typical_ratio * weight_true[i, t]
                )
                tdee_true[i, t] = calculate_tdee(
                    bmr_true[i, t],
                    trajectories["cardio"][i, t] + trajectories["strength"][i, t]
                )

            # Cumulative smoking
            cumulative_smoking[:, t] = cumulative_smoking[:, t - 1] + trajectories["smoking"][:, t] / 20

            # Nutrition
            for key in ["protein", "fat", "carb"]:
                nutrition[key][:, t] = nutrition[key][:, t - 1] + self.rng.normal(0, 0.01, n)

            total = nutrition["protein"][:, t] + nutrition["fat"][:, t] + nutrition["carb"][:, t]
            nutrition["protein"][:, t] /= total
            nutrition["fat"][:, t] /= total
            nutrition["carb"][:, t] /= total

            nutrition["sat_fat"][:, t] = nutrition["fat"][:, t] * self.rng.uniform(0.2, 0.5, n)
            nutrition["simple_carbs"][:, t] = nutrition["carb"][:, t] * self.rng.uniform(0.1, 0.4, n)
            nutrition["fiber"][:, t] = np.clip(self.rng.normal(25, 8, n), 5, 50)
            nutrition["sodium"][:, t] = np.clip(self.rng.normal(3400, 800, n), 1000, 6000)

            delta_p = np.abs(nutrition["protein"][:, t] - 0.30) * 100
            delta_f = np.abs(nutrition["fat"][:, t] - 0.30) * 100
            delta_c = np.abs(nutrition["carb"][:, t] - 0.40) * 100
            nutrition["macro_imb"][:, t] = np.sqrt(delta_p ** 2 + delta_f ** 2 + delta_c ** 2)

            # Simple carbs grams
            non_alcohol_cal = tdee_true[:, t]
            nutrition["simple_carbs_g"][:, t] = (
                    nutrition["simple_carbs"][:, t] / 100 * non_alcohol_cal / 4
            )

        return {
            "weight": weight_true,
            "muscle": muscle_true,
            "bmr": bmr_true,
            "tdee": tdee_true,
            "cumulative_smoking": cumulative_smoking,
            "nutrition": nutrition,
        }
