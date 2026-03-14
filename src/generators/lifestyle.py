"""
Lifestyle trajectory simulation module.
Generates 20-year lifestyle histories using OU processes.
"""
import logging
from typing import Optional, Dict

import numpy as np
import pandas as pd

from src.coefficients import DEFAULT_COEFFICIENTS, ScientificCoefficients
from src.config import SimulationConfig
from src.utils.helpers import clip_biomarker, calculate_metabolic_rate, calculate_tdee

logger = logging.getLogger(__name__)


class LifestyleSimulator:
    def __init__(self, config: SimulationConfig, rng: np.random.Generator,
                 coeff: ScientificCoefficients = DEFAULT_COEFFICIENTS) -> None:
        self.config = config
        self.rng = rng
        self.coeff = coeff
        self.ou_params = config.ou_params
        self.muscle_coeff = coeff.muscle
        self.lifestyle_coeff = coeff.lifestyle
        self.bmr_correction = config.muscle_params["bmr_correction"]
        self.typical_ratio = config.muscle_params["typical_ratio"]
        self.mm_to_weight_kg = config.muscle_params["mm_to_weight_kg"]
        self.base_calories = config.base_calories
        self.macro_norms = config.macro_norms
        self.biomarker_ranges = config.biomarker_ranges

    def simulate(self, cohort_df: pd.DataFrame, years: Optional[int] = None) -> pd.DataFrame:
        years = years or self.config.years
        n_people = len(cohort_df)
        age_start = cohort_df["age_start"].values
        sex = cohort_df["sex"].values
        height = cohort_df["height_cm"].values
        weight_start = cohort_df["weight_start_kg"].values
        muscle_start = cohort_df["muscle_mass_start"].values
        baselines = self._generate_ou_baselines(n_people)
        trajectories = self._generate_ou_trajectories(baselines, years, n_people)
        trajectories = self._apply_stress_events(trajectories, years)
        weight_data = self._simulate_weight_evolution(age_start, sex, height, weight_start, muscle_start, trajectories,
                                                      years)
        data = {"person_id": cohort_df["person_id"].values}
        self._populate_lifestyle_data(data, trajectories, weight_data, age_start, years)
        return pd.DataFrame(data)

    def _generate_ou_baselines(self, n: int) -> Dict[str, np.ndarray]:
        lc = self.lifestyle_coeff
        return {
            "alcohol": np.where(self.rng.random(n) > lc.ALCOHOL_NON_DRINKER_PROB, self.rng.gamma(2, 20, n), 0),
            "cardio": self.rng.uniform(300, 1200, n),
            "strength": self.rng.uniform(0, 600, n),
            "smoking": np.where(self.rng.random(n) < lc.SMOKING_PROB, self.rng.exponential(3, n), 0),
            "stress": self.rng.normal(3, 2, n),
            "sleep": self.rng.normal(7, 1, n),
        }

    def _generate_ou_trajectories(self, baselines: Dict[str, np.ndarray], years: int, n: int) -> Dict[str, np.ndarray]:
        factors = list(self.ou_params.keys())
        n_factors = len(factors)

        # Build correlation matrix from config
        corr = np.eye(n_factors)
        ou_corr_map = self.lifestyle_coeff.OU_CORRELATION_MATRIX
        factor_idx = {f: i for i, f in enumerate(factors)}

        for f1, correlations in ou_corr_map.items():
            if f1 not in factor_idx: continue
            idx1 = factor_idx[f1]
            for f2, val in correlations.items():
                if f2 not in factor_idx: continue
                idx2 = factor_idx[f2]
                corr[idx1, idx2] = val
                corr[idx2, idx1] = val

        # Ensure positive definiteness
        eigvals = np.linalg.eigvals(corr)
        if np.any(eigvals <= 0): corr += np.eye(n_factors) * 1e-6

        L = np.linalg.cholesky(corr)
        trajectories = {key: np.zeros((n, years)) for key in factors}
        for key in factors: trajectories[key][:, 0] = baselines[key]

        for t in range(1, years):
            z = self.rng.normal(0, 1, (n, n_factors))
            correlated_noise = z @ L.T
            for idx, key in enumerate(factors):
                params = self.ou_params[key]
                prev = trajectories[key][:, t - 1]
                baseline = baselines[key]
                new_val = prev + params["theta"] * (baseline - prev) + params["sigma"] * correlated_noise[:, idx]
                trajectories[key][:, t] = np.clip(new_val, params["min"], params["max"])
        return trajectories

    def _apply_stress_events(self, trajectories: Dict[str, np.ndarray], years: int) -> Dict[str, np.ndarray]:
        lc = self.lifestyle_coeff
        n = trajectories["stress"].shape[0]
        event_years = [y for y in range(years) if self.rng.random() < lc.STRESS_EVENT_PROB]
        stress_boost = np.zeros(years)
        sleep_drop = np.zeros(years)
        alcohol_boost = np.zeros(years)
        for ey in event_years:
            stress_boost[ey] += lc.STRESS_BOOST_MAIN
            if ey + 1 < years: stress_boost[ey + 1] += lc.STRESS_BOOST_DECAY
            sleep_drop[ey] += lc.SLEEP_DROP
            alcohol_boost[ey] += lc.ALCOHOL_BOOST_FACTOR

        trajectories["stress"] += stress_boost
        trajectories["sleep"] -= sleep_drop
        trajectories["alcohol"] *= 1 + alcohol_boost
        trajectories["stress"] = np.clip(trajectories["stress"], 0, 10)
        trajectories["sleep"] = np.clip(trajectories["sleep"], 4, 10)
        return trajectories

    def _simulate_weight_evolution(self, age_start, sex, height, weight_start, muscle_start, trajectories, years):
        lc = self.lifestyle_coeff
        n = len(age_start)
        height_sq = (height / 100.0) ** 2
        weight_true = np.zeros((n, years))
        muscle_true = np.zeros((n, years))
        bmr_true = np.zeros((n, years))
        tdee_true = np.zeros((n, years))
        cumulative_smoking = np.zeros((n, years))
        nutrition = self._init_nutrition(n, years)

        weight_true[:, 0] = weight_start
        muscle_true[:, 0] = muscle_start
        self._update_energy_nutrition(0, weight_true, muscle_true, bmr_true, tdee_true, cumulative_smoking, nutrition,
                                      age_start, sex, height, trajectories, height_sq)

        for t in range(1, years):
            age = age_start + t
            bmi_current = weight_true[:, t - 1] / height_sq
            target_weight = (27 + 0.05 * np.maximum(0, age - 30)) * height_sq
            weight_drift = self.rng.normal(lc.WEIGHT_DRIFT_MEAN, lc.WEIGHT_DRIFT_STD, n)
            reversion_force = (target_weight - weight_true[:, t - 1]) * lc.REVERSION_COEF
            appetite_suppression = np.maximum(0,
                                              bmi_current - lc.APPETITE_SUPPRESSION_BMI_THRESHOLD) * lc.APPETITE_SUPPRESSION_COEF
            weight_change = weight_drift + reversion_force - appetite_suppression
            weight_true[:, t] = clip_biomarker(weight_true[:, t - 1] + weight_change, "weight_kg",
                                               self.biomarker_ranges)
            muscle_drift = self.rng.normal(lc.MUSCLE_DRIFT_MEAN, lc.MUSCLE_DRIFT_STD, n)
            muscle_true[:, t] = clip_biomarker(muscle_true[:, t - 1] + muscle_drift, "muscle_mass_factor",
                                               self.biomarker_ranges)
            self._update_energy_nutrition(t, weight_true, muscle_true, bmr_true, tdee_true, cumulative_smoking,
                                          nutrition, age, sex, height, trajectories, height_sq)

        return {"weight": weight_true, "muscle": muscle_true, "bmr": bmr_true, "tdee": tdee_true,
                "cumulative_smoking": cumulative_smoking, "nutrition": nutrition}

    def _init_nutrition(self, n, years):
        return {k: np.zeros((n, years)) for k in
                ["protein", "fat", "carb", "sat_fat", "simple_carbs", "fiber", "sodium", "macro_imb", "simple_carbs_g"]}

    def _update_energy_nutrition(self, t, weight, muscle, bmr, tdee, cum_smoke, nut, age, sex, height, traj, height_sq):
        n = len(age)
        muscle_kg = muscle[:, t] * self.mm_to_weight_kg
        bmr[:, t] = calculate_metabolic_rate(weight[:, t], height, age, sex) + self.bmr_correction * (
                    muscle_kg - self.typical_ratio * weight[:, t])
        tdee[:, t] = calculate_tdee(bmr[:, t], traj["cardio"][:, t] + traj["strength"][:, t])

        if t == 0:
            cum_smoke[:, t] = traj["smoking"][:, t] / 20
        else:
            cum_smoke[:, t] = cum_smoke[:, t - 1] + traj["smoking"][:, t] / 20

        if t == 0:
            self._init_nutrition_year0(nut, t, n)
        else:
            self._update_nutrition_year_t(nut, t, n)
        self._compute_macro_imbalance(nut, t, n)
        nut["simple_carbs_g"][:, t] = nut["simple_carbs"][:, t] / 100 * tdee[:, t] / 4

    def _init_nutrition_year0(self, nut, t, n):
        nut["protein"][:, t] = self.macro_norms["protein"] + self.rng.normal(0, 0.03, n)
        nut["fat"][:, t] = self.macro_norms["fat"] + self.rng.normal(0, 0.04, n)
        nut["carb"][:, t] = 1 - nut["protein"][:, t] - nut["fat"][:, t]
        total = nut["protein"][:, t] + nut["fat"][:, t] + nut["carb"][:, t]
        for k in ["protein", "fat", "carb"]: nut[k][:, t] /= total
        nut["sat_fat"][:, t] = nut["fat"][:, t] * self.rng.uniform(0.2, 0.5, n)
        nut["simple_carbs"][:, t] = nut["carb"][:, t] * self.rng.uniform(0.1, 0.4, n)
        nut["fiber"][:, t] = np.clip(self.rng.normal(25, 8, n), 5, 50)
        nut["sodium"][:, t] = np.clip(self.rng.normal(3400, 800, n), 1000, 6000)

    def _update_nutrition_year_t(self, nut, t, n):
        for k in ["protein", "fat", "carb"]: nut[k][:, t] = nut[k][:, t - 1] + self.rng.normal(0, 0.01, n)
        total = nut["protein"][:, t] + nut["fat"][:, t] + nut["carb"][:, t]
        for k in ["protein", "fat", "carb"]: nut[k][:, t] /= total
        nut["sat_fat"][:, t] = nut["fat"][:, t] * self.rng.uniform(0.2, 0.5, n)
        nut["simple_carbs"][:, t] = nut["carb"][:, t] * self.rng.uniform(0.1, 0.4, n)
        nut["fiber"][:, t] = np.clip(self.rng.normal(25, 8, n), 5, 50)
        nut["sodium"][:, t] = np.clip(self.rng.normal(3400, 800, n), 1000, 6000)

    def _compute_macro_imbalance(self, nut, t, n):
        lc = self.lifestyle_coeff
        dp = np.abs(nut["protein"][:, t] - lc.PROTEIN_TARGET) * 100
        df = np.abs(nut["fat"][:, t] - lc.FAT_TARGET) * 100
        dc = np.abs(nut["carb"][:, t] - lc.CARB_TARGET) * 100
        nut["macro_imb"][:, t] = np.sqrt(dp ** 2 + df ** 2 + dc ** 2)

    def _populate_lifestyle_data(self, data, trajectories, weight_data, age_start, years):
        for y in range(years):
            data[f"age_{y}"] = age_start + y
            data[f"weight_kg_true_{y}"] = weight_data["weight"][:, y]
            data[f"muscle_mass_factor_true_{y}"] = weight_data["muscle"][:, y]
            data[f"bmr_kcal_true_{y}"] = weight_data["bmr"][:, y]
            data[f"tdee_kcal_true_{y}"] = weight_data["tdee"][:, y]
            for key, col_name in [("alcohol", "alcohol_g_per_week"), ("cardio", "cardio_met_minutes"),
                                  ("strength", "strength_met_minutes"), ("stress", "stress_level"),
                                  ("sleep", "sleep_hours")]:
                data[f"{col_name}_{y}"] = trajectories[key][:, y]
            data[f"cigarettes_per_day_{y}"] = trajectories["smoking"][:, y]
            data[f"cumulative_smoking_{y}"] = weight_data["cumulative_smoking"][:, y]
            nut = weight_data["nutrition"]
            data[f"protein_pct_{y}"] = nut["protein"][:, y]
            data[f"fat_pct_{y}"] = nut["fat"][:, y]
            data[f"carb_pct_{y}"] = nut["carb"][:, y]
            data[f"saturated_fat_pct_{y}"] = nut["sat_fat"][:, y]
            data[f"simple_carbs_pct_{y}"] = nut["simple_carbs"][:, y]
            data[f"fiber_g_day_{y}"] = nut["fiber"][:, y]
            data[f"sodium_mg_day_{y}"] = nut["sodium"][:, y]
            data[f"macro_imbalance_{y}"] = nut["macro_imb"][:, y]
            data[f"simple_carbs_g_{y}"] = nut["simple_carbs_g"][:, y]
            data[f"total_met_minutes_{y}"] = trajectories["cardio"][:, y] + trajectories["strength"][:, y]
