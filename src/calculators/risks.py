"""
Health risk calculation module.
Calculates disease risks and generates binary outcomes.
"""

import warnings
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

from src.coefficients import DEFAULT_COEFFICIENTS, ScientificCoefficients
from src.config import SimulationConfig


class RiskCalculator:
    """
    Calculates medical risks and generates binary outcomes.
    """

    # Константы для преобразования единиц
    HDL_TO_TC_CONVERSION = 38.67  # mg/dL to mmol/L для non-HDL

    # Конфигурация определения причины смерти
    DEATH_CAUSE_CONFIG = {
        "cvd": {"threshold": 0.3, "prob": 0.15, "label": "Cardiovascular disease"},
        "diabetes": {"threshold": 0.25, "prob": 0.08, "label": "Diabetes complications"},
        "stroke": {"threshold": 0.2, "prob": 0.1, "label": "Stroke"},
        "cirrhosis": {"threshold": 0.15, "prob": 0.05, "label": "Liver failure"},
    }

    def __init__(
            self,
            config: SimulationConfig,
            rng: np.random.Generator,
            coeff: ScientificCoefficients = DEFAULT_COEFFICIENTS,
    ) -> None:
        """
        Инициализация калькулятора рисков.

        Args:
            config: Объект конфигурации.
            rng: Генератор случайных чисел.
            coeff: Научные коэффициенты.
        """
        self.config = config
        self.rng = rng
        self.coeff = coeff

        self.diabetes_threshold = config.diabetes_threshold
        self.risk_intercepts = config.risk_intercepts

    def calculate(
            self,
            biomarkers_df: pd.DataFrame,
            cohort_df: pd.DataFrame,
            years: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Вычисляет медицинские риски и генерирует бинарные исходы.

        Args:
            biomarkers_df: DataFrame с годовыми биомаркерами (одна строка на пациента).
            cohort_df: DataFrame с данными когорты (одна строка на пациента).
            years: Количество лет наблюдения (по умолчанию из config).

        Returns:
            DataFrame с рисками, исходами и дополнительной информацией.
        """
        years = years or self.config.years
        last_year = years - 1

        # Извлечение финальных значений за последний год
        final_data = self._extract_final_data(biomarkers_df, last_year)

        # Генетические данные
        genetic_data = self._extract_genetic_data(cohort_df, biomarkers_df["person_id"])

        # Расчёт рисков для каждого заболевания
        risks = {}
        risks["cvd"], _ = self._calculate_cvd_risk(
            final_data["age"], final_data["sbp"], final_data["non_hdl_mmol"],
            final_data["cumulative_smoking"], final_data["hba1c"],
            genetic_data["cvd"]
        )
        risks["diabetes"], _ = self._calculate_diabetes_risk(
            final_data["age"], final_data["bmi_corrected"], final_data["hba1c"],
            final_data["cumulative_smoking"], final_data["fiber"],
            genetic_data["diabetes"], final_data["macro_imbalance"],
            final_data["muscle"], final_data["sleep"]
        )
        risks["stroke"], _ = self._calculate_stroke_risk(
            final_data["age"], final_data["sbp"], final_data["hba1c"],
            final_data["cumulative_smoking"], final_data["bmi_corrected"],
            genetic_data["stroke"]
        )
        risks["nafld"], _ = self._calculate_nafld_risk(
            final_data["age"], final_data["bmi_corrected"], final_data["hba1c"],
            final_data["saturated_fat"], final_data["fiber"],
            genetic_data["nafld"]
        )
        risks["cancer"], _ = self._calculate_cancer_risk(
            final_data["age"], final_data["fiber"], final_data["alcohol"],
            final_data["cumulative_smoking"], final_data["bmi_corrected"],
            genetic_data["cancer"]
        )
        risks["cirrhosis"], _ = self._calculate_cirrhosis_risk(
            final_data["age"], final_data["alcohol"], final_data["cumulative_smoking"],
            final_data["bmi_corrected"], genetic_data["cirrhosis"]
        )

        # Композитный health score
        health_score = self._calculate_health_score(
            risks["cvd"], risks["diabetes"], risks["stroke"],
            risks["nafld"], risks["cancer"], risks["cirrhosis"]
        )

        # Формирование выходного DataFrame
        result_df = pd.DataFrame({"person_id": biomarkers_df["person_id"]})
        result_df["cvd_risk_10year"] = risks["cvd"].clip(0, 0.5)
        result_df["diabetes_risk_10year"] = risks["diabetes"].clip(0, 0.5)
        result_df["stroke_risk_10year"] = risks["stroke"].clip(0, 0.4)
        result_df["nafld_risk_10year"] = risks["nafld"].clip(0, 0.6)
        result_df["colorectal_cancer_risk_10year"] = risks["cancer"].clip(0.05, 0.35)
        result_df["cirrhosis_risk_10year"] = risks["cirrhosis"].clip(0.02, 0.25)
        result_df["health_score"] = health_score

        # Бинарные исходы
        result_df["has_cvd"] = self.rng.binomial(1, risks["cvd"].clip(0, 0.5))
        result_df["has_diabetes"] = self.rng.binomial(1, risks["diabetes"].clip(0, 0.5))
        result_df["has_stroke"] = self.rng.binomial(1, risks["stroke"].clip(0, 0.4))
        result_df["has_nafld"] = self.rng.binomial(1, risks["nafld"].clip(0, 0.6))
        result_df["has_colorectal_cancer"] = self.rng.binomial(1, risks["cancer"].clip(0, 0.3))
        result_df["has_cirrhosis"] = self.rng.binomial(1, risks["cirrhosis"].clip(0, 0.2))

        # Дополнительные колонки
        self._check_class_balance(result_df)
        result_df["primary_death_cause"] = self._determine_death_cause(risks)
        result_df["estimated_event_age"] = 60 - 20 * (1 - health_score / 100)
        result_df["sex"] = genetic_data["sex"]

        return result_df

    def _extract_final_data(self, biomarkers_df: pd.DataFrame, last_year: int) -> Dict[str, np.ndarray]:
        """
        Извлекает финальные значения биомаркеров за последний год.

        Args:
            biomarkers_df: DataFrame с биомаркерами.
            last_year: Индекс последнего года.

        Returns:
            Словарь с ключами:
                age, bmi_corrected, hdl, tc, sbp, hba1c, alcohol, fiber,
                saturated_fat, macro_imbalance, muscle, sleep, cumulative_smoking,
                non_hdl_mgdl, non_hdl_mmol.
        """
        data = {
            "age": biomarkers_df[f"age_{last_year}"].values,
            "bmi_corrected": biomarkers_df[f"bmi_corrected_{last_year}"].values,
            "hdl": biomarkers_df[f"hdl_mgdl_{last_year}"].values,
            "tc": biomarkers_df[f"total_cholesterol_mgdl_{last_year}"].values,
            "sbp": biomarkers_df[f"sbp_mmhg_{last_year}"].values,
            "hba1c": biomarkers_df[f"hba1c_percent_{last_year}"].values,
            "alcohol": biomarkers_df[f"alcohol_g_per_week_{last_year}"].values,
            "fiber": biomarkers_df[f"fiber_g_day_{last_year}"].values,
            "saturated_fat": biomarkers_df[f"saturated_fat_pct_{last_year}"].values,
            "macro_imbalance": biomarkers_df[f"macro_imbalance_{last_year}"].values,
            "muscle": biomarkers_df[f"muscle_mass_factor_{last_year}"].values,
            "sleep": biomarkers_df[f"sleep_hours_{last_year}"].values,
            "cumulative_smoking": biomarkers_df[f"cumulative_smoking_{last_year}"].values,
        }
        # Вычисляем non-HDL
        data["non_hdl_mgdl"] = data["tc"] - data["hdl"]
        data["non_hdl_mmol"] = data["non_hdl_mgdl"] / self.HDL_TO_TC_CONVERSION
        return data

    def _extract_genetic_data(self, cohort_df: pd.DataFrame, person_ids: pd.Series) -> Dict[str, np.ndarray]:
        """
        Извлекает генетические данные для заданных person_id.

        Args:
            cohort_df: DataFrame когорты.
            person_ids: Серия с идентификаторами пациентов.

        Returns:
            Словарь с ключами: cvd, diabetes, stroke, nafld, cancer, cirrhosis, sex.
        """
        genetic = cohort_df.set_index("person_id").reindex(person_ids)
        return {
            "cvd": genetic["genetic_risk_cvd"].values,
            "diabetes": genetic["genetic_risk_diabetes"].values,
            "stroke": genetic["genetic_risk_stroke"].values,
            "nafld": genetic["genetic_risk_nafld"].values,
            "cancer": genetic["genetic_risk_cancer"].values,
            "cirrhosis": genetic["genetic_risk_cirrhosis"].values,
            "sex": genetic["sex"].values,
        }

    # Методы расчёта рисков

    def _calculate_cvd_risk(
            self,
            age: np.ndarray,
            sbp: np.ndarray,
            non_hdl_mmol: np.ndarray,
            pack_years: np.ndarray,
            hba1c: np.ndarray,
            genetic_cvd: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CVD risk calculation (SCORE2-like).

        Returns:
            Tuple (risk, logit).
        """
        logit = (
                self.risk_intercepts["cvd"]
                + 0.05 * (age - 40)
                + 0.04 * (sbp - 120)
                + 0.3 * (non_hdl_mmol - 3.5)
                + 0.08 * pack_years
                + 0.8 * (hba1c >= 6.5).astype(float)
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
        """
        Diabetes risk calculation (FINDRISC-like).

        Returns:
            Tuple (risk, points).
        """
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
                + 0.05 * (age - 50)
                + 0.03 * (sbp - 120)
                + 0.4 * (hba1c - 5.5)
                + 0.2 * smoking_binary
                + 0.05 * (bmi_corrected - 25)
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

    # Вспомогательные методы

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
        for disease in ["cvd", "diabetes", "stroke"]:
            col = f"has_{disease}"
            if col in risks_df.columns:
                prevalence = risks_df[col].mean()
                if prevalence < 0.10 or prevalence > 0.50:
                    warnings.warn(
                        f"Class imbalance detected for {disease}: {prevalence:.2%}. "
                        f"Consider adjusting risk thresholds."
                    )

    def _determine_death_cause(self, risks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Determine primary death cause based on risks.

        Args:
            risks: Словарь с ключами 'cvd', 'diabetes', 'stroke', 'cirrhosis'
                   и значениями – массивами рисков.

        Returns:
            Массив строк с причинами смерти.
        """
        n = len(risks["cvd"])
        conditions = []
        choices = []
        for key, cfg in self.DEATH_CAUSE_CONFIG.items():
            if key in risks:
                conditions.append(
                    (risks[key] > cfg["threshold"]) & (self.rng.random(n) < cfg["prob"])
                )
                choices.append(cfg["label"])
        return np.select(conditions, choices, default="Other")
