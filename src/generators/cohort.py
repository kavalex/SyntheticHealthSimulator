"""
Cohort generation module.

Generates baseline cohort with demographics, genetic risks,
and initial biomarkers using coefficients from src.coefficients.
"""

from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd

from src.coefficients import DEFAULT_COEFFICIENTS, ScientificCoefficients
from src.config import SimulationConfig
from src.utils.helpers import generate_truncnorm, generate_genetic_risk


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
            coeff: ScientificCoefficients = DEFAULT_COEFFICIENTS,
    ) -> None:
        """
        Инициализация генератора когорты.

        Args:
            config: Объект конфигурации симуляции.
            rng: Генератор случайных чисел.
            coeff: Научные коэффициенты.
        """
        self.config = config
        self.rng = rng
        self.coeff = coeff

        # Для удобства сохраняем нужные группы коэффициентов
        self.demo = coeff.demo
        self.muscle_coeff = coeff.muscle
        self.genetic = coeff.genetic_risk

        # Импорт параметров из конфига
        self.genetic_std = config.genetic_risk_std
        self.lipid_effect = config.lipid_effect
        self.muscle_params = config.muscle_params

        # Извлечение параметров мышечной массы для BMR
        self.bmr_correction = self.muscle_params["bmr_correction"]
        self.typical_ratio = self.muscle_params["typical_ratio"]
        self.mm_to_weight_kg = self.muscle_params["mm_to_weight_kg"]

    def generate(
            self,
            n_people: int,
            age_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Генерирует базовую когорту с демографией и начальными биомаркерами.

        Args:
            n_people: Количество генерируемых пациентов (должно быть > 0).
            age_range: Опциональный кортеж (min, max) для возраста.
                       Если не указан, используется значение из конфигурации.

        Returns:
            DataFrame с данными когорты (одна строка на пациента).

        Raises:
            ValueError: Если n_people <= 0 или age_range некорректен.
        """
        # Валидация входных данных
        if n_people <= 0:
            raise ValueError(f"n_people must be positive, got {n_people}")
        if age_range is not None and (age_range[0] > age_range[1]):
            raise ValueError(f"Invalid age_range: {age_range}")

        age_range = age_range or self.config.age_range

        # Генерация демографических данных
        age = self._generate_age(n_people, age_range)
        sexes = self._generate_sex(n_people)
        male_mask = sexes == "M"
        female_mask = ~male_mask
        n_male = np.sum(male_mask)
        n_female = n_people - n_male

        height = self._generate_height(n_people, male_mask, female_mask)

        # Генетические риски
        genetic = self._generate_genetic_risks(n_people)

        # Мышечная масса
        muscle = self._generate_muscle_mass(n_male, n_female)

        # Биомаркеры
        hdl = self._generate_hdl(n_male, n_female)
        tc = self._generate_total_cholesterol(n_people, genetic["cvd"])
        sbp = self._generate_sbp(age)
        hba1c = self._generate_hba1c(n_people)

        # Вес и BMI
        bmi_start = self._generate_bmi(n_people)
        height_m = height / 100.0
        weight_start = bmi_start * (height_m ** 2)

        # BMR
        bmr = self._calculate_bmr_vectorized(
            weight_start, height, age, sexes, muscle
        )

        # Липидные факторы
        lipid_hdl, lipid_tc = self._generate_lipid_factors(n_people)

        # Формирование DataFrame
        cohort = pd.DataFrame({
            "person_id": range(1, n_people + 1),
            "age_start": age,
            "sex": sexes,
            "height_cm": height.round(1),
            "weight_start_kg": weight_start.round(1),
            "bmi_start": bmi_start.round(2),
            "muscle_mass_start": muscle.round(2),
            "bmr_start_kcal": bmr.round(0),
            "base_calories": np.where(male_mask,
                                      self.config.base_calories["M"],
                                      self.config.base_calories["F"]),
            "genetic_risk_diabetes": genetic["diabetes"].round(3),
            "genetic_risk_cvd": genetic["cvd"].round(3),
            "genetic_risk_cancer": genetic["cancer"].round(3),
            "genetic_risk_stroke": genetic["stroke"].round(3),
            "genetic_risk_nafld": genetic["nafld"].round(3),
            "genetic_risk_cirrhosis": genetic["cirrhosis"].round(3),
            "initial_hdl_mgdl": hdl.round(1),
            "initial_total_cholesterol_mgdl": tc.round(1),
            "initial_sbp_mmhg": sbp.round(1),
            "initial_hba1c_percent": hba1c.round(2),
            "lipid_factor_hdl": lipid_hdl.round(3),
            "lipid_factor_tc": lipid_tc.round(3),
        })

        return cohort

    # Вспомогательные методы генерации

    def _generate_age(self, n_people: int, age_range: Tuple[int, int]) -> np.ndarray:
        """Генерирует возраст, усечённое нормальное распределение."""
        return generate_truncnorm(
            self.rng,
            self.demo.AGE_MEAN,
            self.demo.AGE_STD,
            age_range[0],
            age_range[1],
            n_people
        ).astype(int)

    def _generate_sex(self, n_people: int) -> np.ndarray:
        """Генерирует пол (50/50)."""
        return self.rng.choice(["M", "F"], n_people)

    def _generate_height(
            self,
            n_people: int,
            male_mask: np.ndarray,
            female_mask: np.ndarray,
    ) -> np.ndarray:
        """Генерирует рост с учётом пола."""
        height = np.zeros(n_people)
        height[male_mask] = self.rng.normal(
            self.demo.HEIGHT_MALE_MEAN,
            self.demo.HEIGHT_MALE_STD,
            np.sum(male_mask)
        )
        height[female_mask] = self.rng.normal(
            self.demo.HEIGHT_FEMALE_MEAN,
            self.demo.HEIGHT_FEMALE_STD,
            np.sum(female_mask)
        )
        return height

    def _generate_genetic_risks(self, n_people: int) -> Dict[str, np.ndarray]:
        """Генерирует все генетические риски."""
        return {
            "diabetes": generate_genetic_risk(
                self.rng, n_people, "diabetes", self.genetic_std
            ),
            "cvd": generate_genetic_risk(
                self.rng, n_people, "cvd", self.genetic_std
            ),
            "cancer": generate_genetic_risk(
                self.rng, n_people, "colorectal", self.genetic_std
            ),
            "stroke": generate_genetic_risk(
                self.rng, n_people, "stroke", self.genetic_std
            ),
            "nafld": generate_genetic_risk(
                self.rng, n_people, "nafld", self.genetic_std
            ),
            "cirrhosis": generate_genetic_risk(
                self.rng, n_people, "cirrhosis", self.genetic_std
            ),
        }

    def _generate_muscle_mass(self, n_male: int, n_female: int) -> np.ndarray:
        """Генерирует мышечную массу с учётом пола."""
        muscle_m = generate_truncnorm(
            self.rng,
            self.muscle_coeff.MUSCLE_MALE_MEAN,
            self.muscle_coeff.MUSCLE_MALE_STD,
            self.muscle_coeff.MUSCLE_MIN,
            self.muscle_coeff.MUSCLE_MAX,
            n_male
        )
        muscle_f = generate_truncnorm(
            self.rng,
            self.muscle_coeff.MUSCLE_FEMALE_MEAN,
            self.muscle_coeff.MUSCLE_FEMALE_STD,
            self.muscle_coeff.MUSCLE_MIN,
            self.muscle_coeff.MUSCLE_MAX,
            n_female
        )
        return np.concatenate([muscle_m, muscle_f])

    def _generate_hdl(self, n_male: int, n_female: int) -> np.ndarray:
        """Генерирует HDL холестерин с учётом пола."""
        hdl_m = generate_truncnorm(
            self.rng,
            self.demo.HDL_MALE_MEAN,
            self.demo.HDL_MALE_STD,
            self.demo.HDL_MIN,
            self.demo.HDL_MAX,
            n_male
        )
        hdl_f = generate_truncnorm(
            self.rng,
            self.demo.HDL_FEMALE_MEAN,
            self.demo.HDL_FEMALE_STD,
            self.demo.HDL_MIN,
            self.demo.HDL_MAX,
            n_female
        )
        return np.concatenate([hdl_m, hdl_f])

    def _generate_total_cholesterol(
            self,
            n_people: int,
            genetic_cvd: np.ndarray,
    ) -> np.ndarray:
        """Генерирует общий холестерин с учётом генетики."""
        tc = generate_truncnorm(
            self.rng,
            self.demo.TC_MEAN,
            self.demo.TC_STD,
            self.demo.TC_MIN,
            self.demo.TC_MAX,
            n_people
        )
        tc += self.demo.TC_GENETIC_FACTOR * genetic_cvd
        return np.clip(tc, self.demo.TC_MIN, self.demo.TC_MAX)

    def _generate_sbp(self, age: np.ndarray) -> np.ndarray:
        """Генерирует систолическое давление с возрастной зависимостью."""
        sbp = self.demo.SBP_INTERCEPT + self.demo.SBP_AGE_SLOPE * (age - 35)
        sbp += self.rng.normal(0, self.demo.SBP_NOISE_STD, len(age))
        return np.clip(sbp, self.demo.SBP_MIN, self.demo.SBP_MAX)

    def _generate_hba1c(self, n_people: int) -> np.ndarray:
        """Генерирует HbA1c."""
        hba1c = generate_truncnorm(
            self.rng,
            self.demo.HBA1C_MEAN,
            self.demo.HBA1C_STD,
            self.demo.HBA1C_GEN_MIN,
            self.demo.HBA1C_GEN_MAX,
            n_people
        )
        # Финальный клиппинг до физиологических границ
        return np.clip(hba1c, self.demo.HBA1C_FINAL_MIN, self.demo.HBA1C_FINAL_MAX)

    def _generate_bmi(self, n_people: int) -> np.ndarray:
        """Генерирует начальный BMI."""
        bmi = generate_truncnorm(self.rng, self.demo.BMI_MEAN, self.demo.BMI_STD, self.demo.BMI_MIN, self.demo.BMI_MAX, n_people)
        return np.clip(bmi, 16, 50)

    def _calculate_bmr_vectorized(
            self,
            weight_kg: np.ndarray,
            height_cm: np.ndarray,
            age: np.ndarray,
            sex: np.ndarray,
            muscle_mass: np.ndarray,
    ) -> np.ndarray:
        """
        Вычисляет BMR с поправкой на мышечную массу (векторизованно).

        Использует формулу Mifflin-St Jeor.
        """
        is_male = sex == "M"
        bmr = np.where(
            is_male,
            10 * weight_kg + 6.25 * height_cm - 5 * age + 5,
            10 * weight_kg + 6.25 * height_cm - 5 * age - 161
        )

        # Поправка на мышечную массу
        muscle_kg = muscle_mass * self.mm_to_weight_kg
        typical_muscle = self.typical_ratio * weight_kg
        correction = self.bmr_correction * (muscle_kg - typical_muscle)

        return bmr + correction

    def _generate_lipid_factors(
            self,
            n: int,
            correlation: float = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерирует коррелированные липидные факторы для HDL и TC.

        Факторы обрезаются до [-LIPID_CLIP, LIPID_CLIP].
        """
        if correlation is None:
            correlation = self.demo.LIPID_CORRELATION
        mean = [0, 0]
        cov = [[1, correlation], [correlation, 1]]
        factors = self.rng.multivariate_normal(mean, cov, n)
        return (
            np.clip(factors[:, 0], -self.demo.LIPID_CLIP, self.demo.LIPID_CLIP),
            np.clip(factors[:, 1], -self.demo.LIPID_CLIP, self.demo.LIPID_CLIP),
        )
