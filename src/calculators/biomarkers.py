"""
Biomarker calculation module - VECTORIZED VERSION.
Calculates annual biomarker measurements from lifestyle data using NumPy broadcasting.
Uses scientific coefficients from coefficients.py.
"""

import logging
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

from src.coefficients import DEFAULT_COEFFICIENTS, ScientificCoefficients
from src.config import SimulationConfig
from src.utils.helpers import clip_biomarker

logger = logging.getLogger(__name__)


class BiomarkerCalculator:
    """
    Calculates biomarkers from lifestyle trajectories using fully vectorized operations.

    Attributes:
        config (SimulationConfig): Основная конфигурация симуляции.
        rng (np.random.Generator): Генератор случайных чисел.
        coeff (ScientificCoefficients): Научные коэффициенты для расчётов.
        verbose (bool): Флаг для вывода отладочной информации (логирование уровня DEBUG).
    """

    # Маппинг типов маркеров для функции обрезки
    MARKER_MAP = {
        "hdl": "hdl_mgdl",
        "tc": "total_cholesterol_mgdl",
        "sbp": "sbp_mmhg",
        "hba1c": "hba1c_percent",
        "weight": "weight_kg",
        "bmi": "bmi",
        "non_hdl_mgdl": "non_hdl_mgdl",
        "non_hdl_mmol": "non_hdl_mmol",
    }

    def __init__(
            self,
            config: SimulationConfig,
            rng: np.random.Generator,
            coefficients: Optional[ScientificCoefficients] = None,
            verbose: bool = False,
    ) -> None:
        """
        Инициализация калькулятора биомаркеров.

        Args:
            config: Конфигурация симуляции.
            rng: Генератор случайных чисел.
            coefficients: Научные коэффициенты (если не указаны, используются DEFAULT_COEFFICIENTS).
            verbose: Включать ли отладочное логирование.
        """
        self.config = config
        self.rng = rng
        self.coeff = coefficients or DEFAULT_COEFFICIENTS
        self.verbose = verbose
        self.lipid_effect = config.lipid_effect
        self.measurement_noise = config.measurement_noise
        self.biomarker_ranges = config.biomarker_ranges
        self.final_biomarker_ranges = config.final_biomarker_ranges

        # Предвычисленные константы (будут заполнены при первом вызове calculate)
        self._height_sq: Optional[np.ndarray] = None
        self._genetic_hdl_offset: Optional[np.ndarray] = None
        self._genetic_tc_offset: Optional[np.ndarray] = None

    def calculate(
            self,
            cohort_df: pd.DataFrame,
            lifestyle_df: pd.DataFrame,
            years: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Основной метод расчёта биомаркеров.

        Args:
            cohort_df: DataFrame с данными когорты (один ряд на пациента).
            lifestyle_df: DataFrame с годовыми данными образа жизни (один ряд на пациента, колонки для каждого года).
            years: Количество лет для расчёта (по умолчанию из config).

        Returns:
            DataFrame с годовыми значениями биомаркеров (с шумом) и дополнительными колонками.
        """
        # 1. Валидация входных данных
        self._validate_inputs(cohort_df, lifestyle_df)

        years = years or self.config.years
        n_people = len(lifestyle_df)
        person_ids = lifestyle_df["person_id"].values
        cohort_by_id = cohort_df.set_index("person_id")

        # 2. Извлечение базовых данных
        baseline = self._extract_baseline_data(cohort_by_id, person_ids)
        height, initial_hdl, initial_tc, lipid_factor_hdl, lipid_factor_tc, initial_hba1c, initial_sbp, sex = baseline

        # 3. Предвычисление констант
        self._precompute_constants(height, lipid_factor_hdl, lipid_factor_tc)

        # 4. Извлечение временных рядов образа жизни
        lifestyle_arrays = self._extract_lifestyle_arrays(lifestyle_df, years)

        # Распаковка для удобства
        weight_true = lifestyle_arrays["weight_true"]
        muscle_true = lifestyle_arrays["muscle_true"]
        age_vals = lifestyle_arrays["age_vals"]
        saturated_fat = lifestyle_arrays["saturated_fat"]
        fiber = lifestyle_arrays["fiber"]
        sodium = lifestyle_arrays["sodium"]
        stress = lifestyle_arrays["stress"]
        simple_carbs_g = lifestyle_arrays["simple_carbs_g"]
        cardio_met = lifestyle_arrays["cardio_met"]
        strength_met = lifestyle_arrays["strength_met"]
        alcohol = lifestyle_arrays["alcohol"]
        sleep = lifestyle_arrays["sleep"]
        macro_imb = lifestyle_arrays["macro_imb"]

        # 5. Вычисление BMI и дельт
        bmi = weight_true / self._height_sq
        deltas = self._compute_deltas(
            saturated_fat, fiber, sodium, stress, simple_carbs_g, bmi
        )
        delta_sat_fat, delta_fiber, delta_sodium, delta_stress, delta_simple_carbs, delta_bmi = deltas

        # 6. Инициализация массивов истинных биомаркеров
        hdl_true = np.zeros((n_people, years))
        tc_true = np.zeros((n_people, years))
        sbp_true = np.zeros((n_people, years))
        hba1c_true = np.zeros((n_people, years))

        # 7. Год 0: начальные значения с учётом генетики
        hdl_baseline = initial_hdl + self._genetic_hdl_offset.flatten()
        tc_baseline = initial_tc + self._genetic_tc_offset.flatten()
        hdl_true[:, 0] = self._clip_biomarker(hdl_baseline, "hdl")
        tc_true[:, 0] = self._clip_biomarker(tc_baseline, "tc")
        sbp_true[:, 0] = self._clip_biomarker(initial_sbp, "sbp")
        hba1c_true[:, 0] = self._clip_biomarker(initial_hba1c, "hba1c")

        # 8. Предвычисление возрастных норм (кэширование)
        age_norm_sbp_all = self._precompute_age_norm_sbp(age_vals)
        age_norm_hba1c_all = self._precompute_age_norm_hba1c(age_vals)

        # 9. Цикл по годам с обновлением биомаркеров
        for t in range(1, years):
            # Получение срезов для текущего года и предыдущего
            curr = slice(t, t + 1)
            prev = slice(t - 1, t)

            # HDL
            hdl_change = self._calc_hdl_change(
                cardio_met[:, curr], strength_met[:, curr],
                delta_sat_fat[:, prev], delta_fiber[:, prev],
                muscle_true[:, curr], bmi[:, curr], age_vals[:, curr], sex
            )
            hdl_true[:, curr] = self._update_hdl(
                hdl_true[:, prev], hdl_change, self._genetic_hdl_offset
            )

            # Total cholesterol
            tc_change = self._calc_tc_change(
                delta_sat_fat[:, prev], delta_fiber[:, prev],
                cardio_met[:, curr] + strength_met[:, curr]
            )
            tc_true[:, curr] = self._update_tc(
                tc_true[:, prev], tc_change, self._genetic_tc_offset
            )

            # SBP
            sbp_change = self._calc_sbp_change(
                delta_bmi[:, prev], bmi[:, curr],
                alcohol[:, curr] / 7.0, delta_sodium[:, prev], delta_stress[:, prev],
                sleep[:, curr], age_vals[:, curr], sbp_true[:, prev],
                age_norm_sbp_all[:, curr]
            )
            sbp_true[:, curr] = self._update_sbp(sbp_true[:, prev], sbp_change)

            # HbA1c
            hba1c_change = self._calc_hba1c_change(
                delta_simple_carbs[:, prev], delta_bmi[:, prev], delta_fiber[:, prev],
                cardio_met[:, curr] + strength_met[:, curr], macro_imb[:, curr],
                muscle_true[:, curr], sleep[:, curr], age_vals[:, curr],
                hba1c_true[:, prev], age_norm_hba1c_all[:, curr]
            )
            hba1c_true[:, curr] = self._update_hba1c(hba1c_true[:, prev], hba1c_change)

        # 10. Логирование итоговой статистики (если включено)
        self._log_final_stats(hba1c_true, sbp_true, years)

        # 11. Построение выходного DataFrame с шумом
        return self._build_output_dataframe(
            person_ids, years, n_people,
            weight_true, hdl_true, tc_true, sbp_true, hba1c_true,
            muscle_true, bmi, self._height_sq.flatten(),
            lifestyle_df, age_vals
        )

    # Вспомогательные методы для валидации и извлечения данных

    def _validate_inputs(self, cohort_df: pd.DataFrame, lifestyle_df: pd.DataFrame) -> None:
        """Проверяет наличие обязательных колонок во входных DataFrame."""
        required_cohort = [
            "person_id", "height_cm", "initial_hdl_mgdl", "initial_total_cholesterol_mgdl",
            "lipid_factor_hdl", "lipid_factor_tc", "initial_hba1c_percent", "initial_sbp_mmhg", "sex"
        ]
        missing_cohort = [col for col in required_cohort if col not in cohort_df.columns]
        if missing_cohort:
            raise ValueError(f"Missing columns in cohort_df: {missing_cohort}")

        required_lifestyle = ["person_id"] + [
            f"{prefix}_{y}" for prefix in
            ["weight_kg_true", "muscle_mass_factor_true", "age", "saturated_fat_pct", "fiber_g_day",
             "sodium_mg_day", "stress_level", "simple_carbs_g", "cardio_met_minutes", "strength_met_minutes",
             "alcohol_g_per_week", "sleep_hours", "macro_imbalance"]
            for y in range(self.config.years)
        ]
        # Проверяем только наличие хотя бы одной колонки каждого префикса
        prefixes = ["weight_kg_true", "muscle_mass_factor_true", "age", "saturated_fat_pct", "fiber_g_day",
                    "sodium_mg_day", "stress_level", "simple_carbs_g", "cardio_met_minutes", "strength_met_minutes",
                    "alcohol_g_per_week", "sleep_hours", "macro_imbalance"]
        for pref in prefixes:
            cols = [c for c in lifestyle_df.columns if c.startswith(pref)]
            if not cols:
                raise ValueError(f"Missing columns with prefix '{pref}' in lifestyle_df")

    def _extract_baseline_data(
            self, cohort_by_id: pd.DataFrame, person_ids: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Извлекает начальные данные из когорты."""
        height = cohort_by_id.loc[person_ids]["height_cm"].values
        initial_hdl = cohort_by_id.loc[person_ids]["initial_hdl_mgdl"].values
        initial_tc = cohort_by_id.loc[person_ids]["initial_total_cholesterol_mgdl"].values
        lipid_factor_hdl = cohort_by_id.loc[person_ids]["lipid_factor_hdl"].values
        lipid_factor_tc = cohort_by_id.loc[person_ids]["lipid_factor_tc"].values
        initial_hba1c = cohort_by_id.loc[person_ids]["initial_hba1c_percent"].values
        initial_sbp = cohort_by_id.loc[person_ids]["initial_sbp_mmhg"].values
        sex = cohort_by_id.loc[person_ids]["sex"].values
        return (height, initial_hdl, initial_tc, lipid_factor_hdl, lipid_factor_tc,
                initial_hba1c, initial_sbp, sex)

    def _extract_lifestyle_arrays(
            self, lifestyle_df: pd.DataFrame, years: int
    ) -> Dict[str, np.ndarray]:
        """Извлекает все временные ряды из lifestyle_df в виде словаря массивов."""

        def extract(prefix: str) -> np.ndarray:
            cols = [f"{prefix}_{y}" for y in range(years)]
            return lifestyle_df[cols].values

        return {
            "weight_true": extract("weight_kg_true"),
            "muscle_true": extract("muscle_mass_factor_true"),
            "age_vals": extract("age"),
            "saturated_fat": extract("saturated_fat_pct"),
            "fiber": extract("fiber_g_day"),
            "sodium": extract("sodium_mg_day"),
            "stress": extract("stress_level"),
            "simple_carbs_g": extract("simple_carbs_g"),
            "cardio_met": extract("cardio_met_minutes"),
            "strength_met": extract("strength_met_minutes"),
            "alcohol": extract("alcohol_g_per_week"),
            "sleep": extract("sleep_hours"),
            "macro_imb": extract("macro_imbalance"),
        }

    def _precompute_constants(
            self, height: np.ndarray, lipid_factor_hdl: np.ndarray, lipid_factor_tc: np.ndarray
    ) -> None:
        """Предвычисляет константы, используемые во всех годах."""
        height_m = height / 100.0
        self._height_sq = (height_m ** 2).reshape(-1, 1)  # (n, 1)
        self._genetic_hdl_offset = (self.lipid_effect * lipid_factor_hdl).reshape(-1, 1)
        self._genetic_tc_offset = (-self.lipid_effect * lipid_factor_tc).reshape(-1, 1)

    def _precompute_age_norm_sbp(self, age_vals: np.ndarray) -> np.ndarray:
        """Предвычисляет возрастную норму SBP для всех лет."""
        coeff = self.coeff.sbp
        return coeff.AGE_NORM_SBP_INTERCEPT + coeff.AGE_NORM_SBP_SLOPE * np.maximum(
            0, age_vals - coeff.AGE_NORM_THRESHOLD
        )

    def _precompute_age_norm_hba1c(self, age_vals: np.ndarray) -> np.ndarray:
        """Предвычисляет возрастную норму HbA1c для всех лет."""
        coeff = self.coeff.hba1c
        return coeff.AGE_NORM_INTERCEPT + coeff.AGE_NORM_SLOPE * np.maximum(
            0, age_vals - coeff.AGE_NORM_THRESHOLD
        )

    def _compute_deltas(
            self,
            saturated_fat: np.ndarray,
            fiber: np.ndarray,
            sodium: np.ndarray,
            stress: np.ndarray,
            simple_carbs_g: np.ndarray,
            bmi: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """Вычисляет разности между годами для всех факторов."""
        delta_sat_fat = np.diff(saturated_fat, axis=1)
        delta_fiber = np.diff(fiber, axis=1)
        delta_sodium = np.diff(sodium, axis=1)
        delta_stress = np.diff(stress, axis=1)
        delta_simple_carbs = np.diff(simple_carbs_g, axis=1)
        delta_bmi = np.diff(bmi, axis=1)
        return (delta_sat_fat, delta_fiber, delta_sodium, delta_stress,
                delta_simple_carbs, delta_bmi)

    # Методы обновления каждого биомаркера

    def _update_hdl(
            self,
            hdl_prev: np.ndarray,
            change: np.ndarray,
            genetic_offset: np.ndarray,
    ) -> np.ndarray:
        """Обновляет HDL: (предыдущее - генетика) + изменение + генетика, затем обрезка."""
        without_genetic = hdl_prev - genetic_offset
        new_lifestyle = without_genetic + change
        return self._clip_biomarker(new_lifestyle + genetic_offset, "hdl")

    def _update_tc(
            self,
            tc_prev: np.ndarray,
            change: np.ndarray,
            genetic_offset: np.ndarray,
    ) -> np.ndarray:
        without_genetic = tc_prev - genetic_offset
        new_lifestyle = without_genetic + change
        return self._clip_biomarker(new_lifestyle + genetic_offset, "tc")

    def _update_sbp(self, sbp_prev: np.ndarray, change: np.ndarray) -> np.ndarray:
        return self._clip_biomarker(sbp_prev + change, "sbp")

    def _update_hba1c(self, hba1c_prev: np.ndarray, change: np.ndarray) -> np.ndarray:
        return self._clip_biomarker(hba1c_prev + change, "hba1c")

    # Методы расчёта изменений для каждого биомаркера (векторизованные)

    def _calc_hdl_change(
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
        """Вычисляет годовое изменение HDL."""
        coeff = self.coeff.hdl
        cardio_benefit = coeff.CARDIO_COEF * (cardio_met / coeff.CARDIO_REFERENCE_MET)
        strength_benefit = coeff.STRENGTH_COEF * (strength_met / coeff.STRENGTH_REFERENCE_MET)
        fiber_benefit = coeff.FIBER_COEF * delta_fiber
        fat_penalty = coeff.SATURATED_FAT_COEF * (delta_sat_fat * coeff.SFA_PPT_CONVERSION)
        muscle_benefit = coeff.MUSCLE_MASS_COEF * muscle_mass
        bmi_penalty = coeff.BMI_LINEAR_COEF * (bmi_curr - coeff.BMI_LINEAR_REFERENCE)
        age_penalty = coeff.AGE_COEF * np.maximum(0, age_curr - coeff.AGE_THRESHOLD)
        sex_factor = np.where(sex == 'F', coeff.FEMALE_BONUS, 0)[:, None]

        total = (cardio_benefit + strength_benefit + fiber_benefit + muscle_benefit +
                 fat_penalty + bmi_penalty + age_penalty + sex_factor)
        return np.clip(total, coeff.CHANGE_MIN, coeff.CHANGE_MAX)

    def _calc_tc_change(
            self,
            delta_sat_fat: np.ndarray,
            delta_fiber: np.ndarray,
            total_met: np.ndarray,
    ) -> np.ndarray:
        """Вычисляет годовое изменение общего холестерина."""
        coeff = self.coeff.tc
        sat_fat_effect = coeff.SATURATED_FAT_COEF * (delta_sat_fat * coeff.SFA_PPT_CONVERSION)
        fiber_effect = coeff.FIBER_COEF * delta_fiber
        activity_effect = coeff.ACTIVITY_COEF * (total_met / coeff.ACTIVITY_REFERENCE_MET)
        return sat_fat_effect + fiber_effect + activity_effect

    def _calc_sbp_change(
            self,
            delta_bmi: np.ndarray,
            bmi_curr: np.ndarray,
            alcohol_day: np.ndarray,
            delta_sodium: np.ndarray,
            delta_stress: np.ndarray,
            sleep_hours: np.ndarray,
            age: np.ndarray,
            current_sbp: np.ndarray,
            age_norm_sbp: np.ndarray,
    ) -> np.ndarray:
        """Вычисляет годовое изменение систолического давления."""
        coeff = self.coeff.sbp
        bmi_delta_effect = coeff.BMI_DELTA_COEF * delta_bmi
        bmi_abs_effect = coeff.BMI_ABSOLUTE_COEF * np.maximum(
            0, bmi_curr - coeff.BMI_ABSOLUTE_THRESHOLD
        )
        alcohol_effect = np.where(
            alcohol_day <= coeff.ALCOHOL_THRESHOLD_G_PER_DAY,
            0,
            (alcohol_day - coeff.ALCOHOL_THRESHOLD_G_PER_DAY) * coeff.ALCOHOL_COEF
        )
        sodium_effect = (delta_sodium / coeff.SODIUM_REFERENCE_MG) * coeff.SODIUM_COEF
        stress_effect = delta_stress * coeff.STRESS_COEF
        sleep_effect = np.maximum(0, coeff.SLEEP_THRESHOLD_HOURS - sleep_hours) * coeff.SLEEP_DEFICIT_COEF
        age_effect = coeff.AGE_COEF * np.maximum(0, age - coeff.AGE_THRESHOLD)
        reversion = (age_norm_sbp - current_sbp) * coeff.REVERSION_COEF

        total = (bmi_delta_effect + bmi_abs_effect + alcohol_effect + sodium_effect +
                 stress_effect + sleep_effect + age_effect + reversion)
        return np.clip(total, coeff.CHANGE_MIN, coeff.CHANGE_MAX)

    def _calc_hba1c_change(
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
            age_norm_hba1c: np.ndarray,
    ) -> np.ndarray:
        """Вычисляет годовое изменение HbA1c."""
        coeff = self.coeff.hba1c
        base_drift = coeff.BASE_DRIFT
        reversion = (age_norm_hba1c - current_hba1c) * coeff.REVERSION_COEF
        carb_effect = (delta_simple_carbs / coeff.SIMPLE_CARBS_REFERENCE_G) * coeff.SIMPLE_CARBS_COEF
        bmi_effect = delta_bmi * coeff.BMI_CHANGE_COEF
        fiber_effect = delta_fiber * coeff.FIBER_COEF
        met_effect = (total_met / coeff.ACTIVITY_REFERENCE_MET) * coeff.ACTIVITY_COEF
        macro_effect = macro_imbalance * coeff.MBI_COEF
        muscle_effect = coeff.MUSCLE_MASS_COEF * muscle_mass
        sleep_deficit_effect = (sleep_hours < coeff.SLEEP_THRESHOLD_HOURS).astype(float) * coeff.SLEEP_DEFICIT_COEF

        total = (base_drift + reversion + carb_effect + bmi_effect + fiber_effect +
                 met_effect + macro_effect + muscle_effect + sleep_deficit_effect)
        return np.clip(total, coeff.CHANGE_MIN, coeff.CHANGE_MAX)

    # Вспомогательные методы

    def _clip_biomarker(self, values: np.ndarray, marker_type: str) -> np.ndarray:
        """Обрезает значения биомаркеров до физиологических диапазонов."""
        key = self.MARKER_MAP.get(marker_type, marker_type)
        # Используем функцию из helpers, передавая self.biomarker_ranges
        return clip_biomarker(values, key, self.biomarker_ranges)

    def _build_output_dataframe(
            self,
            person_ids: np.ndarray,
            years: int,
            n_people: int,
            weight_true: np.ndarray,
            hdl_true: np.ndarray,
            tc_true: np.ndarray,
            sbp_true: np.ndarray,
            hba1c_true: np.ndarray,
            muscle_true: np.ndarray,
            bmi: np.ndarray,
            height_sq: np.ndarray,
            lifestyle_df: pd.DataFrame,
            age_vals: np.ndarray,
    ) -> pd.DataFrame:
        """Строит выходной DataFrame с зашумлёнными измерениями."""
        data_dict = {"person_id": person_ids}
        noise = self.measurement_noise

        for y in range(years):
            weight_noisy = weight_true[:, y] + self.rng.normal(0, noise["weight"], n_people)
            data_dict[f"weight_kg_{y}"] = self._clip_biomarker(weight_noisy, "weight")

            hdl_noisy = hdl_true[:, y] + self.rng.normal(0, noise["hdl"], n_people)
            data_dict[f"hdl_mgdl_{y}"] = self._clip_biomarker(hdl_noisy, "hdl")

            tc_noisy = tc_true[:, y] + self.rng.normal(0, noise["tc"], n_people)
            data_dict[f"total_cholesterol_mgdl_{y}"] = self._clip_biomarker(tc_noisy, "tc")

            sbp_noisy = sbp_true[:, y] + self.rng.normal(0, noise["sbp"], n_people)
            data_dict[f"sbp_mmhg_{y}"] = self._clip_biomarker(sbp_noisy, "sbp")

            hba1c_noisy = hba1c_true[:, y] + self.rng.normal(0, noise["hba1c"], n_people)
            data_dict[f"hba1c_percent_{y}"] = self._clip_biomarker(hba1c_noisy, "hba1c")

            data_dict[f"bmi_{y}"] = self._clip_biomarker(bmi[:, y], "bmi")
            bmi_corr = bmi[:, y] * (1 - 0.01 * muscle_true[:, y])
            data_dict[f"bmi_corrected_{y}"] = self._clip_biomarker(bmi_corr, "bmi")

            non_hdl_mgdl = data_dict[f"total_cholesterol_mgdl_{y}"] - data_dict[f"hdl_mgdl_{y}"]
            non_hdl_mgdl_clipped = self._clip_biomarker(non_hdl_mgdl, "non_hdl_mgdl")
            data_dict[f"non_hdl_mgdl_{y}"] = non_hdl_mgdl_clipped
            data_dict[f"non_hdl_mmol_{y}"] = non_hdl_mgdl_clipped / 38.67

            data_dict[f"muscle_mass_factor_{y}"] = muscle_true[:, y]
            data_dict[f"age_{y}"] = age_vals[:, y]

            # Копирование дополнительных колонок из lifestyle_df
            for col in ["cumulative_smoking", "alcohol_g_per_week", "fiber_g_day",
                        "saturated_fat_pct", "macro_imbalance", "sleep_hours", "total_met_minutes"]:
                data_dict[f"{col}_{y}"] = lifestyle_df[f"{col}_{y}"].values

        return pd.DataFrame(data_dict)

    def _log_final_stats(self, hba1c_true: np.ndarray, sbp_true: np.ndarray, years: int) -> None:
        """Логирует итоговую статистику (только если verbose=True)."""
        if self.verbose:
            logger.info(f"Mean HbA1c year 0: {np.mean(hba1c_true[:, 0]):.2f}, "
                        f"year {years - 1}: {np.mean(hba1c_true[:, -1]):.2f}")
            logger.info(f"Mean SBP year 0: {np.mean(sbp_true[:, 0]):.2f}, "
                        f"year {years - 1}: {np.mean(sbp_true[:, -1]):.2f}")
            logger.info("SBP mean by year: " +
                        ", ".join(f"{y}: {np.mean(sbp_true[:, y]):.2f}" for y in [0, 1, 2, 3, 4, 5, 10, 15, 19]))
            logger.info("HbA1c mean by year: " +
                        ", ".join(f"{y}: {np.mean(hba1c_true[:, y]):.2f}" for y in [0, 1, 2, 3, 4, 5, 10, 15, 19]))
