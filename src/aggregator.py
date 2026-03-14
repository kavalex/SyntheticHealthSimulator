"""
Data aggregation and MAR missing value generation module.
Uses physiological ranges from config.py for boundary filtering.
"""

import logging
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd

from src.config import SimulationConfig

logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Aggregates panel data into one row per patient with optional MAR missing values.
    Uses config.final_biomarker_ranges for boundary filtering (no hardcoded values).
    """

    # Маппинг ключей final_ranges на имена колонок в biomarkers_df
    COLUMN_MAPPING: Dict[str, str] = {
        "bmi": "bmi_{}",
        "hdl_mgdl": "hdl_mgdl_{}",
        "total_cholesterol_mgdl": "total_cholesterol_mgdl_{}",
        "sbp_mmhg": "sbp_mmhg_{}",
        "hba1c_percent": "hba1c_percent_{}",
        "weight_kg": "weight_kg_{}",
    }

    # Параметры генерации MAR
    MAR_BASE_PROB: float = 0.05
    MAR_AGE_FACTOR: float = 0.15  # was 0.08, increased for stronger MAR pattern
    MAR_RISK_FACTOR: float = 0.05
    MAR_AGE_THRESHOLD: int = 50  # was 45, shifted to create clearer age groups
    MAR_RISK_THRESHOLD: float = 0.2
    MAR_PROB_MIN: float = 0.02
    MAR_PROB_MAX: float = 0.15

    MAR_COLUMNS: List[str] = [
        "final_sbp_mmhg",
        "final_hdl_mgdl",
        "final_total_cholesterol_mgdl",
        "final_hba1c_percent",
        "avg_alcohol_g_per_week",
        "avg_total_met_minutes",
    ]

    def __init__(
            self,
            config: SimulationConfig,
            rng: np.random.Generator,
    ) -> None:
        """
        Инициализация агрегатора.

        Args:
            config: Конфигурация симуляции.
            rng: Генератор случайных чисел.
        """
        self.config = config
        self.rng = rng
        self.final_ranges = config.final_biomarker_ranges

    def aggregate(
            self,
            cohort_df: pd.DataFrame,
            biomarkers_df: pd.DataFrame,
            risks_df: pd.DataFrame,
            years: Optional[int] = None,
            apply_boundary_filter: bool = True,
            apply_mar: bool = True,
    ) -> Tuple[pd.DataFrame, np.ndarray, int]:
        """
        Основной метод агрегации.

        Args:
            cohort_df: DataFrame с данными когорты (один ряд на пациента).
            biomarkers_df: DataFrame с годовыми биомаркерами.
            risks_df: DataFrame с рисками и исходами.
            years: Количество лет (по умолчанию из config).
            apply_boundary_filter: Применять ли фильтрацию граничных значений.
            apply_mar: Генерировать ли пропуски MAR.

        Returns:
            Кортеж (aggregated_df, filtered_mask, n_original):
                aggregated_df: итоговый DataFrame с одной строкой на пациента.
                filtered_mask: булева маска пациентов, отфильтрованных по границам.
                n_original: исходное количество пациентов.
        """
        # Валидация входных данных
        self._validate_inputs(cohort_df, biomarkers_df, risks_df)

        years = years or self.config.years
        n_original = len(cohort_df)
        last_year = years - 1

        filtered_mask = self._compute_boundary_mask(
            biomarkers_df, years, apply_boundary_filter
        )

        aggregated = self._build_aggregated_base(
            cohort_df, biomarkers_df, risks_df, years, last_year
        )

        if apply_boundary_filter and filtered_mask.sum() > 0:
            n_before = len(aggregated)
            aggregated = aggregated[~filtered_mask].reset_index(drop=True)
            n_after = len(aggregated)
            logger.info(
                f"Boundary filtering: {n_after}/{n_before} patients retained "
                f"({n_after / n_before * 100:.1f}%)"
            )

        if apply_mar:
            aggregated = self._generate_mar_missing(aggregated)

        return aggregated, filtered_mask, n_original

    def _validate_inputs(
            self,
            cohort_df: pd.DataFrame,
            biomarkers_df: pd.DataFrame,
            risks_df: pd.DataFrame,
    ) -> None:
        """
        Проверяет наличие необходимых колонок во входных DataFrame.
        В случае отсутствия критических колонок выбрасывает ValueError.
        """
        required_cohort = ["person_id", "age_start", "sex", "height_cm", "weight_start_kg", "bmi_start"]
        missing_cohort = [col for col in required_cohort if col not in cohort_df.columns]
        if missing_cohort:
            raise ValueError(f"Missing columns in cohort_df: {missing_cohort}")

        # Проверяем только наличие person_id в biomarkers_df
        if "person_id" not in biomarkers_df.columns:
            raise ValueError("Missing 'person_id' in biomarkers_df")

        if "person_id" not in risks_df.columns:
            raise ValueError("Missing 'person_id' in risks_df")

    def _compute_boundary_mask(
            self,
            biomarkers_df: pd.DataFrame,
            years: int,
            apply_filter: bool,
    ) -> np.ndarray:
        """
        Вычисляет маску пациентов, чьи финальные биомаркеры выходят за допустимые границы.

        Args:
            biomarkers_df: DataFrame с биомаркерами.
            years: Количество лет наблюдения.
            apply_filter: Флаг применения фильтрации.

        Returns:
            Булева маска длиной len(biomarkers_df): True для пациентов, подлежащих удалению.
        """
        if not apply_filter:
            return np.zeros(len(biomarkers_df), dtype=bool)

        n_original = len(biomarkers_df)
        filtered_mask = np.zeros(n_original, dtype=bool)
        last_year = years - 1

        for final_col, (low, high) in self.final_ranges.items():
            template = self.COLUMN_MAPPING.get(final_col)
            if template is None:
                continue
            biomarker_col = template.format(last_year)
            if biomarker_col not in biomarkers_df.columns:
                logger.warning(f"Column {biomarker_col} not found, skipping boundary check for {final_col}")
                continue

            out_of_range = (
                    (biomarkers_df[biomarker_col] < low) |
                    (biomarkers_df[biomarker_col] > high)
            )
            n_out = out_of_range.sum()
            if n_out > 0:
                logger.debug(f"Out of range in {biomarker_col}: {n_out} patients")
            filtered_mask |= out_of_range

        return filtered_mask

    def _build_aggregated_base(
            self,
            cohort_df: pd.DataFrame,
            biomarkers_df: pd.DataFrame,
            risks_df: pd.DataFrame,
            years: int,
            last_year: int,
    ) -> pd.DataFrame:
        """
        Строит базовый агрегированный датасет, объединяя данные из cohort, biomarkers и risks.

        Args:
            cohort_df: DataFrame когорты.
            biomarkers_df: DataFrame биомаркеров.
            risks_df: DataFrame рисков.
            years: Количество лет.
            last_year: Индекс последнего года.

        Returns:
            DataFrame с одной строкой на пациента, содержащий:
                - все колонки из cohort_df,
                - средние и стандартные отклонения факторов образа жизни,
                - финальные значения биомаркеров,
                - риски и исходы,
                - возраст в конце наблюдения.
        """
        aggregated = cohort_df.copy()

        # Усреднённые показатели образа жизни
        lifestyle_cols = [
            "alcohol_g_per_week", "total_met_minutes", "cigarettes_per_day",
            "stress_level", "sleep_hours", "fiber_g_day",
            "saturated_fat_pct", "macro_imbalance",
        ]
        for col in lifestyle_cols:
            year_cols = [f"{col}_{year}" for year in range(years)]
            existing_cols = [c for c in year_cols if c in biomarkers_df.columns]
            if existing_cols:
                aggregated[f"avg_{col}"] = biomarkers_df[existing_cols].mean(axis=1)
                aggregated[f"std_{col}"] = biomarkers_df[existing_cols].std(axis=1)
            else:
                aggregated[f"avg_{col}"] = np.nan
                aggregated[f"std_{col}"] = np.nan

        # Финальные значения биомаркеров
        aggregated["final_bmi"] = biomarkers_df[f"bmi_{last_year}"]
        aggregated["final_hdl_mgdl"] = biomarkers_df[f"hdl_mgdl_{last_year}"]
        aggregated["final_total_cholesterol_mgdl"] = biomarkers_df[f"total_cholesterol_mgdl_{last_year}"]
        aggregated["final_sbp_mmhg"] = biomarkers_df[f"sbp_mmhg_{last_year}"]
        aggregated["final_hba1c_percent"] = biomarkers_df[f"hba1c_percent_{last_year}"]
        aggregated["cumulative_smoking_end"] = biomarkers_df[f"cumulative_smoking_{last_year}"]

        # Объединение с рисками (исключаем дублирующийся столбец sex, если он есть)
        risks_no_sex = risks_df.drop(columns=["sex"], errors="ignore")
        aggregated = aggregated.merge(risks_no_sex, on="person_id", how="left")
        aggregated["age_end"] = aggregated["age_start"] + years

        return aggregated

    def _generate_mar_missing(
            self,
            df: pd.DataFrame,
            age_col: str = "age_end",
            risk_col: str = "cvd_risk_10year",
    ) -> pd.DataFrame:
        """
        Генерирует пропуски MAR в указанных колонках.
        Вероятность пропуска зависит от возраста и риска ССЗ.

        Args:
            df: DataFrame для внесения пропусков.
            age_col: Название колонки с возрастом.
            risk_col: Название колонки с риском ССЗ.

        Returns:
            DataFrame с внесёнными пропусками.
        """
        df_copy = df.copy()

        if age_col in df_copy.columns and risk_col in df_copy.columns:
            age = df_copy[age_col].values
            cvd_risk = df_copy[risk_col].values
            p_missing_base = (
                    self.MAR_BASE_PROB
                    + self.MAR_AGE_FACTOR * (age < self.MAR_AGE_THRESHOLD)
                    - self.MAR_RISK_FACTOR * (cvd_risk > self.MAR_RISK_THRESHOLD)
            )
            p_missing = np.clip(p_missing_base, self.MAR_PROB_MIN, self.MAR_PROB_MAX)
        else:
            p_missing = np.full(len(df_copy), self.MAR_BASE_PROB)
            logger.warning(f"Columns {age_col} or {risk_col} not found, using base MAR probability.")

        for col in self.MAR_COLUMNS:
            if col in df_copy.columns:
                mask = self.rng.random(len(df_copy)) < p_missing
                df_copy.loc[mask, col] = np.nan
                n_missing = mask.sum()
                if n_missing > 0:
                    logger.debug(f"MAR: inserted {n_missing} missing values in {col}")

        return df_copy
