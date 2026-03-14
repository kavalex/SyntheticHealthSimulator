#!/usr/bin/env python3
"""
SyntheticHealthSimulator - Main Facade
Medical Data Generator with Full Functionality and Calibrated Realism
WARNING: NOT FOR CLINICAL USE - EDUCATIONAL PURPOSES ONLY
"""
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd

from calculators.biomarkers import BiomarkerCalculator
from calculators.risks import RiskCalculator
# Import configuration and logging
from config import SimulationConfig, DEFAULT_CONFIG
# Import modular components
from generators.cohort import CohortGenerator
from generators.lifestyle import LifestyleSimulator
from logging_config import setup_logging, get_logger

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

__version__ = "2.1.0"
__version_info__ = (2, 1, 0)


class DataGenerator:
    """
    Synthetic Medical Data Generator - Facade Pattern

    Delegates work to specialized modules:
    - CohortGenerator: Baseline demographics and genetic risks
    - LifestyleSimulator: 20-year lifestyle trajectories (OU processes)
    - BiomarkerCalculator: Annual biomarker measurements with noise
    - RiskCalculator: Disease risks and binary outcomes

    WARNING: NOT FOR CLINICAL USE
    """

    def __init__(
            self,
            config: SimulationConfig = None,
            seed: int = 42,
            log_level: int = logging.INFO,
    ):
        """
        Initialize generator with calibrated parameters.

        Args:
            config: SimulationConfig instance (uses DEFAULT_CONFIG if None)
            seed: Random seed for reproducibility (default: 42)
            log_level: Logging level (default: logging.INFO)
        """
        # ✅ Инициализация конфигурации ПЕРЕД созданием компонентов
        self.config = config or DEFAULT_CONFIG

        # ✅ Инициализация логгера
        setup_logging(level=log_level)
        self.logger = get_logger()

        # ✅ Инициализация RNG ПЕРЕД созданием компонентов
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # ✅ Теперь можно создавать компоненты (config и rng уже определены)
        self.cohort_gen = CohortGenerator(self.config, self.rng)
        self.lifestyle_sim = LifestyleSimulator(self.config, self.rng)
        self.biomarker_calc = BiomarkerCalculator(self.config, self.rng)
        self.risk_calc = RiskCalculator(self.config, self.rng)

        # Parameters from config for backward compatibility
        self.diabetes_threshold = self.config.diabetes_threshold
        self.BIOMARKER_RANGES = self.config.biomarker_ranges
        self.FINAL_BIOMARKER_RANGES = self.config.final_biomarker_ranges
        self.MEASUREMENT_NOISE = self.config.measurement_noise

        self.logger.info(f"DataGenerator initialized (seed={seed}, years={self.config.years})")

    @staticmethod
    def _validate_positive(value: Union[int, float], name: str) -> None:
        """Validate that a value is positive"""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    def generate_cohort(
            self,
            n_people: int = 5000,
            age_range: Tuple[int, int] = None
    ) -> pd.DataFrame:
        """Generate baseline cohort (delegated to CohortGenerator)."""
        self._validate_positive(n_people, "n_people")
        return self.cohort_gen.generate(n_people, age_range or self.config.age_range)

    def generate_lifestyle_history(
            self,
            cohort_df: pd.DataFrame,
            years: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate lifestyle history (delegated to LifestyleSimulator)."""
        years = years or self.config.years
        self._validate_positive(years, "years")
        return self.lifestyle_sim.simulate(cohort_df, years)

    def calculate_biomarkers(
            self,
            cohort_df: pd.DataFrame,
            lifestyle_df: pd.DataFrame,
            years: int = None,
    ) -> pd.DataFrame:
        """Calculate biomarkers (delegated to BiomarkerCalculator)."""
        years = years or self.config.years
        self._validate_positive(years, "years")
        return self.biomarker_calc.calculate(cohort_df, lifestyle_df, years)

    def calculate_health_risks(
            self,
            biomarkers_df: pd.DataFrame,
            cohort_df: pd.DataFrame,
            years: int = None,
    ) -> pd.DataFrame:
        """Calculate health risks (delegated to RiskCalculator)."""
        years = years or self.config.years
        self._validate_positive(years, "years")
        return self.risk_calc.calculate(biomarkers_df, cohort_df, years)

    def create_aggregated_dataset(
            self,
            cohort_df: pd.DataFrame,
            biomarkers_df: pd.DataFrame,
            risks_df: pd.DataFrame,
            years: int = None,
            apply_mar: bool = True,
            filter_boundary: bool = True,
            target_n_people: int = 5000,
    ) -> Tuple[pd.DataFrame, np.ndarray, int]:
        """
        Aggregate data into one row per patient with optional boundary filtering.

        Returns:
            tuple: (aggregated_df, filtered_mask, n_original)
        """
        years = years or self.config.years
        self._validate_positive(years, "years")

        n_original = len(cohort_df)
        filtered_mask = np.zeros(n_original, dtype=bool)

        # Boundary filtering for final-year biomarkers
        if filter_boundary:
            self.logger.info("Filtering boundary artifacts...")
            final_ranges = {
                'final_bmi': (16, 50),
                'final_sbp_mmhg': (80, 200),
                'final_hdl_mgdl': (20, 100),
                'final_total_cholesterol_mgdl': (150, 350),
                'final_hba1c_percent': (3.5, 12.0),
            }
            last_year = years - 1

            for col, (low, high) in final_ranges.items():
                if col == 'final_bmi':
                    biomarker_col = f"bmi_{last_year}"
                elif col == 'final_sbp_mmhg':
                    biomarker_col = f"sbp_mmhg_{last_year}"
                elif col == 'final_hdl_mgdl':
                    biomarker_col = f"hdl_mgdl_{last_year}"
                elif col == 'final_total_cholesterol_mgdl':
                    biomarker_col = f"total_cholesterol_mgdl_{last_year}"
                elif col == 'final_hba1c_percent':
                    biomarker_col = f"hba1c_percent_{last_year}"
                else:
                    continue

                if biomarker_col in biomarkers_df.columns:
                    out_of_range = (
                            (biomarkers_df[biomarker_col] < low) |
                            (biomarkers_df[biomarker_col] > high)
                    )
                    filtered_mask |= out_of_range
                    self.logger.info(
                        f"  {col}: {out_of_range.sum()} rows out of range "
                        f"({out_of_range.mean() * 100:.2f}%)"
                    )

            # Check weight separately
            weight_col = f"weight_kg_{last_year}"
            if weight_col in biomarkers_df.columns:
                weight_out_of_range = (
                        (biomarkers_df[weight_col] < 40) |
                        (biomarkers_df[weight_col] > 200)
                )
                filtered_mask |= weight_out_of_range

        n_filtered = filtered_mask.sum()
        filter_pct = n_filtered / n_original * 100
        self.logger.info(
            f"Boundary filtering: {n_filtered} rows ({filter_pct:.2f}%) marked for removal"
        )

        # Build aggregated dataset
        aggregated = cohort_df.copy()

        # Lifestyle aggregates
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

        # Final biomarkers
        aggregated["final_bmi"] = biomarkers_df[f"bmi_{years - 1}"]
        aggregated["final_bmi_corrected"] = biomarkers_df[f"bmi_corrected_{years - 1}"]
        aggregated["final_hdl_mgdl"] = biomarkers_df[f"hdl_mgdl_{years - 1}"]
        aggregated["final_total_cholesterol_mgdl"] = biomarkers_df[f"total_cholesterol_mgdl_{years - 1}"]
        aggregated["final_non_hdl_mgdl"] = biomarkers_df[f"non_hdl_mgdl_{years - 1}"]
        aggregated["final_non_hdl_mmol"] = biomarkers_df[f"non_hdl_mmol_{years - 1}"]
        aggregated["final_sbp_mmhg"] = biomarkers_df[f"sbp_mmhg_{years - 1}"]
        aggregated["final_hba1c_percent"] = biomarkers_df[f"hba1c_percent_{years - 1}"]
        aggregated["final_muscle_mass"] = biomarkers_df[f"muscle_mass_factor_{years - 1}"]
        aggregated['cumulative_smoking_end'] = biomarkers_df[f'cumulative_smoking_{years - 1}']

        # Merge with risks
        aggregated = aggregated.merge(risks_df, on="person_id", how="left")
        aggregated["age_end"] = aggregated["age_start"] + years

        # Apply boundary filtering
        if filter_boundary and filtered_mask.sum() > 0:
            n_before = len(aggregated)
            aggregated = aggregated[~filtered_mask].reset_index(drop=True)
            n_after = len(aggregated)
            self.logger.info(
                f"After boundary filtering: {n_after} patients "
                f"({n_after / n_before * 100:.1f}% retained)"
            )

        # Generate MAR missing values
        if apply_mar:
            aggregated = self._generate_mar_missing_values(aggregated)

        self.logger.info(f"Aggregated Dataset: {len(aggregated)} patients, {aggregated.shape[1]} columns")

        return aggregated, filtered_mask, n_original

    def _generate_mar_missing_values(
            self,
            df: pd.DataFrame,
            age_col: str = "age_end",
            risk_col: str = "cvd_risk_10year",
    ) -> pd.DataFrame:
        """Generate MAR missing values (2-15% missingness)."""
        df_copy = df.copy()

        if age_col in df_copy.columns and risk_col in df_copy.columns:
            age = df_copy[age_col].values
            cvd_risk = df_copy[risk_col].values
            p_missing_base = 0.05 + 0.08 * (age < 45) - 0.05 * (cvd_risk > 0.2)
            p_missing = np.clip(p_missing_base, 0.02, 0.15)
        else:
            p_missing = np.full(len(df_copy), 0.07)

        cols_with_missing = [
            "final_sbp_mmhg", "final_hdl_mgdl", "final_total_cholesterol_mgdl",
            "final_hba1c_percent", "avg_alcohol_g_per_week", "avg_total_met_minutes",
        ]

        for col in cols_with_missing:
            if col in df_copy.columns:
                mask = self.rng.random(len(df_copy)) < p_missing
                df_copy.loc[mask, col] = np.nan

        return df_copy

    def save(
            self,
            df: pd.DataFrame,
            name: str,
            output_dir: Optional[str] = None,
            generation_params: Optional[dict] = None,
    ) -> str:
        """Save DataFrame as CSV with accompanying metadata JSON."""
        if output_dir is None:
            output_dir = f"data/synthetic_v{__version__}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filepath = output_path / f"{name}.csv"
        df.to_csv(filepath, index=False, encoding="utf-8")

        # Metadata
        metadata = {
            "version": __version__,
            "title": f"Comprehensive Health Risk Dataset - {name}",
            "generated_at": datetime.now().isoformat(),
            "generation_parameters": generation_params or {},
            "warning": "NOT FOR CLINICAL USE - EDUCATIONAL PURPOSES ONLY",
        }

        with open(output_path / f"{name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"{name} saved: {filepath} ({df.shape[0]} rows x {df.shape[1]} columns)")
        return str(filepath)


def main(seed=42, n_people=5000, years=20, verbose=True):
    """Main entry point for data generation."""
    level = logging.DEBUG if verbose else logging.INFO
    generator = DataGenerator(seed=seed, log_level=level)

    target_n_people = n_people
    boundary_coefficient = 2.0
    n_people = int(target_n_people * boundary_coefficient)

    generator.logger.info(f"Starting generation (target={target_n_people}, years={years})")

    # 1/5 Generate cohort
    generator.logger.info("1/5 Generating baseline cohort...")
    cohort = generator.generate_cohort(n_people)

    # 2/5 Generate lifestyle
    generator.logger.info(f"2/5 Generating lifestyle history for {years} years...")
    lifestyle = generator.generate_lifestyle_history(cohort, years=years)

    # 3/5 Calculate biomarkers
    generator.logger.info("3/5 Calculating biomarkers...")
    biomarkers = generator.calculate_biomarkers(cohort, lifestyle, years=years)

    # 4/5 Calculate risks
    generator.logger.info("4/5 Calculating medical risks...")
    risks = generator.calculate_health_risks(biomarkers, cohort, years=years)

    # 5/5 Create aggregated dataset
    generator.logger.info("5/5 Creating aggregated dataset...")
    aggregated, filtered_mask, n_original = generator.create_aggregated_dataset(
        cohort, biomarkers, risks,
        years=years,
        apply_mar=True,
        filter_boundary=True,
        target_n_people=target_n_people,
    )

    valid_ids = set(aggregated["person_id"])

    cohort = cohort[cohort["person_id"].isin(valid_ids)].reset_index(drop=True)
    lifestyle = lifestyle[lifestyle["person_id"].isin(valid_ids)].reset_index(drop=True)
    biomarkers = biomarkers[biomarkers["person_id"].isin(valid_ids)].reset_index(drop=True)
    risks = risks[risks["person_id"].isin(valid_ids)].reset_index(drop=True)

    # Trim to target (если всё ещё больше target)
    if len(aggregated) > target_n_people:
        trim_idx = generator.rng.choice(
            len(aggregated), size=target_n_people, replace=False
        )
        trim_idx = np.sort(trim_idx)
        cohort = cohort.iloc[trim_idx].reset_index(drop=True)
        lifestyle = lifestyle.iloc[trim_idx].reset_index(drop=True)
        biomarkers = biomarkers.iloc[trim_idx].reset_index(drop=True)
        risks = risks.iloc[trim_idx].reset_index(drop=True)
        aggregated = aggregated.iloc[trim_idx].reset_index(drop=True)

    # Save
    generation_params = {
        "target_n_people": target_n_people,
        "years_observation": years,
        "seed": seed,
    }

    generator.save(cohort, "01_cohort_baseline", generation_params=generation_params)
    generator.save(lifestyle, "02_lifestyle_history", generation_params=generation_params)
    generator.save(biomarkers, "03_biomarkers_history", generation_params=generation_params)
    generator.save(risks, "04_health_risks", generation_params=generation_params)
    generator.save(aggregated, "05_aggregated_dataset", generation_params=generation_params)

    generator.logger.info("GENERATION COMPLETED!")


if __name__ == "__main__":
    main()
