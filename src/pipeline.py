"""
Main pipeline for SyntheticHealthSimulator.
Combines all steps: cohort generation, lifestyle simulation, biomarker calculation,
risk calculation, and aggregation.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

from src.aggregator import DataAggregator
from src.calculators.biomarkers import BiomarkerCalculator
from src.calculators.risks import RiskCalculator
from src.config import SimulationConfig, DEFAULT_CONFIG
from src.generators.cohort import CohortGenerator
from src.generators.lifestyle import LifestyleSimulator
from src.logging_config import get_logger, setup_logging
from src.utils.paths import get_version, get_data_path

logger = get_logger()

FILE_NAMES = {
    "cohort": "01_cohort_baseline",
    "lifestyle": "02_lifestyle_history",
    "biomarkers": "03_biomarkers_history",
    "risks": "04_health_risks",
    "aggregated": "05_aggregated_dataset_with_missing",
}


def generate(
        n_people: int = 5000,
        years: int = 20,
        seed: int = 42,
        boundary_coefficient: float = 1.35,
        config: Optional[SimulationConfig] = None,
        output_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
) -> Dict[str, pd.DataFrame]:
    """
    Run the full generation pipeline.

    Args:
        n_people: Target number of patients in final dataset.
        years: Number of years to simulate.
        seed: Random seed.
        boundary_coefficient: Oversampling factor to compensate filtering.
        config: SimulationConfig instance (uses DEFAULT_CONFIG if None).
        output_dir: Directory to save outputs. If None, uses data/synthetic_v<version>.
        log_file: Path to log file. If None, no file logging.
        log_level: Logging level.

    Returns:
        Dictionary with five DataFrames: cohort, lifestyle, biomarkers, risks, aggregated.
    """
    setup_logging(level=log_level, log_file=log_file)

    sim_config = config or DEFAULT_CONFIG
    # Override config parameters with explicit arguments
    sim_config = SimulationConfig(
        years=years,
        target_n_people=n_people,
        seed=seed,
        boundary_coefficient=boundary_coefficient,
        age_range=sim_config.age_range,
        ou_params=sim_config.ou_params,
        calorie_params=sim_config.calorie_params,
        base_calories=sim_config.base_calories,
        macro_norms=sim_config.macro_norms,
        muscle_params=sim_config.muscle_params,
        lipid_effect=sim_config.lipid_effect,
        genetic_risk_std=sim_config.genetic_risk_std,
        risk_intercepts=sim_config.risk_intercepts,
        diabetes_threshold=sim_config.diabetes_threshold,
        measurement_noise=sim_config.measurement_noise,
        biomarker_ranges=sim_config.biomarker_ranges,
        final_biomarker_ranges=sim_config.final_biomarker_ranges,
        mar_missing_range=sim_config.mar_missing_range,
    )

    rng = np.random.default_rng(seed)

    n_generate = int(n_people * boundary_coefficient)
    logger.info(f"Starting pipeline: target={n_people}, generate={n_generate}, years={years}, seed={seed}")

    start_time = time.time()

    # Step 1: Cohort
    logger.info("Step 1/5: Generating baseline cohort")
    cohort_gen = CohortGenerator(sim_config, rng)
    cohort = cohort_gen.generate(n_generate)

    # Step 2: Lifestyle
    logger.info("Step 2/5: Simulating lifestyle history")
    lifestyle_sim = LifestyleSimulator(sim_config, rng)
    lifestyle = lifestyle_sim.simulate(cohort, years)

    # Step 3: Biomarkers
    logger.info("Step 3/5: Calculating biomarkers")
    biomarker_calc = BiomarkerCalculator(sim_config, rng)
    biomarkers = biomarker_calc.calculate(cohort, lifestyle, years)

    # Step 4: Risks
    logger.info("Step 4/5: Calculating health risks")
    risk_calc = RiskCalculator(sim_config, rng)
    risks = risk_calc.calculate(biomarkers, cohort, years)

    # Step 5: Aggregation
    logger.info("Step 5/5: Aggregating data")
    aggregator = DataAggregator(sim_config, rng)
    aggregated, filtered_mask, _ = aggregator.aggregate(
        cohort, biomarkers, risks, years,
        apply_boundary_filter=True,
        apply_mar=True,
    )

    if len(aggregated) > n_people:
        logger.info(f"Trimming from {len(aggregated)} to {n_people}")
        trim_idx = rng.choice(len(aggregated), size=n_people, replace=False)
        aggregated = aggregated.iloc[trim_idx].reset_index(drop=True)

    valid_ids = set(aggregated["person_id"])
    cohort = cohort[cohort["person_id"].isin(valid_ids)].reset_index(drop=True)
    lifestyle = lifestyle[lifestyle["person_id"].isin(valid_ids)].reset_index(drop=True)
    biomarkers = biomarkers[biomarkers["person_id"].isin(valid_ids)].reset_index(drop=True)
    risks = risks[risks["person_id"].isin(valid_ids)].reset_index(drop=True)

    elapsed = time.time() - start_time
    logger.info(f"Pipeline complete: {len(aggregated)} patients in {elapsed:.2f}s")

    # Save datasets
    if output_dir is None:
        version = get_version()
        output_dir = str(get_data_path(version))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    version_str = get_version().lstrip('v')
    for key, filename in FILE_NAMES.items():
        df = locals()[key]
        path = Path(output_dir) / f"{filename}_v{version_str}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"Saved: {path} ({df.shape[0]} rows)")

    return {
        "cohort": cohort,
        "lifestyle": lifestyle,
        "biomarkers": biomarkers,
        "risks": risks,
        "aggregated": aggregated,
    }


if __name__ == "__main__":
    # Example usage
    generate()
