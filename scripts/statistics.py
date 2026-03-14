#!/usr/bin/env python3
"""
SyntheticHealthSimulator - Statistics Generator
Generates a single aggregated CSV with statistics for all datasets.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.paths import get_version, get_data_path, get_output_dir

PERCENTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
FLOAT_PRECISION = 4


@dataclass
class ColumnGroup:
    name: str
    columns: List[str]
    description: str


class StatisticsGenerator:
    DEMOGRAPHIC_COLS = ColumnGroup(
        name="Demographics",
        columns=["age_start", "height_cm", "weight_start_kg", "bmi_start"],
        description="Baseline demographic characteristics"
    )

    GENETIC_COLS = ColumnGroup(
        name="Genetic Risks",
        columns=[
            "genetic_risk_diabetes", "genetic_risk_cvd", "genetic_risk_stroke",
            "genetic_risk_nafld", "genetic_risk_cancer", "genetic_risk_cirrhosis"
        ],
        description="Genetic risk factors [0.5, 2.0]"
    )

    INITIAL_BIOMARKER_COLS = ColumnGroup(
        name="Initial Biomarkers",
        columns=[
            "initial_hdl_mgdl", "initial_total_cholesterol_mgdl",
            "initial_sbp_mmhg", "initial_hba1c_percent"
        ],
        description="Baseline biomarker measurements"
    )

    LIFESTYLE_COLS = ColumnGroup(
        name="Lifestyle Factors",
        columns=[
            "alcohol_g_per_week", "cardio_met_minutes", "strength_met_minutes",
            "cigarettes_per_day", "stress_level", "sleep_hours"
        ],
        description="Annual lifestyle trajectory factors"
    )

    BIOMARKER_COLS = ColumnGroup(
        name="Biomarkers",
        columns=[
            "hdl_mgdl", "total_cholesterol_mgdl", "sbp_mmhg",
            "hba1c_percent", "bmi", "weight_kg"
        ],
        description="Annual biomarker measurements"
    )

    RISK_COLS = ColumnGroup(
        name="Disease Risks",
        columns=[
            "cvd_risk_10year", "diabetes_risk_10year", "stroke_risk_10year",
            "nafld_risk_10year", "colorectal_cancer_risk_10year", "cirrhosis_risk_10year"
        ],
        description="10-year disease risk probabilities"
    )

    OUTCOME_COLS = ColumnGroup(
        name="Disease Outcomes",
        columns=[
            "has_cvd", "has_diabetes", "has_stroke",
            "has_nafld", "has_colorectal_cancer", "has_cirrhosis"
        ],
        description="Binary disease outcomes (0/1)"
    )

    def __init__(self):
        self.version = get_version()
        self.data_path = get_data_path(self.version)
        self.output_dir = get_output_dir() / "statistics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets: Dict[str, pd.DataFrame] = {}
        print(f"Version: {self.version}")
        print(f"Data path: {self.data_path}")
        print(f"Output dir: {self.output_dir}")

    def load_all(self) -> bool:
        print("\nLOADING DATASETS")
        loaders = {
            "01_cohort_baseline": "01_cohort_baseline",
            "02_lifestyle_history": "02_lifestyle_history",
            "03_biomarkers_history": "03_biomarkers_history",
            "04_health_risks": "04_health_risks",
            "05_aggregated_dataset_with_missing": "05_aggregated_dataset_with_missing",
        }
        for key, pattern in loaders.items():
            files = list(self.data_path.glob(f"{pattern}_v*.csv"))
            if not files:
                print(f"✗ {key}: no file found")
                continue
            try:
                self.datasets[key] = pd.read_csv(files[0])
                print(f"✓ {key}: {self.datasets[key].shape}")
            except Exception as e:
                print(f"✗ {key}: FAILED - {e}")
                return False
        return len(self.datasets) > 0

    def _calculate_stats(self, data: np.ndarray, col_name: str) -> Dict:
        if not np.issubdtype(data.dtype, np.number):
            return {'column': col_name, 'count': 0, 'mean': np.nan}

        valid = data[~np.isnan(data)]
        if len(valid) == 0:
            return {'column': col_name, 'count': 0, 'mean': np.nan}

        stats = {
            'column': col_name,
            'count': len(valid),
            'missing': len(data) - len(valid),
            'missing_pct': (len(data) - len(valid)) / len(data) * 100,
            'mean': np.mean(valid),
            'std': np.std(valid),
            'min': np.min(valid),
            'max': np.max(valid),
            'median': np.median(valid),
        }

        for p in PERCENTILES:
            stats[f'p{int(p * 100)}'] = np.percentile(valid, p * 100)

        return stats

    def _get_yearly_cols(self, df: pd.DataFrame, prefix: str) -> List[str]:
        pattern = re.compile(rf'^{prefix}_\d+$')
        cols = [c for c in df.columns if pattern.match(c)]
        return sorted(cols, key=lambda x: int(x.split('_')[-1]))

    def _generate_stats(self, df: pd.DataFrame, groups: List[ColumnGroup],
                        dataset_name: str, by_year: bool = False) -> pd.DataFrame:
        all_stats = []
        for group in groups:
            for col in group.columns:
                if by_year:
                    yearly_cols = self._get_yearly_cols(df, col)
                    for ycol in yearly_cols:
                        year = int(ycol.split('_')[-1])
                        stats = self._calculate_stats(df[ycol].values, ycol)
                        stats['dataset'] = dataset_name
                        stats['group'] = group.name
                        stats['year'] = year
                        all_stats.append(stats)
                else:
                    if col not in df.columns:
                        continue
                    stats = self._calculate_stats(df[col].values, col)
                    stats['dataset'] = dataset_name
                    stats['group'] = group.name
                    stats['year'] = 'N/A'
                    all_stats.append(stats)
        return pd.DataFrame(all_stats)

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        print("\nGENERATING STATISTICS")
        results = {}
        print("[1/5] 01_cohort_baseline...")
        results["01_cohort_baseline"] = self._generate_stats(
            self.datasets["01_cohort_baseline"],
            [self.DEMOGRAPHIC_COLS, self.GENETIC_COLS, self.INITIAL_BIOMARKER_COLS],
            "01_cohort_baseline"
        )
        print("[2/5] 02_lifestyle_history...")
        results["02_lifestyle_history"] = self._generate_stats(
            self.datasets["02_lifestyle_history"],
            [self.LIFESTYLE_COLS],
            "02_lifestyle_history",
            by_year=True
        )
        print("[3/5] 03_biomarkers_history...")
        results["03_biomarkers_history"] = self._generate_stats(
            self.datasets["03_biomarkers_history"],
            [self.BIOMARKER_COLS],
            "03_biomarkers_history",
            by_year=True
        )
        print("[4/5] 04_health_risks...")
        results["04_health_risks"] = self._generate_stats(
            self.datasets["04_health_risks"],
            [self.RISK_COLS, self.OUTCOME_COLS],
            "04_health_risks"
        )
        print("[5/5] 05_aggregated_dataset_with_missing...")
        results["05_aggregated_dataset_with_missing"] = self._generate_stats(
            self.datasets["05_aggregated_dataset_with_missing"],
            [self.DEMOGRAPHIC_COLS, self.BIOMARKER_COLS, self.RISK_COLS, self.OUTCOME_COLS],
            "05_aggregated_dataset_with_missing"
        )
        return results

    def save_all(self, results: Dict[str, pd.DataFrame]) -> None:
        print("\nSAVING STATISTICS")
        combined = pd.concat(results.values(), ignore_index=True)
        for col in combined.select_dtypes(include=[np.number]).columns:
            combined[col] = combined[col].round(FLOAT_PRECISION)
        out_path = self.output_dir / "all_statistics.csv"
        combined.to_csv(out_path, index=False)
        print(f"✓ Saved: {out_path} ({len(combined)} rows)")


def main():
    generator = StatisticsGenerator()
    if not generator.load_all():
        sys.exit(1)
    results = generator.generate_all()
    generator.save_all(results)
    print("\nCOMPLETE!")


if __name__ == "__main__":
    main()
