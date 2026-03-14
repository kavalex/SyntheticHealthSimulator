"""
Common validation functions and DatasetValidator class for synthetic data quality checks.
Consolidates logic from scripts/validator.py and utils/validators.py.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# Attempt to import config, handle case if running in isolated environment
try:
    from src.utils.paths import get_version, get_data_path, get_output_dir
    from src.config import DEFAULT_CONFIG
except ImportError:
    # Fallbacks for standalone execution if necessary
    pass


@dataclass(frozen=True)
class ValidationConfig:
    EXPECTED_FILES: List[str] = field(default_factory=lambda: [
        "01_cohort_baseline", "02_lifestyle_history", "03_biomarkers_history",
        "04_health_risks", "05_aggregated_dataset_with_missing"
    ])
    BIOMARKER_RANGES: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "hdl_mgdl": (20, 100),
        "total_cholesterol_mgdl": (120, 350),
        "sbp_mmhg": (80, 200),
        "hba1c_percent": (3.5, 12.0),
        "weight_kg": (40, 160),
        "bmi": (16, 50),
    })
    TARGET_PREVALENCE: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "has_diabetes": (0.08, 0.30),
        "has_cvd": (0.10, 0.30),
        "has_stroke": (0.05, 0.26),
        "has_nafld": (0.10, 0.30),
        "has_colorectal_cancer": (0.05, 0.25),
        "has_cirrhosis": (0.01, 0.20),
    })
    MAR_COLUMNS: List[str] = field(default_factory=lambda: [
        "final_sbp_mmhg", "final_hdl_mgdl", "final_total_cholesterol_mgdl",
        "final_hba1c_percent", "avg_alcohol_g_per_week", "avg_total_met_minutes"
    ])


@dataclass
class ValidationResult:
    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None
    severity: str = "error"

    def __str__(self) -> str:
        status = "OK" if self.passed else ("WARN" if self.severity == "warning" else "FAIL")
        return f"{status:4} {self.name}: {self.message}"


class DatasetValidator:
    def __init__(self, data_path: Union[str, Path], config: Optional[ValidationConfig] = None):
        self.data_path = Path(data_path)
        self.config = config or ValidationConfig()
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.results: List[ValidationResult] = []

    def load_data(self) -> bool:
        print(f"Loading from: {self.data_path}")
        csv_files = list(self.data_path.glob("*.csv"))
        if not csv_files:
            print("No CSV files found")
            return False
        for file_path in csv_files:
            name = file_path.stem
            base_name = re.sub(r"_v\d+\.\d+\.\d+$", "", name)
            try:
                self.datasets[base_name] = pd.read_csv(file_path)
                print(f"  Loaded {base_name}: {self.datasets[base_name].shape}")
            except Exception as e:
                print(f"  Error loading {name}: {e}")
        return len(self.datasets) > 0

    def validate_all(self) -> List[ValidationResult]:
        self._check_duplicates()
        self._validate_cohort()
        self._validate_biomarkers()
        self._validate_risks()
        self._validate_aggregated()
        self._check_cross_consistency()
        return self.results

    def _add_result(self, name: str, passed: bool, message: str, details: Optional[Dict] = None, severity: str = None):
        if severity is None:
            severity = "error" if not passed else "info"
        self.results.append(ValidationResult(name, passed, message, details, severity))

    def _check_duplicates(self):
        for name, df in self.datasets.items():
            n_dup = df.duplicated().sum()
            if n_dup > 0:
                self._add_result(f"{name}: duplicates", False, f"{n_dup} duplicate rows", severity="warning")
            else:
                self._add_result(f"{name}: duplicates", True, "No duplicates")

    def _validate_cohort(self):
        df = self.datasets.get("01_cohort_baseline")
        if df is None:
            return
        required = ["person_id", "age_start", "sex", "height_cm"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            self._add_result("Cohort: columns", False, f"Missing: {missing}")
            return

        age = df["age_start"]
        # Use DEFAULT_CONFIG if available, else fallback to hardcoded
        try:
            age_range = DEFAULT_CONFIG.age_range
        except NameError:
            age_range = (25, 65)

        if age.min() < age_range[0] or age.max() > age_range[1]:
            self._add_result("Cohort: age range", False, f"Range [{age.min()}, {age.max()}]", severity="warning")
        else:
            self._add_result("Cohort: age range", True, f"Range [{age.min()}, {age.max()}]")

        male_pct = (df["sex"] == "M").mean()
        if 0.45 <= male_pct <= 0.55:
            self._add_result("Cohort: sex balance", True, f"Male {male_pct:.1%}")
        else:
            self._add_result("Cohort: sex balance", False, f"Male {male_pct:.1%}", severity="warning")

    def _validate_biomarkers(self):
        df = self.datasets.get("03_biomarkers_history")
        if df is None:
            return
        last_year = 19
        for marker, (low, high) in self.config.BIOMARKER_RANGES.items():
            col = f"{marker}_{last_year}"
            if col in df.columns:
                in_range = ((df[col] >= low) & (df[col] <= high)).mean()
                if in_range < 0.99:
                    self._add_result(f"Biomarkers: {marker} final", False,
                                     f"{in_range:.1%} in [{low}, {high}]", severity="warning")
                else:
                    self._add_result(f"Biomarkers: {marker} final", True,
                                     f"{in_range:.1%} in range")

    def _validate_risks(self):
        df = self.datasets.get("04_health_risks")
        if df is None:
            return
        for disease, (low, high) in self.config.TARGET_PREVALENCE.items():
            if disease in df.columns:
                prev = df[disease].mean()
                if low <= prev <= high:
                    self._add_result(f"Risks: {disease}", True, f"{prev:.1%} prevalence")
                else:
                    self._add_result(f"Risks: {disease}", False,
                                     f"{prev:.1%} (target {low:.0%}-{high:.0%})", severity="warning")

    def _validate_aggregated(self):
        df = self.datasets.get("05_aggregated_dataset_with_missing")
        if df is None:
            return
        for col in self.config.MAR_COLUMNS:
            if col in df.columns:
                missing = df[col].isna().mean()
                if 0.02 <= missing <= 0.15:
                    self._add_result(f"Aggregated: MAR {col}", True, f"{missing:.1%} missing")
                else:
                    self._add_result(f"Aggregated: MAR {col}", False,
                                     f"{missing:.1%} missing (expected 2-15%)", severity="warning")

    def _check_cross_consistency(self):
        ids_cohort = set(self.datasets.get("01_cohort_baseline", pd.DataFrame()).get("person_id", []))
        ids_agg = set(self.datasets.get("05_aggregated_dataset_with_missing", pd.DataFrame()).get("person_id", []))
        if ids_cohort and ids_agg:
            if ids_agg.issubset(ids_cohort):
                self._add_result("Cross: IDs consistent", True,
                                 f"{len(ids_agg)}/{len(ids_cohort)} retained")
            else:
                self._add_result("Cross: IDs consistent", False,
                                 f"Aggregated IDs not subset of cohort", severity="error")

    def print_report(self) -> None:
        print("\n" + "=" * 50)
        print("VALIDATION REPORT")
        print("=" * 50)
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        warnings = sum(1 for r in self.results if not r.passed and r.severity == "warning")
        errors = sum(1 for r in self.results if not r.passed and r.severity == "error")
        print(f"Total tests: {total}")
        print(f"  PASSED: {passed}")
        print(f"  WARNINGS: {warnings}")
        print(f"  ERRORS: {errors}")
        print("\nDetailed results:")
        for r in self.results:
            print(f"  {r}")


# --- Legacy helper functions (preserved) ---

def check_range(
        df: pd.DataFrame,
        column: str,
        low: float,
        high: float,
        threshold: float = 0.99
) -> Dict:
    if column not in df.columns:
        return {'status': 'MISSING', 'message': f'{column} not found'}

    in_range = ((df[column] >= low) & (df[column] <= high)).mean()
    passed = in_range >= threshold

    return {
        'column': column,
        'range': f'[{low}, {high}]',
        'in_range_pct': f'{in_range:.1%}',
        'status': 'OK' if passed else 'WARN',
        'passed': passed
    }


def check_correlation(
        df: pd.DataFrame,
        col1: str,
        col2: str,
        expected_range: Tuple[float, float]
) -> Dict:
    if col1 not in df.columns or col2 not in df.columns:
        return {'status': 'MISSING', 'message': 'Column(s) not found'}

    valid = df[[col1, col2]].dropna()
    if len(valid) < 100:
        return {'status': 'WARN', 'message': 'Insufficient data for correlation'}

    corr = valid[col1].corr(valid[col2])
    in_range = expected_range[0] <= corr <= expected_range[1]

    return {
        'columns': f'{col1} ↔ {col2}',
        'correlation': round(corr, 3),
        'expected': f'{expected_range[0]}..{expected_range[1]}',
        'status': 'OK' if in_range else 'WARN',
        'passed': in_range
    }


def check_prevalence(
        df: pd.DataFrame,
        column: str,
        target_range: Tuple[float, float]
) -> Dict:
    if column not in df.columns:
        return {'status': 'MISSING', 'message': f'{column} not found'}

    prevalence = df[column].mean()
    in_range = target_range[0] <= prevalence <= target_range[1]

    return {
        'column': column,
        'prevalence': f'{prevalence:.1%}',
        'target': f'{target_range[0]:.0%}-{target_range[1]:.0%}',
        'status': 'OK' if in_range else 'WARN',
        'passed': in_range
    }


def run_validation_checks(checks: List[Dict]) -> pd.DataFrame:
    results = []
    for check in checks:
        results.append(check)
    df = pd.DataFrame(results)
    return df


def print_validation_table(df: pd.DataFrame, title: str = "Validation Results"):
    print(f"\n### {title}")
    print(df.to_markdown(index=False))
