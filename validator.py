#!/usr/bin/env python3
"""
SyntheticHealthSimulator Dataset Validator
"""

import json
import re
import sys
import warnings
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.request import urlopen

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class ValidationResult:
    """Result of a single validation test"""

    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None
    severity: str = "error"  # error, warning, info

    def __str__(self) -> str:
        if self.passed:
            status = "PASS"
        elif self.severity == "warning":
            status = "WARN"
        else:
            status = "FAIL"
        return f"{status} {self.name}: {self.message}"


class DatasetValidator:
    """
    Validator for SyntheticHealthSimulator datasets.
    Supports loading from a local directory or a GitHub release.
    """

    # Expected files in the dataset
    EXPECTED_FILES = [
        "01_cohort_baseline",
        "02_lifestyle_history",
        "03_biomarkers_history",
        "04_health_risks",
        "05_aggregated_dataset_with_missing",
    ]

    # Physiological ranges (from scientific reference, section 8.3)
    BIOMARKER_RANGES = {
        "hdl_mgdl": (20, 100),
        "total_cholesterol_mgdl": (150, 350),
        "sbp_mmhg": (80, 200),
        "hba1c_percent": (3.5, 12.0),
        "weight_kg": (40, 200),
        "bmi": (16, 50),
        "non_hdl_mgdl": (50, 232),
        "non_hdl_mmol": (2.0, 6.0),
        "muscle_mass_factor": (0, 20),
    }

    # Expected correlations (tolerances based on medical literature)
    EXPECTED_CORRELATIONS = {
        ("hdl_mgdl", "total_cholesterol_mgdl"): (-0.2, 0.2),
        ("sbp_mmhg", "bmi"): (0.2, 0.7),
        ("hba1c_percent", "bmi"): (0.1, 0.5),
        ("hdl_mgdl", "bmi"): (-0.5, 0.1),
    }

    # Target disease prevalence (realistic ranges)
    TARGET_PREVALENCE = {
        "has_diabetes": (0.10, 0.30),
        "has_cvd": (0.10, 0.30),
        "has_stroke": (0.10, 0.26),
    }

    # Expected demographic parameters
    DEMOGRAPHIC_PARAMS = {
        "age": {"mean": 35, "std": 8, "min": 20, "max": 50},
        "height_male": {"mean": 175, "std": 7, "min": 150, "max": 200},
        "height_female": {"mean": 162, "std": 6, "min": 135, "max": 190, "std_min": 5.5},
        "sex_ratio_male": 0.48,
    }

    # Genetic risk parameters
    GENETIC_RISK_PARAMS = {
        "genetic_risk_diabetes": {"std": 0.3, "min": 0.5, "max": 2.0, "mean_target": 1.0},
        "genetic_risk_cvd": {"std": 0.4, "min": 0.5, "max": 2.0, "mean_target": 1.0},
        "genetic_risk_cancer": {"std": 0.35, "min": 0.5, "max": 2.0, "mean_target": 1.0},
        "genetic_risk_stroke": {"std": 0.3, "min": 0.5, "max": 2.0, "mean_target": 1.0},
        "genetic_risk_nafld": {"std": 0.25, "min": 0.5, "max": 2.0, "mean_target": 1.0},
        "genetic_risk_cirrhosis": {"std": 0.4, "min": 0.5, "max": 2.0, "mean_target": 1.0},
    }

    # Expected columns in the aggregated dataset
    AGGREGATED_EXPECTED_COLUMNS = [
        "person_id", "age_start", "sex", "height_cm", "weight_start_kg", "bmi_start",
        "final_bmi", "final_hdl_mgdl", "final_total_cholesterol_mgdl", "final_sbp_mmhg",
        "final_hba1c_percent", "avg_alcohol_g_per_week", "avg_total_met_minutes",
        "cvd_risk_10year", "diabetes_risk_10year", "has_cvd", "has_diabetes",
        "age_end"
    ]

    def __init__(
            self,
            data_path: Optional[Union[str, Path]] = None,
            github_repo: Optional[str] = None,
            version: Optional[str] = None,
    ):
        self.data_path = Path(data_path) if data_path else None
        self.github_repo = github_repo
        self.version = version
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.results: List[ValidationResult] = []
        self.metadata: Optional[Dict] = None

    def load_data(self) -> bool:
        if self.data_path and self.data_path.exists():
            return self._load_local()
        elif self.github_repo and self.version:
            return self._load_from_github()
        else:
            print("No data source specified. Use --local or --github with --version")
            return False

    def _load_local(self) -> bool:
        print(f"\nLoading from local directory: {self.data_path}")
        csv_files = list(self.data_path.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {self.data_path}")
            return False

        for file_path in csv_files:
            name = file_path.stem
            base_name = re.sub(r"_v\d+\.\d+\.\d+$", "", name)
            try:
                self.datasets[base_name] = pd.read_csv(file_path)
                print(f"Loaded: {base_name} ({len(self.datasets[base_name])} rows)")
            except Exception as e:
                print(f"Error loading {name}: {e}")

        metadata_path = self.data_path / f"05_aggregated_dataset_with_missing_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
                print(f"  Metadata loaded (version: {self.metadata.get('version', 'unknown')})")
        return len(self.datasets) > 0

    def _load_from_github(self) -> bool:
        release_url = f"https://github.com/{self.github_repo}/releases/download/{self.version}/synthetic_data_{self.version}.zip"
        print(f"\nDownloading release: {release_url}")
        try:
            response = urlopen(release_url, timeout=60)
            zip_data = BytesIO(response.read())
            with zipfile.ZipFile(zip_data) as zf:
                for name in zf.namelist():
                    if name.endswith(".csv"):
                        key = Path(name).stem
                        base_key = re.sub(r"_v\d+\.\d+\.\d+$", "", key)
                        with zf.open(name) as f:
                            self.datasets[base_key] = pd.read_csv(f)
                            print(f"Loaded: {base_key} ({len(self.datasets[base_key])} rows)")
                    elif name.endswith("_metadata.json"):
                        with zf.open(name) as f:
                            self.metadata = json.load(f)
                            print(f"Metadata loaded")
            return len(self.datasets) > 0
        except Exception as e:
            print(f"Error downloading from GitHub: {e}")
            return False

    def validate_all(self) -> List[ValidationResult]:
        if not self.datasets:
            print("No data to validate")
            return []
        print("\nSTARTING VALIDATION ...")

        # General checks for all datasets
        self._check_duplicates()

        self._validate_file_structure()
        if "01_cohort_baseline" in self.datasets:
            self._validate_cohort_baseline()
        if "02_lifestyle_history" in self.datasets:
            self._validate_lifestyle_history()
        if "03_biomarkers_history" in self.datasets:
            self._validate_biomarkers()
            self._check_biomarker_trends()
        if "04_health_risks" in self.datasets:
            self._validate_risks()
        if "05_aggregated_dataset_with_missing" in self.datasets:
            self._validate_aggregated()
            self._check_aggregated_columns()
        self._validate_cross_dataset_consistency()
        return self.results

    def _check_duplicates(self):
        """Check for duplicate rows and uniqueness of person_id in aggregated datasets."""
        for name, df in self.datasets.items():
            n_duplicates = df.duplicated().sum()
            if n_duplicates > 0:
                self.results.append(
                    ValidationResult(
                        f"{name}: Duplicate rows",
                        False,
                        f"Found {n_duplicates} duplicate rows",
                        {"duplicates": n_duplicates},
                        "warning" if n_duplicates < 10 else "error",
                    )
                )
            else:
                self.results.append(
                    ValidationResult(
                        f"{name}: No duplicate rows",
                        True,
                        "All rows are unique",
                        severity="info",
                    )
                )

            if name in ["01_cohort_baseline", "04_health_risks", "05_aggregated_dataset_with_missing"]:
                if "person_id" in df.columns:
                    n_unique = df["person_id"].nunique()
                    if n_unique != len(df):
                        self.results.append(
                            ValidationResult(
                                f"{name}: Unique person_id",
                                False,
                                f"Expected {len(df)} unique IDs, got {n_unique}",
                                {"n_unique": n_unique, "n_rows": len(df)},
                                "error",
                            )
                        )
                    else:
                        self.results.append(
                            ValidationResult(
                                f"{name}: Unique person_id",
                                True,
                                "All person_id are unique",
                                severity="info",
                            )
                        )

    def _validate_file_structure(self):
        found_files = set(self.datasets.keys())
        expected_set = set(self.EXPECTED_FILES)
        missing = expected_set - found_files
        extra = found_files - expected_set
        if missing:
            self.results.append(
                ValidationResult(
                    "File Structure",
                    False,
                    f"Missing files: {missing}",
                    {"missing": list(missing)},
                    "warning" if len(missing) < 3 else "error",
                )
            )
        else:
            self.results.append(
                ValidationResult("File Structure", True, f"All {len(expected_set)} files present")
            )
        if extra:
            self.results.append(
                ValidationResult("Extra Files", True, f"Additional files: {extra}",
                                 {"extra": list(extra)}, "info")
            )

    def _validate_cohort_baseline(self):
        df = self.datasets["01_cohort_baseline"]
        required = ["person_id", "age_start", "sex", "height_cm", "weight_start_kg",
                    "genetic_risk_cvd", "genetic_risk_diabetes"]
        missing_cols = set(required) - set(df.columns)
        if missing_cols:
            self.results.append(
                ValidationResult("Cohort: Required Columns", False, f"Missing columns: {missing_cols}")
            )
            return

        age = df["age_start"]
        age_mean = age.mean()
        age_std = age.std()
        age_min = age.min()
        age_max = age.max()
        age_mean_ok = 30 <= age_mean <= 40
        age_std_ok = 6 <= age_std <= 10
        age_range_ok = (age_min >= self.DEMOGRAPHIC_PARAMS["age"]["min"] and
                        age_max <= self.DEMOGRAPHIC_PARAMS["age"]["max"])
        age_passed = age_mean_ok and age_std_ok and age_range_ok
        self.results.append(
            ValidationResult(
                "Cohort: Age Distribution",
                age_passed,
                f"Mean={age_mean:.1f}, Std={age_std:.1f}, Range=[{age_min},{age_max}]",
                {"mean": age_mean, "std": age_std, "min": age_min, "max": age_max},
                "warning" if not age_passed else "info",
            )
        )

        sex_counts = df["sex"].value_counts(normalize=True)
        male_pct = sex_counts.get("M", 0)
        female_pct = sex_counts.get("F", 0)
        sex_balanced = 0.45 <= male_pct <= 0.55
        self.results.append(
            ValidationResult(
                "Cohort: Sex Balance",
                sex_balanced,
                f"Male: {male_pct:.1%}, Female: {female_pct:.1%}",
                {"male_pct": male_pct, "female_pct": female_pct},
            )
        )

        male_height = df[df["sex"] == "M"]["height_cm"]
        female_height = df[df["sex"] == "F"]["height_cm"]
        if len(male_height) > 0:
            male_mean = male_height.mean()
            male_std = male_height.std()
            male_mean_ok = (self.DEMOGRAPHIC_PARAMS["height_male"]["mean"] - 7 <= male_mean <=
                            self.DEMOGRAPHIC_PARAMS["height_male"]["mean"] + 7)
            male_std_ok = male_std <= 9
            male_range_ok = (male_height.min() >= self.DEMOGRAPHIC_PARAMS["height_male"]["min"] and
                             male_height.max() <= self.DEMOGRAPHIC_PARAMS["height_male"]["max"])
            male_passed = male_mean_ok and male_std_ok and male_range_ok
            self.results.append(
                ValidationResult(
                    "Cohort: Male Height",
                    male_passed,
                    f"Mean={male_mean:.1f}, Std={male_std:.1f}",
                    {"mean": male_mean, "std": male_std},
                    "warning" if not male_passed else "info",
                )
            )
        if len(female_height) > 0:
            female_mean = female_height.mean()
            female_std = female_height.std()
            female_mean_ok = (self.DEMOGRAPHIC_PARAMS["height_female"]["mean"] - 6 <= female_mean <=
                              self.DEMOGRAPHIC_PARAMS["height_female"]["mean"] + 6)
            female_std_ok = female_std >= self.DEMOGRAPHIC_PARAMS["height_female"].get("std_min", 0)
            female_range_ok = (female_height.min() >= self.DEMOGRAPHIC_PARAMS["height_female"]["min"] and
                               female_height.max() <= self.DEMOGRAPHIC_PARAMS["height_female"]["max"])
            female_passed = female_mean_ok and female_std_ok and female_range_ok
            self.results.append(
                ValidationResult(
                    "Cohort: Female Height",
                    female_passed,
                    f"Mean={female_mean:.1f}, Std={female_std:.1f}",
                    {"mean": female_mean, "std": female_std},
                    "warning" if not female_passed else "info",
                )
            )

        for col, params in self.GENETIC_RISK_PARAMS.items():
            if col in df.columns:
                vals = df[col]
                in_range = ((vals >= params["min"]) & (vals <= params["max"])).mean()
                mean_val = vals.mean()
                std_val = vals.std()
                mean_ok = 0.9 <= mean_val <= 1.1
                std_ok = abs(std_val - params["std"]) <= 0.15
                range_ok = in_range > 0.98
                passed = mean_ok and std_ok and range_ok
                self.results.append(
                    ValidationResult(
                        f"Cohort: {col}",
                        passed,
                        f"Mean={mean_val:.3f}, Std={std_val:.3f}, {in_range:.1%} in range",
                        {"mean": mean_val, "std": std_val, "in_range": in_range},
                        "warning" if not passed else "info",
                    )
                )

        for marker, (low, high) in self.BIOMARKER_RANGES.items():
            col_map = {
                "hdl_mgdl": "initial_hdl_mgdl",
                "total_cholesterol_mgdl": "initial_tc_mgdl",
                "hba1c_percent": "initial_hba1c_percent",
                "sbp_mmhg": "initial_sbp_mmhg",
                "bmi": "bmi_start",
            }
            if marker in col_map and col_map[marker] in df.columns:
                vals = df[col_map[marker]]
                in_range = ((vals >= low) & (vals <= high)).mean()
                passed = in_range > 0.99
                self.results.append(
                    ValidationResult(
                        f"Cohort: Initial {marker}",
                        passed,
                        f"{in_range:.1%} in [{low}, {high}]",
                        {"in_range": in_range},
                        "warning" if not passed else "info",
                    )
                )

    def _validate_lifestyle_history(self):
        df = self.datasets["02_lifestyle_history"]

        pattern = re.compile(r"^alcohol_g_per_week_(\d+)$")
        year_cols = [c for c in df.columns if pattern.match(c)]
        n_years = len(year_cols)
        if n_years > 0:
            panel_ok = 18 <= n_years <= 22
            self.results.append(
                ValidationResult(
                    "Lifestyle: Panel Structure",
                    panel_ok,
                    f"Found {n_years} yearly columns (expected 20)",
                    {"n_years": n_years},
                    "warning" if not panel_ok else "info",
                )
            )
        else:
            records_per_person = df.groupby("person_id").size()
            years_coverage = records_per_person.mean()
            panel_ok = 19 <= years_coverage <= 21
            self.results.append(
                ValidationResult(
                    "Lifestyle: Panel Structure (long format)",
                    panel_ok,
                    f"Average {years_coverage:.1f} records per patient (expected 20)",
                    {"mean_years": years_coverage},
                    "warning" if not panel_ok else "info",
                )
            )

        if "alcohol_g_per_week_0" in df.columns and "alcohol_g_per_week_1" in df.columns:
            corr = df["alcohol_g_per_week_0"].corr(df["alcohol_g_per_week_1"])
            autocorr_ok = 0.6 <= corr <= 1.0
            if corr > 0.98:
                severity = "info"
                message = f"Correlation: {corr:.3f} (expected ~0.82)"
            else:
                severity = "info" if autocorr_ok else "warning"
                message = f"Correlation: {corr:.3f} (expected ~0.82)"
            self.results.append(
                ValidationResult(
                    "Lifestyle: OU Autocorrelation (alcohol year0-year1)",
                    autocorr_ok,
                    message,
                    {"correlation": corr},
                    severity,
                )
            )
        else:
            self.results.append(
                ValidationResult(
                    "Lifestyle: OU Autocorrelation",
                    False,
                    "Columns alcohol_g_per_week_0 / _1 not found",
                    severity="warning",
                )
            )

    def _validate_biomarkers(self):
        df = self.datasets["03_biomarkers_history"]
        violations = []
        for marker, (low, high) in self.BIOMARKER_RANGES.items():
            pattern = rf"^{marker}_\d+$"
            cols = [c for c in df.columns if re.match(pattern, c)]
            for col in cols[:3]:
                if col in df.columns:
                    below = (df[col] < low).sum()
                    above = (df[col] > high).sum()
                    if below > 0 or above > 0:
                        violations.append(f"{col}: {below} below {low}, {above} above {high}")
        if violations:
            self.results.append(
                ValidationResult(
                    "Biomarkers: Hard Clipping",
                    False,
                    f"Constraint violations: {len(violations)} cases",
                    {"violations": violations[:5]},
                    "error",
                )
            )
        else:
            self.results.append(
                ValidationResult(
                    "Biomarkers: Hard Clipping", True, "All values within physiological ranges"
                )
            )

        if "hdl_mgdl_0" in df.columns and "hdl_mgdl_1" in df.columns:
            hdl_change = (df["hdl_mgdl_1"] - df["hdl_mgdl_0"]).abs()
            noise_present = hdl_change.std() > 1.0
            self.results.append(
                ValidationResult(
                    "Biomarkers: Measurement Noise",
                    noise_present,
                    f"HDL change std: {hdl_change.std():.2f} mg/dL",
                    {"noise_std": hdl_change.std()},
                    "warning" if not noise_present else "info",
                )
            )

    def _check_biomarker_trends(self):
        """Check expected temporal trends for key biomarkers."""
        df = self.datasets["03_biomarkers_history"]

        markers = ["hba1c_percent", "hdl_mgdl", "sbp_mmhg", "bmi"]
        trends = {
            "hba1c_percent": {"expected": "increase", "median_min": 0},
            "hdl_mgdl": {"expected": "decrease", "median_max": 0},
            "sbp_mmhg": {"expected": "increase", "median_min": 0},
            "bmi": {"expected": "increase", "median_min": -0.5},
        }

        for marker, spec in trends.items():
            cols = [c for c in df.columns if re.match(rf"^{marker}_\d+$", c)]
            if len(cols) < 2:
                self.results.append(
                    ValidationResult(
                        f"Biomarker trend: {marker}",
                        False,
                        f"Not enough yearly columns found ({len(cols)})",
                        severity="warning",
                    )
                )
                continue

            years = sorted([int(c.split("_")[-1]) for c in cols])
            first_year = years[0]
            last_year = years[-1]
            col_first = f"{marker}_{first_year}"
            col_last = f"{marker}_{last_year}"

            if col_first not in df.columns or col_last not in df.columns:
                continue

            change = df[col_last] - df[col_first]
            median_change = change.median()

            if spec["expected"] == "increase":
                passed = median_change > spec.get("median_min", 0)
                message = f"Median change from year {first_year} to {last_year}: {median_change:.3f} (expected increase)"
            else:
                passed = median_change < spec.get("median_max", 0)
                message = f"Median change from year {first_year} to {last_year}: {median_change:.3f} (expected decrease)"

            self.results.append(
                ValidationResult(
                    f"Biomarker trend: {marker}",
                    passed,
                    message,
                    {"median_change": median_change, "first_year": first_year, "last_year": last_year},
                    "warning" if not passed else "info",
                )
            )

    def _validate_risks(self):
        df = self.datasets["04_health_risks"]
        for disease, (min_prev, max_prev) in self.TARGET_PREVALENCE.items():
            if disease in df.columns:
                prevalence = df[disease].mean()
                in_range = min_prev <= prevalence <= max_prev
                severity = (
                    "error" if prevalence < 0.05 or prevalence > 0.50 else
                    "warning" if not in_range else "info"
                )
                self.results.append(
                    ValidationResult(
                        f"Risks: {disease} Prevalence",
                        in_range,
                        f"{prevalence:.1%} (target {min_prev:.0%}-{max_prev:.0%})",
                        {"prevalence": prevalence},
                        severity,
                    )
                )

        if "cvd_risk_10year" in df.columns and "has_cvd" in df.columns:
            corr = df["cvd_risk_10year"].corr(df["has_cvd"])
            calib_ok = corr > 0.1
            self.results.append(
                ValidationResult(
                    "Risks: Risk-Outcome Correlation",
                    calib_ok,
                    f"Correlation between risk and outcome: {corr:.3f}",
                    {"correlation": corr},
                    "warning" if not calib_ok else "info",
                )
            )

    def _validate_aggregated(self):
        df = self.datasets["05_aggregated_dataset_with_missing"]

        missing_cols = [
            "final_sbp_mmhg",
            "final_hdl_mgdl",
            "final_total_cholesterol_mgdl",
            "final_hba1c_percent",
            "avg_alcohol_g_per_week",
            "avg_total_met_minutes",
        ]
        for col in missing_cols:
            if col in df.columns:
                missing_pct = df[col].isna().mean()
                in_range = 0.02 <= missing_pct <= 0.15
                self.results.append(
                    ValidationResult(
                        f"Aggregated: MAR {col}",
                        in_range,
                        f"{missing_pct:.1%} missing (target 2-15%)",
                        {"missing_pct": missing_pct},
                        "warning" if not in_range and missing_pct > 0 else "info",
                    )
                )

        if "age_start" in df.columns and "cvd_risk_10year" in df.columns:
            young = df[df["age_start"] < 30]
            old = df[df["age_start"] >= 30]
            if "final_sbp_mmhg" in df.columns:
                young_missing = young["final_sbp_mmhg"].isna().mean() if len(young) > 0 else 0
                old_missing = old["final_sbp_mmhg"].isna().mean() if len(old) > 0 else 0
                mar_pattern = young_missing > old_missing
                self.results.append(
                    ValidationResult(
                        "Aggregated: MAR Pattern (Age)",
                        mar_pattern,
                        f"Young (<30): {young_missing:.1%}, Old: {old_missing:.1%} missing",
                        {"young_missing": young_missing, "old_missing": old_missing},
                        "warning" if not mar_pattern else "info",
                    )
                )
        else:
            self.results.append(
                ValidationResult(
                    "Aggregated: MAR Pattern",
                    False,
                    "Required columns age_start or cvd_risk_10year missing",
                    severity="warning",
                )
            )

        for (col1, col2), (min_corr, max_corr) in self.EXPECTED_CORRELATIONS.items():
            c1 = f"final_{col1}" if f"final_{col1}" in df.columns else None
            c2 = f"final_{col2}" if f"final_{col2}" in df.columns else None
            if c1 and c2 and c1 in df.columns and c2 in df.columns:
                valid = df[[c1, c2]].dropna()
                if len(valid) > 100:
                    corr = valid[c1].corr(valid[c2])
                    in_range = min_corr <= corr <= max_corr
                    self.results.append(
                        ValidationResult(
                            f"Aggregated: Correlation {c1}-{c2}",
                            in_range,
                            f"r={corr:.3f} (expected {min_corr}..{max_corr})",
                            {"correlation": corr, "n_valid": len(valid)},
                            "warning" if not in_range else "info",
                        )
                    )

    def _check_aggregated_columns(self):
        """Check presence of all key columns in the aggregated dataset."""
        df = self.datasets["05_aggregated_dataset_with_missing"]
        missing = [col for col in self.AGGREGATED_EXPECTED_COLUMNS if col not in df.columns]
        if missing:
            self.results.append(
                ValidationResult(
                    "Aggregated: Required Columns",
                    False,
                    f"Missing columns: {missing}",
                    {"missing": missing},
                    "error",
                )
            )
        else:
            self.results.append(
                ValidationResult(
                    "Aggregated: Required Columns",
                    True,
                    f"All {len(self.AGGREGATED_EXPECTED_COLUMNS)} expected columns present",
                    severity="info",
                )
            )

    def _validate_cross_dataset_consistency(self):
        ids_by_dataset = {
            name: set(df["person_id"].unique())
            for name, df in self.datasets.items()
            if "person_id" in df.columns
        }
        if len(ids_by_dataset) >= 2:
            common_ids = set.intersection(*ids_by_dataset.values())
            all_ids = set.union(*ids_by_dataset.values())
            consistency = len(common_ids) == len(all_ids)
            self.results.append(
                ValidationResult(
                    "Cross-Dataset: Person ID Consistency",
                    consistency,
                    f"{len(common_ids)} common IDs out of {len(all_ids)} unique",
                    {"common": len(common_ids), "total_unique": len(all_ids)},
                    "error" if not consistency else "info",
                )
            )

        if ("05_aggregated_dataset_with_missing" in self.datasets and
                "03_biomarkers_history" in self.datasets):
            agg = self.datasets["05_aggregated_dataset_with_missing"]
            bio = self.datasets["03_biomarkers_history"]
            bmi_cols = [c for c in bio.columns if c.startswith("bmi_") and c[4:].isdigit()]
            if bmi_cols:
                last_year = max([int(c.split("_")[1]) for c in bmi_cols])
                last_year_col = f"bmi_{last_year}"
                if last_year_col in bio.columns:
                    last_bmi = bio.groupby("person_id")[last_year_col].last()
                    if "final_bmi" in agg.columns:
                        merged = agg[["person_id", "final_bmi"]].merge(
                            last_bmi.reset_index(), on="person_id"
                        )
                        diff = (merged["final_bmi"] - merged[last_year_col]).abs().mean()
                        bmi_consistent = diff < 0.5
                        self.results.append(
                            ValidationResult(
                                "Cross-Dataset: BMI Consistency",
                                bmi_consistent,
                                f"Mean final_bmi difference: {diff:.3f} kg/m²",
                                {"mean_diff": diff},
                                "warning" if not bmi_consistent else "info",
                            )
                        )

    def print_report(self):
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        warnings_count = sum(1 for r in self.results if not r.passed and r.severity == "warning")
        errors = sum(1 for r in self.results if not r.passed and r.severity == "error")
        print(f"Total tests: {total}")
        print(f"  PASSED: {passed}")
        print(f"  WARNINGS: {warnings_count}")
        print(f"  FAILED: {errors}")
        print("\nDETAILED RESULTS:")
        for i, result in enumerate(self.results, 1):
            print(f"{i}. {result}")
        if errors == 0:
            if warnings_count == 0:
                print("\nALL CHECKS PASSED SUCCESSFULLY!\n")
            else:
                print("\nCHECKS PASSED WITH MINOR WARNINGS\n")
        else:
            print("\nCRITICAL ERRORS DETECTED\n")
        return errors == 0

    def export_report(self, filepath: str):

        def default_serializer(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "source": str(self.data_path) if self.data_path else f"github:{self.github_repo}:{self.version}",
            "metadata": self.metadata,
            "summary": {
                "total_tests": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "warnings": sum(1 for r in self.results if not r.passed and r.severity == "warning"),
                "errors": sum(1 for r in self.results if not r.passed and r.severity == "error"),
            },
            "details": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                    "details": r.details,
                }
                for r in self.results
            ],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=default_serializer)
        print(f"\nReport saved: {filepath}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SyntheticHealthSimulator Dataset Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validator.py --local data/synthetic_v1.0.2
  python validator.py --github kavalex/SyntheticHealthSimulator --version v1.0.2 --report report.json
        """,
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--local", "-l", type=str, help="Path to local directory")
    source_group.add_argument("--github", "-g", type=str, help="GitHub repo (owner/repo)")
    parser.add_argument("--version", "-v", type=str, help="Version tag (required with --github)")
    parser.add_argument("--report", "-r", type=str, help="Path to save JSON report")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (errors only)")
    args = parser.parse_args()
    if args.github and not args.version:
        parser.error("--github requires --version")
    validator = DatasetValidator(data_path=args.local, github_repo=args.github, version=args.version)
    if not validator.load_data():
        sys.exit(1)
    validator.validate_all()
    success = validator.print_report()
    if args.report:
        validator.export_report(args.report)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
