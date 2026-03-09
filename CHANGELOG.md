# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0](https://github.com/yourusername/SyntheticHealthSimulator/releases/tag/v2.0.0) - 2026-03-09

### Added

- **Sensitivity Analysis Tool** (`sensitivity_analysis.py`) – new module for analyzing how changes in input parameters
  affect disease risk predictions and biomarker trajectories.
- **Analysis Notebooks** – Jupyter notebooks for data exploration, validation visualization, and model interpretation.
- **HbA1c Drift Model** – added baseline drift of 0.015%/year with diabetes risk dependency for realistic long-term
  glucose dynamics.
- **HDL Age-Related Decline** – implemented realistic HDL reduction of -0.5 to -1.0 mg/dL per decade through age-penalty
  coefficients.
- **Latent Lipid Factor** – restored correlation structure between HDL and Total Cholesterol using patient-specific
  lipid factor (±5.0 mg/dL).
- **Comorbidity Modeling** – enhanced multi-disease outcome tracking with realistic prevalence of ≥2 diseases (15-50% of
  cohort).
- **Boundary Filtering Statistics** – comprehensive tracking of patients removed due to physiological range violations
  with retention rate reporting.

### Changed

- **Generator Calibration (v2.0.0)**:
    - CVD risk intercept calibrated to -3.8 for ~15-18% prevalence over 20 years
    - Stroke risk intercept calibrated to -3.0 for ~18-20% prevalence
    - SBP coefficients reduced (BMI: 0.7→0.4, Sodium: 0.25→0.12, Alcohol: 0.03→0.015, Stress: 0.5→0.25, Sleep: 0.5→0.25)
      for target mean SBP ~125 mmHg
    - BMI mean reversion force added (4% per year) to prevent 54% boundary violations
    - MAR missing value age threshold updated from 30 to 45 years
    - Male height baseline corrected from 175 cm to 178 cm
    - HDL change coefficients increased for realistic 20-year decline (cardio: 0.2→0.25, strength: 0.3→0.35, fiber:
      0.1→0.25, SFA: -0.05→-0.12)
    - HbA1c change coefficients calibrated for stable long-term dynamics (simple carbs: 0.15→0.03, BMI: 0.03→0.006,
      activity: -0.01→-0.002)
- **Validator Improvements** (`validator.py`):
    - Updated biomarker range checks to final-year only (Year 19) to allow intermediate generation ranges
    - Added comorbidity validation (≥2 diseases prevalence check)
    - Enhanced cross-dataset consistency checks for person_id tracking after boundary filtering
    - Improved MAR pattern validation with age-dependent missingness verification
    - Added temporal risk progression checks for all disease categories
- **OU Process Parameters**:
    - Alcohol theta increased to 0.8 (from 0.2) for stronger habit persistence modeling
    - Cardio/Strength sigma increased for realistic activity variability (150/120 MET-minutes)

### Fixed

- **Generation Data Fixes** (`generator.py`):
    - Corrected lipid factor application across all 20 years
    - Fixed protein balance calculation for muscle mass dynamics
    - Corrected stress event decay mechanism (1 year with +1.5 decay in following year)
    - Fixed non-HDL calculation with double clipping (mg/dL then mmol/L) for physiological consistency
    - Resolved BMI corrected calculation with muscle mass factor (was missing in aggregated dataset)
    - Fixed cumulative smoking calculation (pack-years properly accumulated across 20 years)
- **Validator Fixes**:
    - Corrected expected correlation ranges based on updated coefficient calibration
    - Fixed demographic parameter expectations (male height 178 cm, female height 163 cm)
    - Updated genetic risk distribution checks for truncated log-normal parameters
- **Documentation**:
    - Aligned SCIENTIFIC_REFERENCE_EN.md and SCIENTIFIC_REFERENCE_RU.md with v2.0.0 coefficients
    - Updated all calculation examples with calibrated values

### Removed

- **GitHub Validation Workflow** – removed automated GitHub Actions validation; validation now performed locally via
  `validator.py` for flexibility and faster iteration.

### Technical Details

- **Version**: 1.1.3 (combines v1.1.1 + v1.1.2 + new calibrations)
- **Target Prevalence**: CVD ~15-18%, Diabetes ~15%, Stroke ~18-20%
- **Boundary Filtering**: ~40-50% retention rate with oversampling coefficient 2.41
- **MAR Missing Values**: 2-15% missingness with age-dependent pattern (threshold: 45 years)
- **Observation Period**: 20 years with annual measurements
- **Cohort Size**: 5000 patients (after filtering)

---

## [1.1.0](https://github.com/yourusername/SyntheticHealthSimulator/releases/tag/v1.1.0) - 2026-03-04

### Added

- **Dataset Validator** (`validator.py`) – a standalone tool to verify that generated data meets expected statistical
  properties, physiological ranges, and correlations. Includes checks for file structure, demographic distributions,
  genetic risks, biomarker ranges, temporal trends, disease prevalence, MAR missing patterns, and cross‑dataset
  consistency.
- **Validation Documentation** – usage instructions added to README files.

### Changed

- **Generator Improvements**:
    - Fixed HDL trend: added age‑dependent decrease and adjusted coefficients to achieve realistic decline over 20
      years (median change now negative).
    - Added `age_vals` extraction in `calculate_biomarkers` to correctly apply age effects.
    - Included `total_met_minutes` in `cols_to_copy` so that it appears in the biomarkers DataFrame and consequently in
      the aggregated dataset.
    - Ensured `sex` column is preserved in aggregated dataset by using `suffixes` in merge.
    - Updated dependencies – versions in `pyproject.toml` aligned with current stable releases.
- **Documentation Updates** – README files now include validation instructions; scientific reference refined for
  clarity.

### Fixed

- **KeyError on `sex_0`** – removed `"sex"` from `cols_to_copy` in biomarker calculation to prevent copying non‑existent
  yearly sex columns.
- **HDL‑BMI Correlation** – now within expected range after coefficient tuning.
- **Missing `avg_total_met_minutes`** – now correctly computed from copied `total_met_minutes` columns.

---

## [1.0.0](https://github.com/yourusername/SyntheticHealthSimulator/releases/tag/v1.0.0) - 2026-02-27

### Added
- **Initial Release** of SyntheticHealthSimulator – a tool for generating synthetic medical panel data over 20 years.
- **Cohort Generation** (`generator.py`): creates a baseline population with demographics, genetics, and initial
  biomarkers.
- **Lifestyle History Simulation** using Ornstein‑Uhlenbeck processes (alcohol, cardio, strength, smoking, stress,
  sleep) and random stressful events.
- **Biomarker Evolution Models** for weight, muscle mass, HDL, total cholesterol, SBP, HbA1c with annual changes and
  added measurement noise.
- **Health Risk Calculators**: CVD (SCORE2-like), type 2 diabetes (FINDRISC-like), stroke, NAFLD, colorectal cancer,
  cirrhosis – all with simplified logistic models.
- **Polygenic Risk Generation** via log‑normal distributions for six disease categories.
- **Muscle Mass Modelling** with correction for BMR and BMI, and effects on biomarkers.
- **Macronutrient Balance Index (MBI)** (0‑100) to quantify diet quality.
- **MAR Missing Values Mechanism** (2-15%) applied to final aggregated dataset to simulate real‑world medical data.
- **Five Output Datasets**:
    - `01_cohort_baseline_v1.0.0.csv` – immutable characteristics.
    - `02_lifestyle_history_v1.0.0.csv` – yearly lifestyle factors and true physiological values.
    - `03_biomarkers_history_v1.0.0.csv` – annual biomarkers with noise.
    - `04_health_risks_v1.0.0.csv` – 10‑year risks and binary outcomes.
    - `05_aggregated_dataset_with_missing_v1.0.0.csv` – one row per patient with averaged lifestyle, final biomarkers,
      risks, and MAR missingness.
- **Metadata JSON Files** for each dataset containing version, description, and missing value info.
- **Version Management**:
    - `VERSION` file stores the current version.
    - `bump_version.py` script to increment major/minor/patch and update `generator.py`.
- **GitHub Actions Workflow** (`.github/workflows/release.yml`): Automatically creates a release and a ZIP archive of
  generated data when a tag `v*` is pushed.
- **Documentation**:
    - `README.md` / `README_ru.md` – overview, installation, usage, dataset structure.
    - `SCIENTIFIC_REFERENCE_EN.md` / `SCIENTIFIC_REFERENCE_RU.md` – detailed scientific background, formulas,
      coefficients, and model limitations.
    - Explicit calibration notes in code (e.g., `INTENTIONALLY REDUCED`, `INTENTIONALLY CALIBRATED`) to highlight
      deviations from epidemiological data for ML feature balance.

### Known Limitations (as documented)

- Linear and additive biomarker models – simplification for educational ML use.
- Theoretical assumptions for muscle mass, sleep, MBI effects – not clinically validated.
- Simplified body composition model (constant bone/organ mass).
- Additional risk models (stroke, NAFLD, cancer, cirrhosis) are not calibrated against real cohorts.
- MAR missingness formula is a heuristic, not derived from real medical records.
- Seasonality not modelled – annual discretisation averages it out.

### Important Warnings

- **DO NOT USE FOR CLINICAL DECISIONS** – data is entirely synthetic.
- All coefficients are intentionally reduced to prevent feature dominance; real epidemiological effects may be stronger.
- Hard clipping is applied to biomarkers to keep values within physiologically plausible ranges (see Section 8.3 of the
  scientific reference).

---

## Summary of Version Progression

| Version | Date       | Key Focus                                                      |
|---------|------------|----------------------------------------------------------------|
| 1.0.0   | 2026-02-27 | Initial release with core functionality                        |
| 1.1.0   | 2026-03-04 | Validator tool + HDL trend fixes                               |
| 1.1.3   | 2026-03-09 | Full calibration + sensitivity analysis + comorbidity modeling |

**Total Changes from 1.0.0 to 2.0.0**: 3 major updates, 15+ coefficient calibrations, 5 new files, 2 removed workflows,
comprehensive validation framework.
