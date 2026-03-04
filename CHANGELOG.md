# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-03-04

### Added
- **Dataset validator** (`validator.py`) – a standalone tool to verify that generated data meets expected statistical properties, physiological ranges, and correlations. Includes checks for file structure, demographic distributions, genetic risks, biomarker ranges, temporal trends, disease prevalence, MAR missing patterns, and cross‑dataset consistency.
- **Validation documentation** – usage instructions added to README files.

### Changed
- **Generator improvements**:
  - Fixed HDL trend: added age‑dependent decrease and adjusted coefficients to achieve realistic decline over 20 years (median change now negative).
  - Added `age_vals` extraction in `calculate_biomarkers` to correctly apply age effects.
  - Included `total_met_minutes` in `cols_to_copy` so that it appears in the biomarkers DataFrame and consequently in the aggregated dataset.
  - Ensured `sex` column is preserved in aggregated dataset by using `suffixes` in merge.
- **Updated dependencies** – versions in `pyproject.toml` aligned with current stable releases.
- **Documentation updates** – README files now include validation instructions; scientific reference refined for clarity.

### Fixed
- **KeyError on `sex_0`** – removed `"sex"` from `cols_to_copy` in biomarker calculation to prevent copying non‑existent yearly sex columns.
- **HDL‑BMI correlation** – now within expected range after coefficient tuning.
- **Missing `avg_total_met_minutes`** – now correctly computed from copied `total_met_minutes` columns.

---

## [1.0.0] - 2026-02-27

### Added
- **Initial release** of SyntheticHealthSimulator – a tool for generating synthetic medical panel data over 20 years.
- **Cohort generation** (`generator.py`): creates a baseline population with demographics, genetics, and initial biomarkers.
- **Lifestyle history simulation** using Ornstein‑Uhlenbeck processes (alcohol, cardio, strength, smoking, stress, sleep) and random stressful events.
- **Biomarker evolution** models for weight, muscle mass, HDL, total cholesterol, SBP, HbA1c with annual changes and added measurement noise.
- **Health risk calculators**: CVD (SCORE2-like), type 2 diabetes (FINDRISC-like), stroke, NAFLD, colorectal cancer, cirrhosis – all with simplified logistic models.
- **Polygenic risk** generation via log‑normal distributions for six disease categories.
- **Muscle mass modelling** with correction for BMR and BMI, and effects on biomarkers.
- **KBJU imbalance index** (0‑100) to quantify diet quality.
- **MAR missing values** mechanism (2‑15%) applied to final aggregated dataset to simulate real‑world medical data.
- **Five output datasets**:
  1. `01_cohort_baseline_v1.0.0.csv` – immutable characteristics.
  2. `02_lifestyle_history_v1.0.0.csv` – yearly lifestyle factors and true physiological values.
  3. `03_biomarkers_history_v1.0.0.csv` – annual biomarkers with noise.
  4. `04_health_risks_v1.0.0.csv` – 10‑year risks and binary outcomes.
  5. `05_aggregated_dataset_with_missing_v1.0.0.csv` – one row per patient with averaged lifestyle, final biomarkers, risks, and MAR missingness.
- **Metadata JSON** files for each dataset containing version, description, and missing value info.
- **Version management**:
  - `VERSION` file stores the current version.
  - `bump_version.py` script to increment major/minor/patch and update `generator.py`.
- **GitHub Actions workflow** (`.github/workflows/release.yml`):
  - Automatically creates a release and a ZIP archive of generated data when a tag `v*` is pushed.
- **Documentation**:
  - `README.md` / `README_ru.md` – overview, installation, usage, dataset structure.
  - `SCIENTIFIC_REFERENCE_EN.md` / `SCIENTIFIC_REFERENCE_RU.md` – detailed scientific background, formulas, coefficients, and model limitations.
- **Explicit calibration notes** in code (e.g., `INTENTIONALLY REDUCED`, `INTENTIONALLY CALIBRATED`) to highlight deviations from epidemiological data for ML feature balance.

### Known Limitations (as documented)
- Linear and additive biomarker models – simplification for educational ML use.
- Theoretical assumptions for muscle mass, sleep, KBJU index effects – not clinically validated.
- Simplified body composition model (constant bone/organ mass).
- Additional risk models (stroke, NAFLD, cancer, cirrhosis) are not calibrated against real cohorts.
- MAR missingness formula is a heuristic, not derived from real medical records.
- Seasonality not modelled – annual discretisation averages it out.

### Important Warnings
- **DO NOT USE FOR CLINICAL DECISIONS** – data is entirely synthetic.
- All coefficients are intentionally reduced to prevent feature dominance; real epidemiological effects may be stronger.
- Hard clipping is applied to biomarkers to keep values within physiologically plausible ranges (see Section 8.3 of the scientific reference).

---

[1.1.0]: https://github.com/yourusername/SyntheticHealthSimulator/releases/tag/v1.1.0
[1.0.0]: https://github.com/yourusername/SyntheticHealthSimulator/releases/tag/v1.0.0
