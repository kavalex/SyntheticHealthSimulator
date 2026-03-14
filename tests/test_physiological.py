"""
Tests for physiological plausibility
"""

import pytest

from config import SimulationConfig
from generator import DataGenerator


@pytest.fixture
def full_simulation():
    """Generate complete simulation for testing."""
    gen = DataGenerator(
        seed=42,
        config=SimulationConfig(years=5, target_n_people=200)
    )
    cohort = gen.generate_cohort(n_people=200)
    lifestyle = gen.generate_lifestyle_history(cohort, years=5)
    biomarkers = gen.calculate_biomarkers(cohort, lifestyle, years=5)
    risks = gen.calculate_health_risks(biomarkers, cohort, years=5)
    return {
        "cohort": cohort,
        "lifestyle": lifestyle,
        "biomarkers": biomarkers,
        "risks": risks,
        "config": gen.config
    }


class TestBiomarkerRanges:
    """Biomarkers should stay in physiologically valid ranges."""

    def test_bmi_bounds(self, full_simulation):
        """BMI should be in 16-50 range."""
        bio = full_simulation["biomarkers"]
        config = full_simulation["config"]

        for year in range(config.years):
            bmi_col = f"bmi_{year}"
            assert bmi_col in bio.columns, f"Column {bmi_col} not found"
            assert bio[bmi_col].between(16, 50).all(), \
                f"Year {year}: BMI out of bounds, range=[{bio[bmi_col].min():.1f}, {bio[bmi_col].max():.1f}]"

    def test_hdl_bounds(self, full_simulation):
        """HDL should be in 20-100 mg/dL."""
        bio = full_simulation["biomarkers"]
        config = full_simulation["config"]

        for year in range(config.years):
            hdl_col = f"hdl_mgdl_{year}"
            assert bio[hdl_col].between(20, 100).all(), \
                f"Year {year}: HDL out of bounds"

    def test_sbp_bounds(self, full_simulation):
        """Systolic BP should be in 80-200 mmHg."""
        bio = full_simulation["biomarkers"]
        config = full_simulation["config"]

        for year in range(config.years):
            sbp_col = f"sbp_mmhg_{year}"
            assert bio[sbp_col].between(80, 200).all(), \
                f"Year {year}: SBP out of bounds"

    def test_hba1c_bounds(self, full_simulation):
        """HbA1c should be in 3.5-12.0%."""
        bio = full_simulation["biomarkers"]
        config = full_simulation["config"]

        for year in range(config.years):
            hba1c_col = f"hba1c_percent_{year}"
            assert bio[hba1c_col].between(3.5, 12.0).all(), \
                f"Year {year}: HbA1c out of bounds"


class TestTemporalBehavior:
    """Biomarkers should show realistic temporal dynamics."""

    def test_no_extreme_drift(self, full_simulation):
        """Biomarkers shouldn't drift to extremes over time."""
        bio = full_simulation["biomarkers"]
        config = full_simulation["config"]

        last_year = config.years - 1

        # Check HbA1c doesn't universally increase to diabetes
        hba1c_start = bio["hba1c_percent_0"].mean()
        hba1c_end = bio[f"hba1c_percent_{last_year}"].mean()
        drift = hba1c_end - hba1c_start

        # Allow some increase (aging) but not dramatic
        assert drift < 1.0, f"HbA1c drift {drift:.2f} too large over {config.years} years"

    def test_weight_realistic_range(self, full_simulation):
        """Weight changes should be realistic."""
        bio = full_simulation["biomarkers"]
        config = full_simulation["config"]

        last_year = config.years - 1

        weight_start = bio["weight_kg_0"]
        weight_end = bio[f"weight_kg_{last_year}"]
        change_pct = ((weight_end - weight_start) / weight_start * 100).abs()

        # 99% of population shouldn't change >50% weight
        assert (change_pct < 50).mean() > 0.99, \
            f"Unrealistic weight changes: {(change_pct >= 50).sum()} patients >50%"


class TestCorrelations:
    """Expected correlations should be present."""

    def test_hdl_total_cholesterol_realistic_correlation(self, full_simulation):
        """HDL and total cholesterol have weak realistic correlation."""
        bio = full_simulation["biomarkers"]
        config = full_simulation["config"]
        last_year = config.years - 1

        hdl = bio[f"hdl_mgdl_{last_year}"]
        tc = bio[f"total_cholesterol_mgdl_{last_year}"]
        corr = hdl.corr(tc)

        # Реалистичная слабая корреляция: -0.3 до +0.3
        assert -0.3 < corr < 0.3, (
            f"HDL-TC correlation {corr:.3f} out of realistic range (-0.3 to +0.3)"
        )


class TestCohortDemographics:
    """Baseline cohort should have expected properties."""

    def test_sex_distribution(self, full_simulation):
        """Should have both sexes represented."""
        cohort = full_simulation["cohort"]
        male_pct = (cohort["sex"] == "M").mean()
        assert 0.4 < male_pct < 0.6, f"Sex imbalance: {male_pct:.1%} male"

    def test_age_distribution(self, full_simulation):
        """Age should be in specified range."""
        cohort = full_simulation["cohort"]
        config = full_simulation["config"]

        assert cohort["age_start"].between(*config.age_range).all()
        assert config.age_range[0] < cohort["age_start"].mean() < config.age_range[1]

    def test_bmi_consistency(self, full_simulation):
        """BMI should match weight/height calculation."""
        cohort = full_simulation["cohort"]
        calculated_bmi = cohort["weight_start_kg"] / (cohort["height_cm"] / 100) ** 2
        diff = (calculated_bmi - cohort["bmi_start"]).abs()
        assert diff.max() < 0.1, f"BMI inconsistent: max diff={diff.max():.3f}"


class TestDiseaseRisks:
    """Disease risks should be in plausible ranges."""

    def test_risk_bounds(self, full_simulation):
        """Individual risks should be in [0, 1]."""
        risks = full_simulation["risks"]

        risk_cols = [c for c in risks.columns if c.endswith('_risk_10year')]
        for col in risk_cols:
            assert risks[col].between(0, 1).all(), f"{col} out of [0,1] bounds"

    def test_class_balance_reasonable(self, full_simulation):
        """Disease prevalence should not be extreme."""
        risks = full_simulation["risks"]
        config = full_simulation["config"]

        disease_cols = ['has_cvd', 'has_diabetes', 'has_stroke', 'has_nafld']

        # Адаптивный порог: для коротких симуляций допускаем 4%
        expected_min = 0.04 if config.years <= 5 else 0.08

        for col in disease_cols:
            if col in risks.columns:
                prevalence = risks[col].mean()
                assert expected_min <= prevalence < 0.50, (
                    f"{col} prevalence {prevalence:.1%} out of range "
                    f"[{expected_min:.0%}, 50%)"
                )
