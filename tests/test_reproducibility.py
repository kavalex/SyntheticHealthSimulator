"""
Tests for reproducibility and determinism
"""

import pandas as pd

from config import SimulationConfig


class TestDeterminism:
    """Ensure identical seeds produce identical results."""

    def test_cohort_reproducibility(self):
        """Same seed = same cohort."""
        from generator import DataGenerator

        gen1 = DataGenerator(seed=42)
        gen2 = DataGenerator(seed=42)

        cohort1 = gen1.generate_cohort(n_people=100)
        cohort2 = gen2.generate_cohort(n_people=100)

        pd.testing.assert_frame_equal(cohort1, cohort2)

    def test_different_seeds_different_results(self):
        """Different seeds should produce different cohorts."""
        from generator import DataGenerator

        gen1 = DataGenerator(seed=42)
        gen2 = DataGenerator(seed=43)

        cohort1 = gen1.generate_cohort(n_people=100)
        cohort2 = gen2.generate_cohort(n_people=100)

        # At least some values should differ
        assert not cohort1.equals(cohort2)

    def test_full_pipeline_reproducibility(self):
        """Entire pipeline should be reproducible."""
        from generator import DataGenerator

        def run_pipeline(seed, years=5):
            gen = DataGenerator(
                seed=seed,
                config=SimulationConfig(years=years, target_n_people=50)
            )
            cohort = gen.generate_cohort(n_people=50)
            lifestyle = gen.generate_lifestyle_history(cohort, years=years)
            biomarkers = gen.calculate_biomarkers(cohort, lifestyle, years=years)
            risks = gen.calculate_health_risks(biomarkers, cohort, years=years)
            return cohort, lifestyle, biomarkers, risks

        result1 = run_pipeline(42)
        result2 = run_pipeline(42)

        for df1, df2 in zip(result1, result2):
            pd.testing.assert_frame_equal(df1, df2)


class TestConfigImpact:
    """Test that configuration affects results."""

    def test_different_years_different_output(self):
        """Different years config should produce different trajectory lengths."""
        from generator import DataGenerator

        gen_short = DataGenerator(seed=42, config=SimulationConfig(years=3))
        gen_long = DataGenerator(seed=42, config=SimulationConfig(years=10))

        cohort = gen_short.generate_cohort(n_people=10)

        lifestyle_short = gen_short.generate_lifestyle_history(cohort, years=3)
        lifestyle_long = gen_long.generate_lifestyle_history(cohort, years=10)

        # Longer simulation should have more columns
        assert len(lifestyle_long.columns) > len(lifestyle_short.columns)
