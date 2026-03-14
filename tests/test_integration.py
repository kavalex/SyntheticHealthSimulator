"""
Integration tests for complete pipeline
"""

from config import SimulationConfig
from generator import DataGenerator


class TestFullPipeline:
    """End-to-end pipeline tests."""

    def test_pipeline_completes(self):
        """Full pipeline should run without errors."""
        gen = DataGenerator(
            seed=42,
            config=SimulationConfig(years=5, target_n_people=50)
        )

        cohort = gen.generate_cohort(n_people=50)
        lifestyle = gen.generate_lifestyle_history(cohort, years=5)
        biomarkers = gen.calculate_biomarkers(cohort, lifestyle, years=5)
        risks = gen.calculate_health_risks(biomarkers, cohort, years=5)

        # Basic sanity checks
        assert len(cohort) == 50
        assert len(lifestyle) == 50
        assert len(biomarkers) == 50
        assert len(risks) == 50

    def test_dataframe_shapes(self):
        """DataFrames should have expected column counts."""
        gen = DataGenerator(
            seed=42,
            config=SimulationConfig(years=3, target_n_people=10)
        )

        cohort = gen.generate_cohort(n_people=10)
        # Cohort: person_id + demographics + baseline biomarkers
        assert len(cohort.columns) > 15

        lifestyle = gen.generate_lifestyle_history(cohort, years=3)
        # Lifestyle: person_id + yearly columns for multiple variables
        assert len(lifestyle.columns) > 30

        biomarkers = gen.calculate_biomarkers(cohort, lifestyle, years=3)
        # Biomarkers: person_id + yearly measurements
        assert len(biomarkers.columns) > 20

    def test_no_missing_in_critical_columns(self):
        """Critical columns should have no NaN before MAR."""
        gen = DataGenerator(
            seed=42,
            config=SimulationConfig(years=5, target_n_people=50)
        )

        cohort = gen.generate_cohort(n_people=50)
        lifestyle = gen.generate_lifestyle_history(cohort, years=5)
        biomarkers = gen.calculate_biomarkers(cohort, lifestyle, years=5)

        # Critical identifiers and baseline
        assert cohort["person_id"].notna().all()
        assert cohort["age_start"].notna().all()
        assert cohort["sex"].notna().all()

        # Biomarkers should be complete (before MAR)
        for year in range(5):
            assert biomarkers[f"weight_kg_{year}"].notna().all()
            assert biomarkers[f"bmi_{year}"].notna().all()

    def test_run_method(self):
        """Test unified run() method if available."""
        gen = DataGenerator(
            seed=42,
            config=SimulationConfig(years=3, target_n_people=20)
        )

        # Если у вас есть метод run_full_pipeline()
        if hasattr(gen, 'run_full_pipeline'):
            result = gen.run_full_pipeline(n_people=20)

            assert "cohort" in result
            assert "lifestyle" in result
            assert "biomarkers" in result
            assert "risks" in result
