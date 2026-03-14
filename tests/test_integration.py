"""
Integration tests for complete pipeline - adapted for new structure
"""
import pandas as pd
from pathlib import Path
from src.pipeline import generate
from src.utils.validators import DatasetValidator

class TestPipelineIntegration:
    def test_pipeline_completes(self):
        result = generate(n_people=50, years=5, seed=42, log_level=40)
        assert "cohort" in result
        assert "lifestyle" in result
        assert "biomarkers" in result
        assert "risks" in result
        assert "aggregated" in result
        assert len(result["cohort"]) == 50
        assert len(result["risks"]) == 50
        assert len(result["aggregated"]) == 50

    def test_reproducibility(self):
        result1 = generate(n_people=30, years=3, seed=123, log_level=40)
        result2 = generate(n_people=30, years=3, seed=123, log_level=40)
        pd.testing.assert_frame_equal(result1["cohort"], result2["cohort"])
        pd.testing.assert_frame_equal(result1["risks"], result2["risks"])

    def test_different_seeds_different_results(self):
        result1 = generate(n_people=30, years=3, seed=123, log_level=40)
        result2 = generate(n_people=30, years=3, seed=456, log_level=40)
        assert not result1["cohort"].equals(result2["cohort"])

    def test_dataframe_shapes_consistent(self):
        result = generate(n_people=20, years=3, seed=42, log_level=40)
        ids_cohort = set(result["cohort"]["person_id"])
        ids_risks = set(result["risks"]["person_id"])
        ids_agg = set(result["aggregated"]["person_id"])
        assert ids_cohort == ids_risks == ids_agg
        assert result["aggregated"].shape[1] < result["lifestyle"].shape[1]

    def test_validator_class_location(self):
        # Test that DatasetValidator is correctly imported from utils
        from src.utils.validators import DatasetValidator
        assert DatasetValidator is not None

    def test_validator_runs_on_generated_data(self, tmp_path):
        # Generate small dataset
        result = generate(n_people=20, years=3, seed=42, log_level=40, output_dir=str(tmp_path))
        
        # Run validator
        validator = DatasetValidator(tmp_path)
        assert validator.load_data()
        
        validator.validate_all()
        
        # Check results
        assert len(validator.results) > 0
        # Most checks should pass
        passed = sum(1 for r in validator.results if r.passed)
        assert passed > len(validator.results) * 0.5  # At least half should pass

