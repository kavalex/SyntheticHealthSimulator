#!/usr/bin/env python3
"""
SyntheticHealthSimulator - Validation script.
Runs dataset validation and saves report to output/validation_report.txt.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.paths import get_version, get_data_path, get_output_dir
from validator import DatasetValidator


def main():
    """Run validation on the latest generated dataset."""
    version = get_version()
    data_path = get_data_path(version)

    if not data_path.exists():
        print(f"Data directory not found: {data_path}")
        print("Please run generation first.")
        sys.exit(1)

    validator = DatasetValidator(data_path)
    if not validator.load_data():
        sys.exit(1)

    validator.validate_all()

    # Save report to file
    output_dir = get_output_dir()
    report_path = output_dir / "validation_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        # Redirect print output to file
        import contextlib
        with contextlib.redirect_stdout(f):
            validator.print_report()

    print(f"Validation report saved to: {report_path}")

    # Also print summary to console
    passed = sum(1 for r in validator.results if r.passed)
    total = len(validator.results)
    print(f"Validation complete: {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
